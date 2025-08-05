import time
import warnings
from dotenv import load_dotenv
from langchain.tools import tool
from constants.index import MODEL_NAME
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_tavily import TavilySearch
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
except ImportError as e:
    print(f"Warning: Search tools not available: {e}")

warnings.filterwarnings("ignore")
load_dotenv()


@tool
def get_current_time():
    """Getting current date and time"""
    return time.ctime()


class StreamingSearchChain:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)

        try:
            self.search_tool = TavilySearch(max_results=4)
            print("✅ Search agent initialized")
        except Exception as e:
            print(f"⚠️ Search initialization failed: {e}")

    def execute_search_chain(self, query: str, update_callback=None):
        """Execute search chain: Search -> Stream -> Generate Response"""

        # Step 1: Show search query
        if update_callback:
            update_callback("query", f"🔍 Searching: {query}")

        try:
            # Step 2: Execute search tool
            if update_callback:
                update_callback("status", "🌐 Querying search engines...")

            # Get search results
            search_results_raw = self.search_tool.invoke({"query": query})

            # Step 3: Process and stream sources
            sources = []
            search_results = []
            search_context = ""

            # Extract results from the response structure
            results_list = []
            if isinstance(search_results_raw, dict) and "results" in search_results_raw:
                results_list = search_results_raw["results"]
            elif isinstance(search_results_raw, list):
                results_list = search_results_raw

            for item in results_list:
                for item in results_list:
                    if isinstance(item, dict):
                        title = item.get("title", "No title")
                        url = item.get("url", "")
                        content = item.get("content", "")

                        # Store processed results
                        search_results.append(
                            {
                                "title": title,
                                "url": url,
                                "content": (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                ),
                                "score": item.get("score", 0),
                            }
                        )

                        # Build context for agent
                        search_context += (
                            f"Title: {title}\nURL: {url}\nContent: {content}\n\n"
                        )

                        if title != "No title":
                            source = f"{title}"
                            if url:
                                source += f" ({url})"
                            sources.append(source)

                            # Stream each source
                            if update_callback:
                                update_callback("source", source)
                                time.sleep(0.3)  # Streaming delay

            # Step 4: Generate response using search context
            if update_callback:
                update_callback("status", "🤖 Generating response...")

            # Create prompt with search context
            prompt = f"""
            Based on the following search results, provide a comprehensive answer to the query: "{query}"
            
            Search Results:
            {search_context}
            
            Please provide a well-structured answer based on this information.
            """

            response = self.model.invoke(prompt)
            agent_answer = response.content

            # Step 5: Show final answer
            if update_callback:
                update_callback("answer", agent_answer)

            return {
                "search_query": query,
                "agent_answer": agent_answer,
                "search_results": search_results[:4],
                "sources": sources[:4],
                "search_context": search_context,
                "error": None,
            }

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            if update_callback:
                update_callback("error", error_msg)

            return {
                "search_query": query,
                "agent_answer": error_msg,
                "search_results": [],
                "sources": [],
                "search_context": "",
                "error": str(e),
            }


# State and Graph Setup
class AgentGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    search_data: Dict[str, Any]


search_chain = StreamingSearchChain()


def run_search_chain(state: AgentGraphState) -> AgentGraphState:
    messages = state["messages"]
    user_query = messages[-1].content

    print(f"\n🔍 Processing: '{user_query}'")

    # Execute the search chain (without callback for graph execution)
    search_data = search_chain.execute_search_chain(user_query)

    return {
        "messages": [AIMessage(content=search_data["agent_answer"])],
        "search_data": search_data,
    }


# Build Graph
graph = StateGraph(AgentGraphState)
graph.add_node("search_chain", run_search_chain)
graph.add_edge(START, "search_chain")
graph.add_edge("search_chain", END)

compiled_graph = graph.compile(checkpointer=InMemorySaver())
