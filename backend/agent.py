from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import time
import warnings

# LangChain Agent specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Suppress the ddgs warning
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*has been renamed.*")

# Load environment variables
load_dotenv()

class EnhancedSearchAgent:
    """Enhanced search agent with detailed results and error handling"""
    
    def __init__(self):
        # Configure search tool with better settings
        self.search_tool = DuckDuckGoSearchRun(
            max_results=8,
            region='wt-wt',
            time='d',
            safesearch='moderate'
        )
        
        # Configure model
        self.model = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0.1
        )
        
        # Create agent
        self.prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.model, [self.search_tool], self.prompt)
        
        # Configure executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.search_tool],
            verbose=True,
            max_iterations=4,
            max_execution_time=45,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
    
    def format_detailed_results(self, raw_search_data):
        """Format raw search data into readable results"""
        if not raw_search_data:
            return "No detailed search data available."
        
        # Split and format results
        lines = raw_search_data.strip().split('…')
        formatted_results = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and len(line) > 25:
                if not line.endswith('.'):
                    line += '...'
                formatted_results.append(f"📰 Result {i}: {line}")
        
        return "\n\n".join(formatted_results) if formatted_results else raw_search_data
    
    def search_with_details(self, query: str) -> str:
        """Enhanced search with detailed results"""
        try:
            print(f"\n🔍 Searching for: {query}")
            
            # Execute search with retry logic
            max_retries = 2
            result = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = self.agent_executor.invoke({'input': query})
                    break
                except Exception as e:
                    if attempt < max_retries and any(word in str(e).lower() for word in ["timeout", "network", "connection"]):
                        print(f"🔄 Retry {attempt + 1}/{max_retries} in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        raise e
            
            # Extract results
            final_answer = result.get('output', 'No summary available')
            
            # Get detailed search data from intermediate steps
            raw_search_data = None
            intermediate_steps = result.get('intermediate_steps', [])
            
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(action, 'tool') and action.tool == 'duckduckgo_search':
                        raw_search_data = str(observation)
                        break
            
            # Format comprehensive response
            response_parts = []
            response_parts.append(f"🎯 **SUMMARY:**\n{final_answer}")
            
            if raw_search_data:
                detailed_results = self.format_detailed_results(raw_search_data)
                response_parts.append(f"\n📊 **DETAILED SEARCH RESULTS:**\n{detailed_results}")
            else:
                response_parts.append("\nℹ️ No detailed search data captured.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Search Error: {error_msg}")
            
            # Provide specific error responses
            if "timeout" in error_msg.lower():
                return "⏰ Search timed out. The service may be busy. Please try again in a moment."
            elif "duckduckgo" in error_msg.lower():
                return "🔍 Search service temporarily unavailable. Please try again later."
            else:
                return f"❌ Search failed: {error_msg}"

# --- LangGraph State Definition ---
class AgentGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the enhanced agent
enhanced_agent = EnhancedSearchAgent()

# --- LangGraph Node Function ---
def run_enhanced_agent(state: AgentGraphState) -> AgentGraphState:
    """Enhanced node function with detailed search results"""
    messages = state['messages']
    user_query = messages[-1].content

    print(f"\n--- 🚀 Enhanced Agent Processing: '{user_query}' ---")
    
    try:
        # Use enhanced search
        agent_response_content = enhanced_agent.search_with_details(user_query)
        
    except Exception as e:
        error_msg = str(e)
        print(f"--- ❌ Agent Error: {error_msg} ---")
        
        # Fallback responses
        if "timeout" in error_msg.lower():
            agent_response_content = "⏰ I'm experiencing network delays. The search is taking longer than expected. Please try again in a few moments."
        elif "duckduckgo" in error_msg.lower():
            agent_response_content = "🔍 The search service is temporarily unavailable. This could be due to high traffic or maintenance. Please try again later."
        else:
            agent_response_content = f"❌ An unexpected error occurred: {error_msg}. Please try rephrasing your request."

    print(f"--- ✅ Agent Response Ready ---\n")
    return {'messages': [AIMessage(content=agent_response_content)]}

# --- Build Enhanced LangGraph ---
def create_enhanced_graph():
    """Create and compile the enhanced graph"""
    graph = StateGraph(AgentGraphState)
    graph.add_node('enhanced_agent_node', run_enhanced_agent)
    graph.add_edge(START, 'enhanced_agent_node')
    graph.add_edge('enhanced_agent_node', END)
    
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

compiled_graph = create_enhanced_graph()
