from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import time
import warnings

# LangChain Agent specific imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Try to import search tools with fallback
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Search tools not available: {e}")
    SEARCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*has been renamed.*")

# Load environment variables
load_dotenv()

class EnhancedSearchAgent:
    """Enhanced search agent with detailed results and error handling"""
    
    def __init__(self):
        # Configure model
        self.model = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0.1
        )
        
        if SEARCH_AVAILABLE:
            try:
                # Configure search tool with better settings
                self.search_tool = DuckDuckGoSearchRun(
                    max_results=5,
                    region='wt-wt',
                    safesearch='moderate'
                )
                
                # Create agent
                self.prompt = hub.pull("hwchase17/react")
                agent = create_react_agent(self.model, [self.search_tool], self.prompt)
                
                # Configure executor
                self.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=[self.search_tool],
                    verbose=False,
                    max_iterations=3,
                    max_execution_time=30,
                    return_intermediate_steps=True,
                    handle_parsing_errors=True
                )
                self.search_enabled = True
                print("✅ Search agent initialized successfully")
            except Exception as e:
                print(f"⚠️ Search initialization failed: {e}")
                self.search_enabled = False
        else:
            self.search_enabled = False
            print("⚠️ Search functionality disabled - dependencies not available")
    
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
        """Enhanced search with detailed results or fallback to direct LLM response"""
        if not self.search_enabled:
            return self.direct_llm_response(query)
        
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
                response_parts.append("\nℹ️ Search completed successfully.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Search Error: {error_msg}")
            
            # Fallback to direct LLM response
            return self.direct_llm_response(query)
    
    def direct_llm_response(self, query: str) -> str:
        """Direct LLM response when search is not available"""
        try:
            prompt = f"""
            You are a helpful AI assistant. The user has asked: "{query}"
            
            Please provide a comprehensive and helpful response based on your knowledge.
            If this appears to be a question that would benefit from current information,
            please mention that your knowledge has a cutoff date and recommend that the user
            search for the most recent information.
            
            Format your response clearly and helpfully.
            """
            
            response = self.model.invoke([HumanMessage(content=prompt)])
            
            return f"""🤖 **AI ASSISTANT RESPONSE:**

{response.content}

ℹ️ *Note: Search functionality is currently unavailable. This response is based on my training data. For the most current information, please verify with recent sources.*"""
            
        except Exception as e:
            return f"❌ I apologize, but I'm experiencing technical difficulties: {str(e)}"

# --- LangGraph State Definition ---
class AgentGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the enhanced agent
try:
    enhanced_agent = EnhancedSearchAgent()
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")
    enhanced_agent = None

# --- LangGraph Node Function ---
def run_enhanced_agent(state: AgentGraphState) -> AgentGraphState:
    """Enhanced node function with detailed search results"""
    messages = state['messages']
    user_query = messages[-1].content

    print(f"\n--- 🚀 Enhanced Agent Processing: '{user_query}' ---")
    
    if enhanced_agent is None:
        agent_response_content = "❌ Agent is not properly initialized. Please check the configuration."
    else:
        try:
            # Use enhanced search or fallback
            agent_response_content = enhanced_agent.search_with_details(user_query)
            
        except Exception as e:
            error_msg = str(e)
            print(f"--- ❌ Agent Error: {error_msg} ---")
            
            # Final fallback
            agent_response_content = f"""❌ I encountered an error while processing your request: {error_msg}

🤖 **Basic Response:**
I understand you're asking about: "{user_query}"

Unfortunately, I'm experiencing technical difficulties with my advanced features. For the best results, please:
1. Try rephrasing your question
2. Check for any recent updates or current information online
3. Contact support if this issue persists

I apologize for the inconvenience."""

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