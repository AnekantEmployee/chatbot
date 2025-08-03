from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

load_dotenv()

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    # Remove retriever from state to avoid serialization issues

def create_retriever(text: str):
    """Create a retriever from the given text"""
    try:
        # Ensure we have an event loop for async operations
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        # Fallback: return a simple text-based retriever

# Global variable to store retriever (not in state)
_current_retriever = None

def set_current_retriever(retriever):
    """Set the current retriever globally"""
    global _current_retriever
    _current_retriever = retriever

def get_current_retriever():
    """Get the current retriever"""
    global _current_retriever
    return _current_retriever
    """Create a simple text-based retriever as fallback"""
    from langchain.schema import Document
    
    class SimpleRetriever:
        def __init__(self, text):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.chunks = splitter.create_documents([text])
        
        def invoke(self, query):
            # Simple keyword-based retrieval
            query_lower = query.lower()
            scored_chunks = []
            
            for chunk in self.chunks:
                content_lower = chunk.page_content.lower()
                score = sum(1 for word in query_lower.split() if word in content_lower)
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            # Sort by score and return top 4
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            return [chunk for score, chunk in scored_chunks[:4]]
    
    return SimpleRetriever(text)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_context(state: RAGState) -> RAGState:
    """Retrieve relevant context based on the user's question"""
    messages = state['messages']
    retriever = get_current_retriever()  # Get from global variable
    
    # Get the last human message as the question
    question = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            question = message.content
            break
    
    if not question or not retriever:
        return {'context': ""}
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    return {'context': context}

def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer based on the context and question"""
    messages = state['messages']
    context = state.get('context', "")
    
    # Get the last human message as the question
    question = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            question = message.content
            break
    
    if not question:
        return {'messages': [AIMessage(content="I couldn't find a question to answer.")]}
    
    # Create the prompt
    prompt = f"""
    You're a helpful assistant for document Q&A.
    Answer only from the provided context. Don't use external knowledge.
    If the context is not sufficient, say "I don't have enough information in the document to answer this question."

    Context: {context}
    
    Question: {question}
    
    Answer in English:
    """
    
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {'messages': [response]}

# Create the RAG graph
def create_rag_graph():
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node('retrieve', retrieve_context)
    graph.add_node('generate', generate_answer)
    
    # Add edges
    graph.add_edge(START, 'retrieve')
    graph.add_edge('retrieve', 'generate')
    graph.add_edge('generate', END)
    
    # Compile with checkpointer
    checkpointer = InMemorySaver()
    compiled_rag = graph.compile(checkpointer=checkpointer)
    
    return compiled_rag

# Create the compiled RAG graph
rag_graph = create_rag_graph()