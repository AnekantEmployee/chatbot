import os
import tempfile
import streamlit as st
from backend.rag import rag_graph
from constants.index import RAG_CONFIG_ID
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader


def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and extract text using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        context_text = "\n\n".join(doc.page_content for doc in docs)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return context_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def process_rag_query(question):
    """Process RAG query using LangGraph"""
    try:
        # Create initial state with just the question
        initial_state = {
            'messages': [HumanMessage(content=question)],
            'context': ""
        }
        
        # Invoke the RAG graph
        result = rag_graph.invoke(initial_state, config={'configurable': {'thread_id': RAG_CONFIG_ID}})
        
        # Extract the response
        response = result['messages'][-1].content
        return response
    except Exception as e:
        return f"Error processing RAG query: {str(e)}"
