import streamlit as st
from components.rag_page import rag_page_component
from components.agent_page import agent_page_component
from components.chatbot_page import chatbot_page_component

# Page configuration
st.set_page_config(page_title="Chatbot App", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select Tab", ["Chatbot", "Agent", "RAG"])

CONFIG = {'configurable': {'thread_id': '1'}}
AGENT_CONFIG = {'configurable': {'thread_id': 'agent_1'}}
RAG_CONFIG = {'configurable': {'thread_id': 'rag_1'}}

# Initialize session states for all tabs
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'agent_history' not in st.session_state:
    st.session_state['agent_history'] = []
if 'rag_history' not in st.session_state:
    st.session_state['rag_history'] = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state['pdf_uploaded'] = False
if 'pdf_content' not in st.session_state:
    st.session_state['pdf_content'] = ""
if 'pdf_filename' not in st.session_state:
    st.session_state['pdf_filename'] = ""
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None


if tab == "Chatbot":
    chatbot_page_component()
    
elif tab == "Agent":
    agent_page_component()

elif tab == "RAG":
    rag_page_component()

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("App Info")
    st.write("**Chatbot**: General conversation")
    st.write("**Agent**: Advanced AI assistant")
    st.write("**RAG**: Document-based Q&A")
    
    if st.session_state['pdf_uploaded']:
        st.write(f"📄 **Current PDF**: {st.session_state['pdf_filename']}")
        st.write(f"📊 **Document length**: {len(st.session_state['pdf_content'])} characters")
        st.write("🔍 **Status**: Retriever ready")