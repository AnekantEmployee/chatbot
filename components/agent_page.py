import streamlit as st
from backend.agent import compiled_graph
from constants.index import AGENT_CONFIG
from langchain_core.messages import HumanMessage

def agent_page_component():
    st.title("🤖 Agent")
    
    # Display existing agent messages
    for message in st.session_state['agent_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Agent input
    agent_input = st.chat_input("Ask the agent...")

    if agent_input:
        # Add user message to agent history
        st.session_state['agent_history'].append({'role': 'user', 'content': agent_input})
        with st.chat_message('user'):
            st.markdown(agent_input)
        
        # Get agent response
        agent_result = compiled_graph.invoke({'messages': HumanMessage(content=agent_input)}, config=AGENT_CONFIG)
        agent_response = agent_result['messages'][-1].content
        
        st.session_state['agent_history'].append({'role': 'ai', 'content': agent_response})
        with st.chat_message('ai'):
            st.markdown(agent_response)
