import streamlit as st
from backend.llm import chatbot
from constants.index import CONFIG
from langchain_core.messages import HumanMessage

def chatbot_page_component():
    st.title("💬 Chatbot")
    
    # Display existing chat messages
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # User input
    user_input = st.chat_input("Type here")

    if user_input:
        # Add user message
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)
            
        # Get chatbot response
        result = chatbot.invoke({'messages': HumanMessage(content=user_input)}, config=CONFIG)
        
        response = result['messages'][-1].content
        st.session_state['chat_history'].append({'role': 'ai', 'content': response})
        with st.chat_message('ai'):
            st.markdown(response)