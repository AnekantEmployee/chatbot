from dotenv import load_dotenv
from constants.index import MODEL_NAME
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model=MODEL_NAME)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)

def chat(state: ChatState) -> ChatState:
    messages = state['messages']
    response = model.invoke(messages)
    
    return {'messages': [response]}

graph.add_node('chat', chat)

graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)