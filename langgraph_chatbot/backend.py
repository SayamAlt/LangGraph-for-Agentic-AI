from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    # Extract user query from state
    user_query = state['messages']
    # Produce a response using LLM
    response = llm.invoke(user_query)
    # Add response to state
    return {'messages': [response]}

# Create a state graph
graph = StateGraph(ChatState)

# Add nodes to the graph
graph.add_node('chat_node', chat_node)

# Add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Create a checkerpoint saver
checkpointer = InMemorySaver()

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)
    