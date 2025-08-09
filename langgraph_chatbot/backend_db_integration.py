from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import sqlite3

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

# Create a SQLite database
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) # check_same_thread=False is required for SQLite as our LangGraph chatbot is multi-threaded for multiple conversations

# Create a checkpoint saver
checkpointer = SqliteSaver(conn=conn)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# Test 
# CONFIG = {"configurable": {"thread_id": "thread-2"}}

# response = chatbot.invoke({"messages": [HumanMessage(content="What is the capital of India? Acknowledge my name while answering")]}, config=CONFIG)

# print(response)

# Extract number of threads in the chatbot database
# print(checkpointer.list(None)) # We need to pass in a config to list the threads

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    # print("Number of unique threads:", len(all_threads))
    # print("Unique threads:", list(all_threads))
    return list(all_threads)