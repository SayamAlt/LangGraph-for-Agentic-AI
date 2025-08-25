from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3, requests, os

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

llm = ChatOpenAI()

# Create a web search tool
web_search = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: Annotated[str, "add, subtract, multiply, divide"]) -> float:
    """
        A custom calculator function that performs basic arithmetic operations on two numbers.
        Supported operations are: add, subtract, multiply, divide.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num

        return {"first_number": first_num, "second_number": second_num, "operation": operation, "result": result}
    except Exception as e:
        print(f"Error occurred: {e}")
        
@tool
def get_stock_price(symbol: str) -> dict:
    """
        Fetches the latest stock price for a given symbol (e.g. 'AAPL', 'MSFT', etc.) 
        using Alpha Vantage's API.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        return response.json() 
    except Exception as e:
        print(f"Error occurred: {e}")
       
# Create a list of tools
tools = [web_search, calculator, get_stock_price]

# Bind LLM to tools
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    # Extract user query from state
    user_query = state['messages']
    # Produce a response using LLM
    response = llm_with_tools.invoke(user_query)
    # Add response to state
    return {'messages': [response]}

tool_node = ToolNode(tools=tools) # Node that executes tool calls

# Create a state graph
graph = StateGraph(ChatState)

# Add nodes to the graph
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# Create a checkpoint saver
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# config = {'configurable': {'thread_id': 'thread-1'}}

# response = chatbot.invoke({'messages': [HumanMessage(content="Hi, my name is Sayam")]}, config=config, stream_mode='messages')

# # print([message_chunk.content for message_chunk, metadata in response if message_chunk.content])
# print(chatbot.get_state(config=config).values)

def retrieve_all_threads():
    all_threads = set()
    
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)