from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import asyncio,  aiosqlite, requests, threading, os
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """ Schedule a coroutine on the backend async event loop """
    return _submit_async(coro)

# Initialize LLM
llm = ChatOpenAI()

# Initialize tools
search_tool = DuckDuckGoSearchRun(region="us-en")

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
        
client = MultiServerMCPClient(
    {
        "arithmetic": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/sayamkumar/Desktop/Data Science/MCP/math-mcp-server/main.py"]
        },
        "expense_tracker": {
            "transport": "streamable_http", # if this fails, try "sse"
            "url": "https://nearby-chocolate-toucan.fastmcp.app/mcp"
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []
    
mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]

# Bind LLM to tools if available
llm_with_tools = llm.bind_tools(tools) if tools else llm

# Define chat state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# Nodes
async def chat_node(state: ChatState):
    """ LLM node that may answer or request a tool call. """
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools=tools) if tools else None

# Define checkpointer
async def _init_checkpointer():
    connection = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn=connection)

checkpointer = run_async(_init_checkpointer())

# Define state graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# Helper functions
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())