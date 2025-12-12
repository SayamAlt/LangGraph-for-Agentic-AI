from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated, TypedDict
from langchain_core.tools import tool
import requests, os

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize LLM
llm = ChatOpenAI()

# Create a get_stock_price tool
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
        
# Create a purchase_stock tool
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
        Simulate the purchase of a given quantity of a stock symbol.
        
        HUMAN-IN-THE-LOOP:
        Before continuing the purchase, this tool will interrupt and prompt the user to confirm all relevant stock information and price before continuing the purchase.
        Wait for a human decision ("Yes" / "No)
    """
    # This pauses the graph execution and returns the control back to the user
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (Yes/No)")
    
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Successfully purchased {quantity} shares of {symbol}",
            "symbol": symbol,
            "quantity": quantity
        }
    else:
        return {
            "status": "failure",
            "message": f"Purchase of {quantity} shares of {symbol} canceled",
            "symbol": symbol,
            "quantity": quantity
        }
        
# Create a list of tools
tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# Define chat state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# Define a chat node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create a tool node
tool_node = ToolNode(tools=tools)

# Initialize a checkpointer for persistence during interrupts
checkpointer = MemorySaver()

# Define the graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# Simulate a conversation
if __name__ == "__main__":
    # Use a fixed thread ID for this conversation
    thread_id = "thread-1"
    
    while True:
        user_input = input("Enter your message: ")
        
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Goodbye!")
            break
    
        # Build initial state for this chat turn
        initial_state = {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
        
        # Invoke the graph
        response = chatbot.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        
        # Check for HITL interrupt from purchase_stock
        interrupts = response.get("__interrupt__", [])
        
        if interrupts:
            # Get the interrupt message
            interrupt_message = interrupts[0].value
            print(f"HITL interrupt: {interrupt_message}")
            
            decision = input("Your decision: (Yes/No) ").strip().lower()
            
            # Resume graph execution with the human decision ("yes", "no", anything else)
            response = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}}
            )
            
        # Get the latest message from the assistant
        messages = response["messages"]
        last_message = messages[-1]
        print(f"Assistant: {last_message.content}\n")