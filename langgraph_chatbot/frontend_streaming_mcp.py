import streamlit as st
from backend_langgraph_mcp import chatbot, retrieve_all_threads, submit_async_task
import queue
from uuid import uuid4
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# Utility functions
def generate_thread_id():
    return str(uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values
    # Check if messages key exists in state values, if not, return an empty list
    return state.get("messages", [])

# Initialize session state variables
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# Configure sidebar
st.set_page_config(page_title="LangGraph MCP Chatbot", page_icon="📊")
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Display a list of all thread IDs in the sidebar and load previous conversation history
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = []
        
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        
        st.session_state["message_history"] = temp_messages
    
# Display previous conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
   
# Take user input     
user_input = st.chat_input("Type here...")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
        
    # Define chat config
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn"
    }
    
    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so that generator can set or modify it
        status_holder = {"box": None}
        
        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()
            
            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages"
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as e:
                    event_queue.put(("error", e))
                finally:
                    event_queue.put(None)
            
            submit_async_task(run_stream())
            
            while True:
                item = event_queue.get()
                
                if item is None:
                    break
                
                message_chunk, metadata = item
                
                if message_chunk == "error":
                    raise metadata
                
                # Lazily create and update the status holder when any tool is invoked
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            label=f"🔧 Using `{tool_name}`…",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True
                        )
                        
                # Stream only assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
                    
        ai_message = st.write_stream(ai_only_stream())
        
        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label=f"✅ Tool finished",
                state="complete",
                expanded=False
            )
            
    # Save assistant response to message history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})