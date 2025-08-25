import streamlit as st
from backend_db_tool_integrated import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from uuid import uuid4

def generate_thread_id():
    return str(uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    st.session_state['message_history'] = []
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    
def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config=config).values
    return state.get('messages', [])

# Initialize session state variables
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
    
add_thread(st.session_state['thread_id'])

st.set_page_config(page_title="LangGraph", page_icon="📊")

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    
st.sidebar.header('My Conversations')
    
# Display a list of all thread IDs in the sidebar
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        
        # Load previous conversation history for the selected thread
        messages = load_conversation(thread_id)

        previous_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            elif isinstance(message, ToolMessage):
                continue  # Skip tool messages
            else:
                role = 'assistant'
            previous_messages.append({"role": role, "content": message.content})  
            
        st.session_state['message_history'] = previous_messages    

# Displaying conversation history
for message in st.session_state['message_history']:
    content = message.get("content")
    # Skip if content is None or empty string
    if not content or (isinstance(content, str) and content.strip() == ""):
        continue
    
    with st.chat_message(message['role']):
        st.text(message['content'])
        
user_input = st.chat_input("Enter your message")

if user_input:
    # Add user input to message history
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable": {'thread_id': st.session_state['thread_id']},
              "metadata": {'thread_id': st.session_state['thread_id']},
              "run_name": 'chat_turn'}
    
    # with st.chat_message("assistant"):
    #     ai_message = st.write_stream(
    #         message_chunk.content for message_chunk, metadata in chatbot.stream(
    #             {"messages": [HumanMessage(content=user_input)]}, 
    #             config=CONFIG, 
    #             stream_mode="messages") 
    #         if message_chunk.content
    #     )
    
    # Add the AI message to chat history
    with st.chat_message("assistant"):
        status_holder = {"box": None} # Use a mutable status holder so that the streaming generator can modify it later
        def stream_ai_message():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Lazily create and update the status holder when any tool is invoked
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            label=f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True
                        )
                
                # Check if message chunk is an AI message and has content 
                if isinstance(message_chunk, AIMessage) and isinstance(message_chunk.content, str):
                    # yield only AI message tokens
                    yield message_chunk.content
                    
        ai_message = st.write_stream(stream_ai_message())
        
        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label=f"✅ Tool finished",
                state="complete",
                expanded=False
            )
        
    # Add assistant response to message history
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})  