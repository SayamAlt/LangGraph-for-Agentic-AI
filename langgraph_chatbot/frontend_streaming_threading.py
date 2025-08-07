import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
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
    st.session_state['chat_threads'] = []
    
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
            else:
                role = 'assistant'
            previous_messages.append({"role": role, "content": message.content})  
            
        st.session_state['message_history'] = previous_messages    

# Displaying conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
        
user_input = st.chat_input("Enter your message")

if user_input:
    # Add user input to message history
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}
    
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=CONFIG, 
                stream_mode="messages") 
            if message_chunk.content
        )
        
    # Add assistant response to message history
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})  