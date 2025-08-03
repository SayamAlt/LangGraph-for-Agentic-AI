import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

# {'role': 'user', 'content': 'Hi'}
# {'role': 'assistant', 'content': 'Hello'}

st.set_page_config(page_title="LangGraph", page_icon="📊")
st.title("LangGraph Chatbot")

config = {"configurable": {"thread_id": "1"}}

# with st.chat_message("user"): # Avatar parameters are optional
#     st.text("Hi, my name is Sayam.")
    
# with st.chat_message("assistant"):
#     st.text("Hello Sayam, how can I help you today?")
    
# with st.chat_message("user"):
#     st.text("Can you tell me my name?")
    
# user_input = st.chat_input("Enter your message")

# if user_input:
#     with st.chat_message("user"):
#         st.text(user_input)

# st.session_state -> dict -> key-value pairs

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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
      
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    
    # Extract assistant response
    ai_message = response['messages'][-1].content
    
    # Add assistant response to message history
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})  
    
    with st.chat_message("assistant"):
        st.text(ai_message)