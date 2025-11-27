import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend_langgraph_rag import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata

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
    return state.get("messages", [])

# Initialize session state variables
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
    
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
    
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_documents = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1] # Display the latest threads first
selected_thread = None

# Define sidebar UI config
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID**: `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()
    
if thread_documents:
    latest_document = list(thread_documents.values())[-1]
    st.sidebar.success(
        f"Using `{latest_document.get('filename')}` "
        f"({latest_document.get('chunks')} chunks from {latest_document.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed for this chat. Upload a PDF first.")
    
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

if uploaded_pdf:
    if uploaded_pdf.name in thread_documents:
        st.sidebar.info(f"`{uploaded_pdf.name}` already indexed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name
            )
            thread_documents[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
            
st.sidebar.subheader("Previous Chats")

if not threads:
    st.sidebar.write("No past conversations found.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# Define main page UI config
st.title("Multi Utility Chatbot")

# Display chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
        
user_input = st.chat_input("Enter your query")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
        
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn"
    }
    
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        
        def stream_ai_message():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
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
            
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
                    
        ai_message = st.write_stream(stream_ai_message())
        
        if status_holder["box"] is not None:
            status_holder["box"].update(label=f"✅ Tool finished", state="complete", expanded=False)
            
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    
    document_metadata = thread_document_metadata(thread_key)
    
    if document_metadata:
        st.caption(
            f"Document indexed: {document_metadata.get('filename')}"
            f"(chunks: {document_metadata.get('chunks')}, pages: {document_metadata.get('documents')})"
        )
        
st.divider()

# Display message history for the selected thread
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)
    
    temporary_messages = []
    
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temporary_messages.append({"role": role, "content": msg.content})
        
    st.session_state["message_history"] = temporary_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()