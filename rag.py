from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

loader = PyPDFLoader("intro-to-ml.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def rag_tool(query: str):
    """
        Retrieve relevant information from the PDF and answer the query.
        Use this tool when the user asks factual/conceptual questions that might be answered from the stored documents.
    """
    result = retriever.invoke(query)
    
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    
    return {
        "query": query,
        "context": context,
        "metadata": metadata
    }
    
tools = [rag_tool]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(content="Using the PDF notes, explain how to split a node in a Decision Tree")
        ]
    }
)

print(result["messages"][-1].content)