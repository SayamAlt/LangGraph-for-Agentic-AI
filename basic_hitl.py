from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? Yes/No"
    })
    
    if decision["approved"].lower() == "no":
        return {"messages": [AIMessage(content="User rejected the question")]}

    else:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
# Create a state graph
graph = StateGraph(ChatState)

# Add nodes to the graph
graph.add_node('chat', chat_node)

# Add edges to the graph
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

# Create checkpointer for persistence during interrupts
checkpointer = MemorySaver()

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)
print(chatbot)

# Create a new thread ID for this conversation
config = {
    "configurable": {
        "thread_id": "1234"
    }
}

# Define an initial state
initial_state = {
    "messages": [
        ("user", "Explain gradient descent in very simple terms.")
    ]
}

# Invoke the chatbot
response = chatbot.invoke(initial_state, config=config)

print(response)

interrupt_message = response["__interrupt__"][0].value

user_input = input(f"\nBackend message - {interrupt_message} \n Approve this question? (Yes/No): ")

# Resume the graph execution with the approval decision
final_response = chatbot.invoke(
    Command(resume={"approved": user_input}),
    config=config
)

print(final_response)