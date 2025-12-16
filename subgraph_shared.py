from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class ParentState(TypedDict):
    question: str
    english_answer: str
    translated_answer: str
    
parent_llm = ChatOpenAI(model="gpt-4o-mini")
translator_llm = ChatOpenAI(model="gpt-4o")

def translate_text(state: ParentState):
    prompt = f"""
        Translate the following text to Spanish.
        Keep it natural and clear. Return only the translated text, nothing else.
        
        Text:
        {state['english_answer']}
    """.strip()
    
    translated_text = translator_llm.invoke(prompt).content
    return {'translated_answer': translated_text}

graph = StateGraph(ParentState)

graph.add_node('translate_text', translate_text)

graph.add_edge(START, 'translate_text')
graph.add_edge('translate_text', END)

translator_graph = graph.compile()

def generate_answer(state: ParentState):
    prompt = f"""
        You are a helpful assistant. Answer eloquently.\n
        
        Question: {state['question']}
    """.strip()
    
    answer = parent_llm.invoke(prompt).content
    return {'english_answer': answer}

parent_graph = StateGraph(ParentState)

parent_graph.add_node("generate_answer", generate_answer)
parent_graph.add_node("translate_answer", translator_graph)

parent_graph.add_edge(START, "generate_answer")
parent_graph.add_edge("generate_answer", "translate_answer")
parent_graph.add_edge("translate_answer", END)

parent_workflow = parent_graph.compile()

result = parent_workflow.invoke({"question": "Which company has the maximum market cap in the world?"})
print(result)