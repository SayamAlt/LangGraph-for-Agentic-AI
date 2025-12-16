from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

class TranslationState(TypedDict):
    input_text: str
    translated_text: str
    
def translate_text(state: TranslationState):
    prompt = f"""
        Translate the following text to Spanish.
        Keep it natural and clear. Return only the translated text, nothing else.
        
        Text:
        {state['input_text']}
    """.strip()
    
    translated_text = llm.invoke(prompt).content
    return {'translated_text': translated_text}

subgraph = StateGraph(TranslationState)

subgraph.add_node("translate_text", translate_text)

subgraph.add_edge(START, "translate_text")
subgraph.add_edge("translate_text", END)

translation_workflow = subgraph.compile()

class ParentState(TypedDict):
    question: str
    english_answer: str
    translated_answer: str
    
def generate_answer(state: ParentState):
    prompt = f"""
        You are a helpful assistant. Answer eloquently.\n
        
        Question: {state['question']}
    """.strip()
    
    answer = llm.invoke(prompt).content
    return {'english_answer': answer}

def translate_answer(state: ParentState):
    # Invoke the translation subgraph
    translated_answer = translation_workflow.invoke({"input_text": state["english_answer"]})
    return {"translated_answer": translated_answer["translated_text"]}

parent_graph = StateGraph(ParentState)

parent_graph.add_node("generate_answer", generate_answer)
parent_graph.add_node("translate_answer", translate_answer)

parent_graph.add_edge(START, "generate_answer")
parent_graph.add_edge("generate_answer", "translate_answer")
parent_graph.add_edge("translate_answer", END)

parent_workflow = parent_graph.compile()

result = parent_workflow.invoke({"question": "What is the capital of France?"})
print(result)