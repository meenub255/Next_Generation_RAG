import os
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str 
    documents: List

web_tool = TavilySearch(max_results=3)

# Connect to OpenAI
openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
json_parser = JsonOutputParser()

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""You are a strict grader assessing relevance. 
        Document: {context} 
        Question: {question} 
        
        You must output ONLY valid JSON with a single key 'score' and a value of either 'yes' or 'no'. Do not include any other text.
        \n{format_instructions}""",
        input_variables=["context", "question"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )
    
    chain = prompt | openai_llm | json_parser
    
    filtered_docs = []
    web_search = "No"
    
    for d in documents:
        try:
            grade = chain.invoke({"question": question, "context": d.page_content})
            if grade.get('score').lower() == 'yes':
                filtered_docs.append(d)
            else:
                web_search = "Yes"
        except Exception as e:
            print(f"Grading error: {e}")
            web_search = "Yes"
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    question = state["question"]
    
    prompt = PromptTemplate(
        template="""Rewrite the following user question into a highly optimized search engine query. 
        Output ONLY the rewritten query, nothing else.
        Question: {question}""", 
        input_variables=["question"]
    )
    
    chain = prompt | openai_llm
    better_q = chain.invoke({"question": question}).content.strip()
    return {"question": better_q}

def web_search_node(state):
    response = web_tool.invoke({"query": state["question"]})
    
    web_results = []
    
    # Handle the new dictionary format (extracts the 'results' list)
    if isinstance(response, dict) and "results" in response:
        for d in response["results"]:
            web_results.append(Document(page_content=d.get("content", "")))
            
    # Fallback just in case LangChain reverts to a flat list
    elif isinstance(response, list):
        for d in response:
            if isinstance(d, dict):
                web_results.append(Document(page_content=d.get("content", "")))
            
    return {"documents": state["documents"] + web_results}

def generate(state):
    return {"generation": "Final generation logic will be handled in main.py"}

workflow = StateGraph(GraphState)

# NOTE: No 'retrieve' node here — documents come pre-fetched from HyDE in main.py.
# Routing starts directly at grade_documents so passed documents are not wiped.
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate", generate)

workflow.add_edge(START, "grade_documents")

def decide_to_generate(state):
    return "transform_query" if state["web_search"] == "Yes" else "generate"

workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()