from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List

# Connect to OpenAI
openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

json_parser = JsonOutputParser()

system_prompt = """You are an expert researcher. Break down the user's complex query 
into simple, atomic sub-queries that a search engine can answer.

You MUST output ONLY a valid JSON object with a single key 'questions' containing a list of strings.
Example: {{"questions": ["sub-query 1", "sub-query 2"]}}

User Query: {query}
\n{format_instructions}"""

prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["query"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

planner = prompt | openai_llm | json_parser

def plan_query(query: str) -> List[str]:
    try:
        result = planner.invoke({"query": query})
        return result.get("questions", [query])
    except Exception as e:
        print(f"Decomposition failed, falling back to original query: {e}")
        return [query]