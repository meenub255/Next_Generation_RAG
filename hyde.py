from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.llms.openai import OpenAI

def build_hyde_engine(index):
    # Connect directly to OpenAI
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    hyde = HyDEQueryTransform(include_original=True)
    base_query_engine = index.as_query_engine(similarity_top_k=5)
    hyde_engine = TransformQueryEngine(base_query_engine, query_transform=hyde)
    
    return hyde_engine