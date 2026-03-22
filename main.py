import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Generator, Callable, Optional
from dotenv import load_dotenv

load_dotenv()

# 1. SET YOUR API KEYS HERE (or use environment variables)
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["TAVILY_API_KEY"] = "tvly-..."

from langchain_openai import ChatOpenAI
from hyde import build_hyde_engine, Settings
from query_decomposition import plan_query
from corrective_rag import app as crag_app
from reranker import Reranker
from dynamic_prompting import LearningManager
from llama_index.core import VectorStoreIndex
# --- FIX: Separate LlamaIndex and LangChain Document classes ---
from llama_index.core import Document as LlamaDocument
from langchain_core.documents import Document as LangchainDocument

class SelfHealingRAGSystem:
    def __init__(self):
        print("Initializing OpenAI-powered Self-Healing RAG...")
        self.reranker = Reranker()
        self.learning_manager = LearningManager()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        print("Creating vector database...")
        self.vector_index = VectorStoreIndex.from_documents([
            LlamaDocument(text="Corrective RAG (CRAG) improves retrieval quality by adding a lightweight relevance grader that evaluates each retrieved document. Documents scored as irrelevant are filtered out, and a web search fallback is triggered to supplement missing context."),
            LlamaDocument(text="CRAG stands for Corrective Retrieval-Augmented Generation. It is a self-correcting RAG framework that assesses retrieved document quality using an LLM grader and dynamically corrects the retrieval by using web search when local documents are deemed insufficient."),
            LlamaDocument(text="HyDE (Hypothetical Document Embeddings) is a retrieval technique that generates a hypothetical answer to the query and uses its embedding to find semantically similar documents in the vector store, improving recall."),
            LlamaDocument(text="Retrieval quality metrics in RAG systems include context precision, context recall, faithfulness, and answer relevancy. CRAG improves these by filtering irrelevant documents before generation."),
            LlamaDocument(text="Query decomposition breaks a complex question into multiple simpler sub-queries to improve recall. Each sub-query retrieves different aspects of the topic, which are then merged and reranked before answering."),
        ])
        
        self.hyde_engine = build_hyde_engine(self.vector_index)
        self.executor = ThreadPoolExecutor(max_workers=5)

    def process_query(self, query: str, stream: bool = False, status_cb: Optional[Callable[[str], None]] = None):
        if status_cb: status_cb(f"Processing Query: {query}")
        
        if status_cb: status_cb("1. Decomposing Query...")
        sub_queries = plan_query(query)
        if status_cb: status_cb(f"Sub-queries generated: {len(sub_queries)}")
        
        all_retrieved_docs = []
        
        if status_cb: status_cb("2. Retrieving via HyDE (Parallel)...")
        
        def retrieve_sq(sq):
            response = self.hyde_engine.query(sq)
            return [LangchainDocument(page_content=n.text) for n in response.source_nodes]

        results = list(self.executor.map(retrieve_sq, sub_queries))
        for docs in results:
            all_retrieved_docs.extend(docs)
            
        if status_cb: status_cb(f"3. Grading {len(all_retrieved_docs)} Documents (CRAG)...")
        crag_state = {"question": query, "documents": all_retrieved_docs, "web_search": "No", "generation": ""}
        crag_result = crag_app.invoke(crag_state)
        filtered_docs = crag_result["documents"]
        
        if status_cb: status_cb(f"4. Reranking {len(filtered_docs)} Filtered Results...")
        doc_texts = [d.page_content for d in filtered_docs]
        
        if not doc_texts:
            # All local docs were filtered as irrelevant AND web search added nothing useful
            return "No relevant documents found. The CRAG grader filtered all local results and the web search fallback returned no usable content. Try refining the query."
            
        final_docs = self.reranker.rerank(query, doc_texts, top_k=3)
        
        if status_cb: status_cb("5. Generating Final Answer...")
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(final_docs))
        prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided context.
Be concise, accurate, and directly address the question.

Context:
{context}

Question: {query}

Answer:"""
        if stream:
            return (chunk.content for chunk in self.llm.stream(prompt))
        
        final_answer = self.llm.invoke(prompt).content.strip()
        return final_answer

if __name__ == "__main__":
    system = SelfHealingRAGSystem()
    print("\n✅ System ready! Type your question below (or 'quit' to exit).\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        answer = system.process_query(query)
        print("\n" + "="*60)
        print("ANSWER:", answer)
        print("="*60 + "\n")