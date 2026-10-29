import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document as LangchainDocument
from query_decomposition import plan_query
from corrective_rag import app as crag_app
from reranker import Reranker
from qdrant_store import QdrantRAGStore
from neo4j_store import Neo4jGraphStore

class HybridGraphVectorRAG:
    """
    Genuine Hybrid Graph RAG + Vector RAG Engine.
    Executes parallel retrieval across:
    1. Qdrant Vector Store (dense embeddings + metadata payload filters)
    2. Neo4j AuraDB (knowledge graph traversal + Cypher queries with security WHERE clauses)
    Fuses results, applies CRAG relevance grading & fallback, and reranks via Cross-Encoder.
    """
    def __init__(
        self,
        qdrant_store: Optional[QdrantRAGStore] = None,
        neo4j_store: Optional[Neo4jGraphStore] = None
    ):
        print("Initializing Hybrid Graph + Vector RAG Engine...")
        self.qdrant_store = qdrant_store or QdrantRAGStore()
        self.neo4j_store = neo4j_store or Neo4jGraphStore()
        self.reranker = Reranker()
        
        # Check OpenAI key availability (ignore default placeholder strings)
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key and not openai_key.startswith("your-") and "your-openai-api-key" not in openai_key:
            try:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_key)
            except Exception:
                self.llm = None
        else:
            self.llm = None
            print("Notice: OPENAI_API_KEY not set or placeholder detected. Operating in direct retrieval synthesis mode.")

        self.executor = ThreadPoolExecutor(max_workers=5)
        self.last_retrieval_details: Dict[str, Any] = {}

    def retrieve_vector(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieves dense semantic passages from Qdrant with payload metadata filters."""
        return self.qdrant_store.search(
            query=query,
            user_clearance=user_clearance,
            taxonomy=taxonomy,
            top_k=top_k
        )

    def retrieve_graph(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Executes Cypher query on Neo4j AuraDB with taxonomy and clearance access control filters."""
        return self.neo4j_store.query_graph_with_access_control(
            query=query,
            user_clearance=user_clearance,
            taxonomy=taxonomy,
            top_k=top_k
        )

    def process_query(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        stream: bool = False,
        status_cb: Optional[Callable[[str], None]] = None
    ):
        """
        Executes end-to-end Hybrid RAG:
        1. Query Decomposition
        2. Qdrant Vector Retrieval with Access Control
        3. Neo4j AuraDB Cypher Graph Retrieval with Access Control
        4. Context Fusion & CRAG Grading
        5. Cross-Encoder Reranking
        6. Grounded Answer Synthesis
        """
        if status_cb:
            status_cb(f"Processing Query: '{query}' [Clearance: {user_clearance.upper()}, Taxonomy: {taxonomy}]")

        # 1. Query Decomposition
        if status_cb:
            status_cb("1. Decomposing Query into Atomic Concepts...")
        sub_queries = plan_query(query)
        if status_cb:
            status_cb(f"Sub-queries generated ({len(sub_queries)}): {', '.join(sub_queries)}")

        # 2. Parallel Vector Search via Qdrant
        if status_cb:
            status_cb("2. Vector RAG: Querying Qdrant with Metadata Filters...")
        vector_results = []
        for sq in sub_queries:
            sq_results = self.retrieve_vector(sq, user_clearance=user_clearance, taxonomy=taxonomy, top_k=3)
            vector_results.extend(sq_results)

        # Deduplicate vector results by doc ID
        seen_vector_ids = set()
        unique_vector_results = []
        for doc in vector_results:
            if doc["id"] not in seen_vector_ids:
                seen_vector_ids.add(doc["id"])
                unique_vector_results.append(doc)

        # 3. Graph RAG via Neo4j AuraDB Cypher Query
        if status_cb:
            status_cb("3. Graph RAG: Executing Cypher query on Neo4j AuraDB with Access Control...")
        graph_records, cypher_query = self.retrieve_graph(
            query=query,
            user_clearance=user_clearance,
            taxonomy=taxonomy,
            top_k=6
        )
        if status_cb:
            status_cb(f"Neo4j returned {len(graph_records)} relational facts.")

        # Format Graph Triplets
        graph_facts = []
        for rec in graph_records:
            concept = rec.get("concept")
            rel = rec.get("relation")
            target = rec.get("target_concept")
            doc_title = rec.get("title", "")
            if concept and rel and target:
                graph_facts.append(f"Knowledge Graph Fact: ({concept}) -[:{rel}]-> ({target}) [Source: {doc_title}]")

        # 4. Context Fusion
        if status_cb:
            status_cb(f"4. Fusing {len(unique_vector_results)} Vector chunks + {len(graph_facts)} Graph relationships...")

        fused_documents = []
        for doc in unique_vector_results:
            fused_documents.append(
                LangchainDocument(
                    page_content=f"[Vector Match: {doc['title']} | Taxonomy: {doc['taxonomy']}] {doc['text']}"
                )
            )

        for fact in graph_facts:
            fused_documents.append(
                LangchainDocument(page_content=fact)
            )

        # 5. CRAG Relevance Grading
        if status_cb:
            status_cb(f"5. Grading {len(fused_documents)} Combined Contexts (CRAG)...")

        crag_state = {
            "question": query,
            "documents": fused_documents,
            "web_search": "No",
            "generation": ""
        }
        crag_result = crag_app.invoke(crag_state)
        filtered_docs = crag_result["documents"]

        # 6. Cross-Encoder Reranking
        if status_cb:
            status_cb(f"6. Reranking {len(filtered_docs)} Filtered Results via Cross-Encoder...")

        doc_texts = [d.page_content for d in filtered_docs]
        
        # Access control audit check
        audit_report = self.qdrant_store.audit_access_control(
            user_clearance=user_clearance,
            taxonomy=taxonomy
        )

        # Store details for inspection
        self.last_retrieval_details = {
            "query": query,
            "user_clearance": user_clearance,
            "taxonomy": taxonomy,
            "cypher_query": cypher_query,
            "graph_records": graph_records,
            "graph_facts": graph_facts,
            "vector_results": unique_vector_results,
            "audit_report": audit_report,
            "neo4j_backend": self.neo4j_store.backend_type,
            "qdrant_backend": self.qdrant_store.backend_description
        }

        if not doc_texts:
            msg = (
                f"No accessible documents found for clearance '{user_clearance}' "
                f"and taxonomy '{taxonomy}'. Documents may be restricted by access control filters."
            )
            if stream:
                return (chunk for chunk in [msg])
            return msg

        final_passages = self.reranker.rerank(query, doc_texts, top_k=4)

        # 7. Answer Generation
        if status_cb:
            status_cb("7. Synthesizing Grounded Answer...")

        context_str = "\n\n".join(f"[{i+1}] {passage}" for i, passage in enumerate(final_passages))
        
        prompt = f"""You are an advanced Hybrid Graph RAG + Vector RAG assistant with strict enterprise metadata access control.
Answer the user's question accurately using ONLY the provided context below.
Cite both Vector source documents and Knowledge Graph relationship facts where applicable.
Respect security boundaries: never reveal or speculate about information outside the authorized clearance.

Authorized Clearance Level: {user_clearance.upper()}
Taxonomy Domain: {taxonomy}

Context (Synthesized from Qdrant Vector Store & Neo4j AuraDB Cypher Queries):
{context_str}

Question: {query}

Answer:"""

        if not self.llm:
            fallback_answer = (
                f"**Synthesized Answer (Direct Hybrid Retrieval Summary)**\n\n"
                f"Retrieved via Qdrant Vector Search & Neo4j AuraDB Cypher queries under clearance `{user_clearance.upper()}`:\n\n"
                + "\n\n".join(f"• {p}" for p in final_passages)
            )
            if stream:
                return (chunk for chunk in [fallback_answer])
            return fallback_answer

        def safe_stream_generator():
            try:
                for chunk in self.llm.stream(prompt):
                    yield chunk.content
            except Exception as e:
                yield (
                    f"\n\n*(Notice: LLM API returned {type(e).__name__}: {str(e)[:80]}. Displaying direct grounded context instead:)*\n\n"
                    + "\n\n".join(f"• {p}" for p in final_passages)
                )

        if stream:
            return safe_stream_generator()

        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            fallback_answer = (
                f"**Synthesized Answer (Direct Hybrid Retrieval Summary)**\n\n"
                f"*(Note: LLM API returned {type(e).__name__}: {str(e)[:70]}... Displaying grounded context directly)*\n\n"
                f"**Retrieved Context Passages (Neo4j Cypher + Qdrant):**\n\n"
                + "\n\n".join(f"• {p}" for p in final_passages)
            )
            return fallback_answer
