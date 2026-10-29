import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Generator, Callable, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document as LangchainDocument
from hybrid_rag import HybridGraphVectorRAG
from qdrant_store import QdrantRAGStore
from neo4j_store import Neo4jGraphStore

class SelfHealingRAGSystem:
    """
    Unified Self-Healing Hybrid RAG System:
    Combines Qdrant Vector Store with Neo4j AuraDB Knowledge Graph,
    executing Cypher queries and metadata/taxonomy access control filters.
    """
    def __init__(self):
        print("Initializing Enterprise Hybrid Graph RAG + Vector RAG System...")
        self.qdrant_store = QdrantRAGStore()
        self.neo4j_store = Neo4jGraphStore()
        self.hybrid_engine = HybridGraphVectorRAG(
            qdrant_store=self.qdrant_store,
            neo4j_store=self.neo4j_store
        )
        self.last_retrieval_details = self.hybrid_engine.last_retrieval_details

    def process_query(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        stream: bool = False,
        status_cb: Optional[Callable[[str], None]] = None
    ):
        """
        Executes query retrieval across Qdrant and Neo4j AuraDB with access control filters.
        """
        result = self.hybrid_engine.process_query(
            query=query,
            user_clearance=user_clearance,
            taxonomy=taxonomy,
            stream=stream,
            status_cb=status_cb
        )
        self.last_retrieval_details = self.hybrid_engine.last_retrieval_details
        return result

if __name__ == "__main__":
    system = SelfHealingRAGSystem()
    print("\n" + "="*70)
    print(">> Enterprise Hybrid Graph RAG + Vector RAG Console")
    print(f"[*] Vector Store : {system.qdrant_store.backend_description}")
    print(f"[*] Graph Store  : {system.neo4j_store.backend_type}")
    print("="*70 + "\n")
    
    print("Default clearances: [1] public  [2] internal  [3] confidential")
    clearance_map = {"1": "public", "2": "internal", "3": "confidential"}

    while True:
        try:
            query = input("\nQuery (or 'quit' to exit): ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            c_choice = input("Select Clearance [1=public, 2=internal, 3=confidential] (default 1): ").strip()
            clearance = clearance_map.get(c_choice, "public")

            t_choice = input("Taxonomy Filter [all, rag_architecture, vector_databases, graph_theory, security_governance] (default 'all'): ").strip()
            taxonomy = t_choice if t_choice else "all"

            print(f"\nExecuting Hybrid Search with clearance='{clearance.upper()}' and taxonomy='{taxonomy}'...")
            
            def log_callback(msg):
                print(f"  [Pipeline] {msg}")

            answer = system.process_query(
                query=query,
                user_clearance=clearance,
                taxonomy=taxonomy,
                stream=False,
                status_cb=log_callback
            )

            details = system.last_retrieval_details
            print("\n" + "-"*70)
            print("[CYPHER QUERY EXECUTED] (Neo4j AuraDB):")
            print(details.get("cypher_query", "N/A"))
            print("\n[GRAPH RELATIONSHIPS RETRIEVED]:")
            for fact in details.get("graph_facts", [])[:5]:
                print(f"  * {fact}")
            
            print("\n[QDRANT VECTOR MATCHES]:")
            for doc in details.get("vector_results", [])[:3]:
                print(f"  * [{doc['title']}] (Taxonomy: {doc['taxonomy']} | Access: {doc['access_level']} | Score: {doc['score']})")

            print("\n[ACCESS CONTROL AUDIT]:")
            audit = details.get("audit_report", {})
            print(f"  * Accessible: {audit.get('accessible_count', 0)} / {audit.get('total_documents', 0)} documents")
            print(f"  * Blocked by Policy: {audit.get('blocked_count', 0)} documents")

            print("\n" + "="*70)
            print("FINAL ANSWER:\n", answer)
            print("="*70)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break