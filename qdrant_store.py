import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Default sample documents with multi-level taxonomy and access control
DEFAULT_RAG_DOCUMENTS = [
    {
        "id": "doc-01",
        "title": "Corrective RAG Overview",
        "text": "Corrective RAG (CRAG) improves retrieval quality by adding a lightweight relevance grader that evaluates each retrieved document. Irrelevant documents are discarded and web search fallback is triggered.",
        "taxonomy": "rag_architecture",
        "access_level": "public",
        "department": "research",
        "entities": ["CRAG", "Corrective RAG", "Relevance Grader", "Web Search Fallback"]
    },
    {
        "id": "doc-02",
        "title": "HyDE Retrieval Mechanics",
        "text": "HyDE (Hypothetical Document Embeddings) generates a hypothetical response to an input query and utilizes its semantic vector to locate nearby relevant documents in vector space, boosting zero-shot recall.",
        "taxonomy": "rag_architecture",
        "access_level": "public",
        "department": "research",
        "entities": ["HyDE", "Hypothetical Document", "Vector Space", "Semantic Search"]
    },
    {
        "id": "doc-03",
        "title": "Query Decomposition Framework",
        "text": "Query decomposition breaks a multifaceted user prompt into isolated atomic sub-queries. Sub-queries are retrieved in parallel and merged across graph and vector pipelines.",
        "taxonomy": "rag_architecture",
        "access_level": "public",
        "department": "engineering",
        "entities": ["Query Decomposition", "Sub-queries", "Parallel Retrieval"]
    },
    {
        "id": "doc-04",
        "title": "Qdrant Vector Database Architecture",
        "text": "Qdrant is an open-source vector similarity search engine with native support for payload payload-based filtering, HNSW indexing, and hardware-accelerated distance metrics for enterprise RAG.",
        "taxonomy": "vector_databases",
        "access_level": "internal",
        "department": "engineering",
        "entities": ["Qdrant", "HNSW", "Payload Filtering", "Vector Search"]
    },
    {
        "id": "doc-05",
        "title": "Neo4j AuraDB Knowledge Graph Synergy",
        "text": "Neo4j AuraDB is a fully managed cloud graph database supporting Cypher queries. Graph RAG navigates multi-hop relationship paths between entities that pure vector embeddings miss.",
        "taxonomy": "graph_theory",
        "access_level": "internal",
        "department": "engineering",
        "entities": ["Neo4j", "AuraDB", "Cypher", "Knowledge Graph", "Multi-Hop Traversal"]
    },
    {
        "id": "doc-06",
        "title": "Enterprise Zero-Trust Security Protocol",
        "text": "CONFIDENTIAL: Internal enterprise security architecture requires zero-trust attribute-based access control (ABAC). Metadata filters must sanitize retrieved context before sending prompts to the LLM to prevent data leakage.",
        "taxonomy": "security_governance",
        "access_level": "confidential",
        "department": "executive",
        "entities": ["Zero-Trust", "ABAC", "Metadata Filter", "Data Leakage Prevention", "Clearance"]
    },
    {
        "id": "doc-07",
        "title": "Proprietary Model Fine-Tuning Hyperparameters",
        "text": "CONFIDENTIAL: Production LoRA fine-tuning hyperparameters for domain-adapted RAG rerankers: rank=64, alpha=128, target modules [q_proj, v_proj], learning rate 2e-5 with cosine decay schedule.",
        "taxonomy": "security_governance",
        "access_level": "confidential",
        "department": "executive",
        "entities": ["LoRA", "Fine-Tuning", "Hyperparameters", "Proprietary Weights"]
    }
]

# Clearance hierarchy: Public < Internal < Confidential
CLEARANCE_HIERARCHY = {
    "public": ["public"],
    "internal": ["public", "internal"],
    "confidential": ["public", "internal", "confidential"]
}

class QdrantRAGStore:
    """
    Qdrant vector store integration supporting both Qdrant Cloud and local in-memory instances,
    with native payload metadata access control filters (Clearance & Taxonomy).
    """
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client: Optional[QdrantClient] = None
        self.embedding_fn = None
        self.vector_dim = 384  # Default MiniLM dimension
        
        self._init_embeddings()
        self._init_client()
        self._ensure_collection()
        self.seed_default_documents()

    def _init_embeddings(self):
        """Initializes SentenceTransformer or OpenAI embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.vector_dim = 384
            self.embedding_fn = lambda text: model.encode(text).tolist()
            self.embedding_source = "SentenceTransformer (all-MiniLM-L6-v2)"
        except Exception as e:
            # Fallback lightweight deterministic embedding
            import hashlib
            import numpy as np
            self.vector_dim = 384
            def hash_embed(text: str) -> List[float]:
                h = hashlib.sha256(text.encode()).digest()
                rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
                vec = rng.randn(self.vector_dim)
                return (vec / np.linalg.norm(vec)).tolist()
            self.embedding_fn = hash_embed
            self.embedding_source = "Local Deterministic Hash Fallback"

    def _init_client(self):
        """Initializes connection to Qdrant Cloud or local in-memory instance."""
        if self.url and self.api_key:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
                self.is_cloud = True
                self.backend_description = f"Qdrant Cloud ({self.url})"
                return
            except Exception as e:
                print(f"[Qdrant] Cloud connection failed: {e}. Falling back to in-memory.")
        
        # Local in-memory instance for zero-config execution
        self.client = QdrantClient(location=":memory:")
        self.is_cloud = False
        self.backend_description = "Qdrant In-Memory (Zero Configuration)"

    def _ensure_collection(self):
        """Ensures collection exists with appropriate vector parameters."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    )
                )
                # Create payload indexes for metadata filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="access_level",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="taxonomy",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
        except Exception as e:
            print(f"[Qdrant] Error configuring collection: {e}")

    def seed_default_documents(self):
        """Seeds the default document corpus with taxonomy and access levels."""
        try:
            count = self.client.count(collection_name=self.collection_name).count
            if count == 0:
                self.add_documents(DEFAULT_RAG_DOCUMENTS)
        except Exception as e:
            print(f"[Qdrant] Error seeding default documents: {e}")

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Embeds and indexes documents into Qdrant with payload metadata."""
        points = []
        for doc in docs:
            text = doc.get("text", "")
            vector = self.embedding_fn(text)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.get("id", text[:30])))
            payload = {
                "doc_id": doc.get("id"),
                "title": doc.get("title", "Untitled"),
                "text": text,
                "taxonomy": doc.get("taxonomy", "general").lower(),
                "access_level": doc.get("access_level", "public").lower(),
                "department": doc.get("department", "general").lower(),
                "entities": doc.get("entities", [])
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes vector similarity search with metadata access control filters.
        Enforces clearance hierarchy and domain taxonomy.
        """
        query_vector = self.embedding_fn(query)
        allowed_clearances = CLEARANCE_HIERARCHY.get(user_clearance.lower(), ["public"])

        must_conditions: List[models.Condition] = [
            models.FieldCondition(
                key="access_level",
                match=models.MatchAny(any=allowed_clearances)
            )
        ]

        if taxonomy and taxonomy.lower() != "all":
            must_conditions.append(
                models.FieldCondition(
                    key="taxonomy",
                    match=models.MatchValue(value=taxonomy.lower())
                )
            )

        filter_condition = models.Filter(must=must_conditions)

        # Query points using Qdrant client
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter_condition,
            limit=top_k,
            with_payload=True
        )

        retrieved = []
        for point in results.points:
            retrieved.append({
                "id": point.payload.get("doc_id"),
                "title": point.payload.get("title"),
                "text": point.payload.get("text"),
                "taxonomy": point.payload.get("taxonomy"),
                "access_level": point.payload.get("access_level"),
                "department": point.payload.get("department"),
                "score": round(float(point.score), 4)
            })

        return retrieved

    def audit_access_control(self, user_clearance: str = "public", taxonomy: str = "all") -> Dict[str, Any]:
        """
        Calculates allowed documents vs blocked documents for compliance reporting.
        """
        allowed_clearances = CLEARANCE_HIERARCHY.get(user_clearance.lower(), ["public"])
        
        # Fetch all points without filter to compare
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True
        )

        total_count = len(all_points)
        allowed_docs = []
        blocked_docs = []

        for p in all_points:
            lvl = p.payload.get("access_level", "public")
            tax = p.payload.get("taxonomy", "general")
            doc_info = {
                "id": p.payload.get("doc_id"),
                "title": p.payload.get("title"),
                "access_level": lvl,
                "taxonomy": tax
            }

            clearance_ok = lvl in allowed_clearances
            taxonomy_ok = (taxonomy.lower() == "all" or tax == taxonomy.lower())

            if clearance_ok and taxonomy_ok:
                allowed_docs.append(doc_info)
            else:
                reason = []
                if not clearance_ok:
                    reason.append(f"Requires clearance '{lvl}', user has '{user_clearance}'")
                if not taxonomy_ok:
                    reason.append(f"Taxonomy '{tax}' does not match '{taxonomy}'")
                doc_info["blocked_reason"] = "; ".join(reason)
                blocked_docs.append(doc_info)

        return {
            "total_documents": total_count,
            "accessible_count": len(allowed_docs),
            "blocked_count": len(blocked_docs),
            "allowed_documents": allowed_docs,
            "blocked_documents": blocked_docs,
            "user_clearance": user_clearance,
            "selected_taxonomy": taxonomy
        }
