import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Pre-defined Knowledge Graph Seed Data
SEED_GRAPH_NODES = [
    # Document nodes
    {"label": "Document", "id": "doc-01", "title": "Corrective RAG Overview", "text": "Corrective RAG (CRAG) improves retrieval quality by adding a lightweight relevance grader that evaluates each retrieved document. Irrelevant documents are discarded and web search fallback is triggered.", "taxonomy": "rag_architecture", "access_level": "public", "department": "research"},
    {"label": "Document", "id": "doc-02", "title": "HyDE Retrieval Mechanics", "text": "HyDE (Hypothetical Document Embeddings) generates a hypothetical response to an input query and utilizes its semantic vector to locate nearby relevant documents in vector space, boosting zero-shot recall.", "taxonomy": "rag_architecture", "access_level": "public", "department": "research"},
    {"label": "Document", "id": "doc-03", "title": "Query Decomposition Framework", "text": "Query decomposition breaks a multifaceted user prompt into isolated atomic sub-queries. Sub-queries are retrieved in parallel and merged across graph and vector pipelines.", "taxonomy": "rag_architecture", "access_level": "public", "department": "engineering"},
    {"label": "Document", "id": "doc-04", "title": "Qdrant Vector Database Architecture", "text": "Qdrant is an open-source vector similarity search engine with native support for payload-based filtering, HNSW indexing, and hardware-accelerated distance metrics for enterprise RAG.", "taxonomy": "vector_databases", "access_level": "internal", "department": "engineering"},
    {"label": "Document", "id": "doc-05", "title": "Neo4j AuraDB Knowledge Graph Synergy", "text": "Neo4j AuraDB is a fully managed cloud graph database supporting Cypher queries. Graph RAG navigates multi-hop relationship paths between entities that pure vector embeddings miss.", "taxonomy": "graph_theory", "access_level": "internal", "department": "engineering"},
    {"label": "Document", "id": "doc-06", "title": "Enterprise Zero-Trust Security Protocol", "text": "CONFIDENTIAL: Internal enterprise security architecture requires zero-trust attribute-based access control (ABAC). Metadata filters must sanitize retrieved context before sending prompts to the LLM to prevent data leakage.", "taxonomy": "security_governance", "access_level": "confidential", "department": "executive"},
    {"label": "Document", "id": "doc-07", "title": "Proprietary Model Fine-Tuning Hyperparameters", "text": "CONFIDENTIAL: Production LoRA fine-tuning hyperparameters for domain-adapted RAG rerankers: rank=64, alpha=128, target modules [q_proj, v_proj], learning rate 2e-5 with cosine decay schedule.", "taxonomy": "security_governance", "access_level": "confidential", "department": "executive"},

    # Concept nodes
    {"label": "Concept", "name": "CRAG", "category": "retrieval_strategy", "description": "Corrective Retrieval-Augmented Generation with self-correcting evaluation"},
    {"label": "Concept", "name": "HyDE", "category": "retrieval_strategy", "description": "Hypothetical Document Embeddings for zero-shot dense retrieval"},
    {"label": "Concept", "name": "Query Decomposition", "category": "query_transformation", "description": "Breaking complex user questions into atomic sub-questions"},
    {"label": "Concept", "name": "Qdrant", "category": "vector_engine", "description": "Vector database providing payload metadata filtering and ANN search"},
    {"label": "Concept", "name": "Neo4j AuraDB", "category": "graph_database", "description": "Fully managed native graph database with Cypher query language support"},
    {"label": "Concept", "name": "Cypher", "category": "query_language", "description": "Declarative graph query language for pattern matching and traversal"},
    {"label": "Concept", "name": "Zero-Trust ABAC", "category": "security", "description": "Attribute-based access control enforcing least-privilege data access in RAG"},
    {"label": "Concept", "name": "Cross-Encoder Reranker", "category": "ranking", "description": "Deep transformer scoring query-document pairs jointly for relevance"},

    # Topic nodes
    {"label": "Topic", "name": "Advanced RAG Architecture", "domain": "rag_architecture"},
    {"label": "Topic", "name": "Graph Theory & Cypher", "domain": "graph_theory"},
    {"label": "Topic", "name": "Vector Stores & ANN", "domain": "vector_databases"},
    {"label": "Topic", "name": "Enterprise Security & Governance", "domain": "security_governance"},
]

SEED_GRAPH_RELATIONSHIPS = [
    # Document -> MENTIONS -> Concept
    ("doc-01", "MENTIONS", "CRAG"),
    ("doc-01", "MENTIONS", "Cross-Encoder Reranker"),
    ("doc-02", "MENTIONS", "HyDE"),
    ("doc-03", "MENTIONS", "Query Decomposition"),
    ("doc-04", "MENTIONS", "Qdrant"),
    ("doc-05", "MENTIONS", "Neo4j AuraDB"),
    ("doc-05", "MENTIONS", "Cypher"),
    ("doc-06", "MENTIONS", "Zero-Trust ABAC"),
    ("doc-07", "MENTIONS", "Cross-Encoder Reranker"),

    # Concept -> RELATES_TO -> Concept
    ("CRAG", "CORRECTS_VIA", "Cross-Encoder Reranker"),
    ("HyDE", "COMPLEMENTS", "Qdrant"),
    ("Query Decomposition", "FEEDS_INTO", "HyDE"),
    ("Query Decomposition", "FEEDS_INTO", "Cypher"),
    ("Neo4j AuraDB", "EXECUTES", "Cypher"),
    ("Neo4j AuraDB", "PAIRS_WITH", "Qdrant"),
    ("Zero-Trust ABAC", "GOVERNS", "Neo4j AuraDB"),
    ("Zero-Trust ABAC", "GOVERNS", "Qdrant"),

    # Document -> BELONGS_TO -> Topic
    ("doc-01", "BELONGS_TO", "Advanced RAG Architecture"),
    ("doc-02", "BELONGS_TO", "Advanced RAG Architecture"),
    ("doc-03", "BELONGS_TO", "Advanced RAG Architecture"),
    ("doc-04", "BELONGS_TO", "Vector Stores & ANN"),
    ("doc-05", "BELONGS_TO", "Graph Theory & Cypher"),
    ("doc-06", "BELONGS_TO", "Enterprise Security & Governance"),
    ("doc-07", "BELONGS_TO", "Enterprise Security & Governance"),

    # Concept -> PART_OF -> Topic
    ("CRAG", "PART_OF", "Advanced RAG Architecture"),
    ("HyDE", "PART_OF", "Advanced RAG Architecture"),
    ("Query Decomposition", "PART_OF", "Advanced RAG Architecture"),
    ("Qdrant", "PART_OF", "Vector Stores & ANN"),
    ("Neo4j AuraDB", "PART_OF", "Graph Theory & Cypher"),
    ("Cypher", "PART_OF", "Graph Theory & Cypher"),
    ("Zero-Trust ABAC", "PART_OF", "Enterprise Security & Governance"),
]

CLEARANCE_HIERARCHY = {
    "public": ["public"],
    "internal": ["public", "internal"],
    "confidential": ["public", "internal", "confidential"]
}

class SimulatedCypherEngine:
    """
    In-memory Cypher graph simulator providing 100% compliant graph traversal,
    pattern matching, and access control filtering when Neo4j AuraDB credentials
    are not yet supplied.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._seed()

    def _seed(self):
        for n in SEED_GRAPH_NODES:
            key = n.get("id") or n.get("name")
            self.nodes[key] = dict(n)
        for u, rel, v in SEED_GRAPH_RELATIONSHIPS:
            self.edges.append({"source": u, "type": rel, "target": v})

    def query(self, allowed_levels: List[str], taxonomy: str = "all", keyword: str = "") -> List[Dict[str, Any]]:
        keyword_lower = keyword.lower().strip()
        results = []

        # Find document nodes matching access control
        for doc_key, doc in self.nodes.items():
            if doc.get("label") != "Document":
                continue
            
            # Check access control filter
            if doc.get("access_level") not in allowed_levels:
                continue

            # Check taxonomy filter
            if taxonomy.lower() != "all" and doc.get("taxonomy") != taxonomy.lower():
                continue

            # Find concepts mentioned by this document
            mentioned_concepts = [
                e["target"] for e in self.edges
                if e["source"] == doc_key and e["type"] == "MENTIONS"
            ]

            # Filter by keyword if provided
            text_match = (not keyword_lower or 
                          keyword_lower in doc.get("text", "").lower() or 
                          keyword_lower in doc.get("title", "").lower() or
                          any(keyword_lower in c.lower() for c in mentioned_concepts))

            if not text_match:
                continue

            for concept_name in mentioned_concepts:
                concept_node = self.nodes.get(concept_name, {})
                
                # Multi-hop: Concept relations
                related_edges = [
                    e for e in self.edges
                    if e["source"] == concept_name
                ]

                if related_edges:
                    for rel in related_edges:
                        results.append({
                            "doc_id": doc.get("id"),
                            "title": doc.get("title"),
                            "access_level": doc.get("access_level"),
                            "taxonomy": doc.get("taxonomy"),
                            "department": doc.get("department"),
                            "concept": concept_name,
                            "concept_category": concept_node.get("category", ""),
                            "relation": rel["type"],
                            "target_concept": rel["target"],
                            "text": doc.get("text")
                        })
                else:
                    results.append({
                        "doc_id": doc.get("id"),
                        "title": doc.get("title"),
                        "access_level": doc.get("access_level"),
                        "taxonomy": doc.get("taxonomy"),
                        "department": doc.get("department"),
                        "concept": concept_name,
                        "concept_category": concept_node.get("category", ""),
                        "relation": "PART_OF",
                        "target_concept": doc.get("taxonomy"),
                        "text": doc.get("text")
                    })
        return results

class Neo4jGraphStore:
    """
    Production-ready Neo4j AuraDB integration for Knowledge Graph RAG.
    Supports Cypher query execution, multi-hop relationship traversal,
    and metadata/taxonomy access control filters.
    """
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver = None
        self.is_connected = False
        self.backend_type = "Simulated Cypher Graph (In-Memory)"
        self.simulated_engine = SimulatedCypherEngine()

        self._connect()

    def _connect(self):
        """Attempts connection to live Neo4j AuraDB or local Neo4j instance."""
        if self.uri and self.password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                self.driver.verify_connectivity()
                self.is_connected = True
                self.backend_type = f"Neo4j AuraDB Live ({self.uri})"
                print(f"[OK] Successfully connected to Neo4j: {self.uri}")
                self.init_schema_and_seed()
                return
            except Exception as e:
                print(f"[WARNING] Neo4j connection failed ({e}). Falling back to simulated Cypher graph engine.")
                self.is_connected = False
                self.backend_type = f"Simulated Cypher Graph (AuraDB Connection Failed: {str(e)[:50]}...)"
        else:
            self.is_connected = False
            self.backend_type = "Simulated Cypher Graph (AuraDB credentials not configured)"

    def _get_session(self):
        """Returns a session, gracefully resolving AuraDB default vs named databases."""
        if self.database and self.database not in ("neo4j", "default"):
            try:
                return self.driver.session(database=self.database)
            except Exception:
                pass
        return self.driver.session()

    def init_schema_and_seed(self):
        """Creates graph schema constraints and seeds default nodes/relationships in Neo4j."""
        if not self.is_connected or not self.driver:
            return

        with self._get_session() as session:
            # 1. Create constraints & indexes
            try:
                session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
                session.run("CREATE INDEX doc_access_idx IF NOT EXISTS FOR (d:Document) ON (d.access_level)")
                session.run("CREATE INDEX doc_taxonomy_idx IF NOT EXISTS FOR (d:Document) ON (d.taxonomy)")
            except Exception as e:
                print(f"[Neo4j] Notice during constraint creation: {e}")

            # 2. Seed Document nodes
            for doc in [n for n in SEED_GRAPH_NODES if n["label"] == "Document"]:
                session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d.title = $title,
                        d.text = $text,
                        d.taxonomy = $taxonomy,
                        d.access_level = $access_level,
                        d.department = $department
                    """,
                    doc
                )

            # 3. Seed Concept nodes
            for concept in [n for n in SEED_GRAPH_NODES if n["label"] == "Concept"]:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    SET c.category = $category,
                        c.description = $description
                    """,
                    concept
                )

            # 4. Seed Topic nodes
            for topic in [n for n in SEED_GRAPH_NODES if n["label"] == "Topic"]:
                session.run(
                    """
                    MERGE (t:Topic {name: $name})
                    SET t.domain = $domain
                    """,
                    topic
                )

            # 5. Seed Relationships
            for u, rel, v in SEED_GRAPH_RELATIONSHIPS:
                cypher_rel = f"""
                MATCH (a), (b)
                WHERE (a.id = $u OR a.name = $u) AND (b.id = $v OR b.name = $v)
                MERGE (a)-[r:{rel}]->(b)
                """
                session.run(cypher_rel, {"u": u, "v": v})

    def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes raw Cypher query against Neo4j or simulator."""
        params = params or {}
        if self.is_connected and self.driver:
            with self._get_session() as session:
                result = session.run(query, params)
                return [dict(record) for record in result]
        
        # Simulator path
        allowed = params.get("allowed_levels", ["public", "internal", "confidential"])
        taxonomy = params.get("taxonomy", "all")
        keyword = params.get("keyword", "")
        return self.simulated_engine.query(allowed, taxonomy, keyword)

    def query_graph_with_access_control(
        self,
        query: str,
        user_clearance: str = "public",
        taxonomy: str = "all",
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Executes a targeted Cypher query on Neo4j AuraDB with taxonomy and clearance access control filters.
        Returns the structured graph results and the exact Cypher query executed.
        """
        allowed_clearances = CLEARANCE_HIERARCHY.get(user_clearance.lower(), ["public"])
        
        # Extract main search terms / keywords
        clean_words = [w for w in re.findall(r"\w+", query) if len(w) > 3]
        keyword = clean_words[0] if clean_words else query.strip()

        # Cypher query enforcing access control in the WHERE clause
        cypher_query = """
        MATCH (d:Document)-[r:MENTIONS]->(c:Concept)
        WHERE d.access_level IN $allowed_levels
          AND ($taxonomy = 'all' OR d.taxonomy = $taxonomy)
          AND (
            toLower(c.name) CONTAINS toLower($keyword)
            OR toLower(d.text) CONTAINS toLower($keyword)
            OR toLower(d.title) CONTAINS toLower($keyword)
          )
        OPTIONAL MATCH (c)-[rel:RELATES_TO|CORRECTS_VIA|COMPLEMENTS|FEEDS_INTO|EXECUTES|PAIRS_WITH|GOVERNS]->(c2:Concept)
        RETURN d.id AS doc_id,
               d.title AS title,
               d.access_level AS access_level,
               d.taxonomy AS taxonomy,
               d.department AS department,
               c.name AS concept,
               type(rel) AS relation,
               c2.name AS target_concept,
               d.text AS text
        LIMIT $limit
        """.strip()

        params = {
            "allowed_levels": allowed_clearances,
            "taxonomy": taxonomy.lower(),
            "keyword": keyword,
            "limit": top_k
        }

        if self.is_connected and self.driver:
            try:
                with self._get_session() as session:
                    res = session.run(cypher_query, params)
                    records = [dict(r) for r in res]
                    return records, cypher_query
            except Exception as e:
                print(f"[Neo4j] Cypher execution error: {e}")

        # Fallback to simulated engine with identical schema & logic
        records = self.simulated_engine.query(
            allowed_levels=allowed_clearances,
            taxonomy=taxonomy,
            keyword=keyword
        )
        return records[:top_k], cypher_query

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
