import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

from neo4j_store import Neo4jGraphStore, SEED_GRAPH_NODES, SEED_GRAPH_RELATIONSHIPS

def generate_knowledge_graph(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
) -> Neo4jGraphStore:
    """
    Initializes and seeds the Knowledge Graph in Neo4j AuraDB (or simulated engine),
    establishing entity nodes, taxonomic relationships, and security access levels.
    """
    print("Connecting to Neo4j Knowledge Graph...")
    graph_store = Neo4jGraphStore(uri=uri, username=username, password=password, database=database)
    
    print(f"Graph Store Active: {graph_store.backend_type}")
    print(f"Seeding {len(SEED_GRAPH_NODES)} nodes and {len(SEED_GRAPH_RELATIONSHIPS)} relationships...")
    
    if graph_store.is_connected:
        graph_store.init_schema_and_seed()
        print("[OK] Knowledge Graph seeded successfully into live Neo4j AuraDB!")
    else:
        print("[INFO] Operating with simulated in-memory Cypher Knowledge Graph.")
        
    return graph_store

def query_graph_cypher(
    graph_store: Neo4jGraphStore,
    query: str,
    user_clearance: str = "public",
    taxonomy: str = "all"
):
    """
    Queries Neo4j AuraDB using Cypher with strict taxonomy and clearance filtering.
    """
    print(f"\n" + "="*60)
    print(f"Querying Knowledge Graph for: '{query}'")
    print(f"Clearance Level : {user_clearance.upper()}")
    print(f"Taxonomy Filter : {taxonomy}")
    print("="*60)

    records, cypher = graph_store.query_graph_with_access_control(
        query=query,
        user_clearance=user_clearance,
        taxonomy=taxonomy,
        top_k=5
    )

    print("\n[Cypher Query Executed]:")
    print(cypher)

    print(f"\n[Retrieved Graph Facts ({len(records)})]:")
    if records:
        for idx, rec in enumerate(records, 1):
            concept = rec.get("concept")
            rel = rec.get("relation")
            target = rec.get("target_concept")
            access = rec.get("access_level")
            tax = rec.get("taxonomy")
            print(f" {idx}. ({concept}) -[:{rel}]-> ({target}) | Doc: {rec.get('title')} [Access: {access}, Taxonomy: {tax}]")
    else:
        print(" (No matching graph nodes found within current clearance & taxonomy restrictions.)")

if __name__ == "__main__":
    neo4j_uri = os.getenv("NEO4J_URI")
    if not neo4j_uri:
        print("[INFO] NEO4J_URI not set in .env. Will run with simulated Cypher graph engine.")
        print("[INFO] To connect a free Neo4j AuraDB instance, add NEO4J_URI & NEO4J_PASSWORD to your .env file.\n")
    
    kg = generate_knowledge_graph()

    # Test 1: Public clearance query on CRAG
    query_graph_cypher(kg, "CRAG", user_clearance="public", taxonomy="rag_architecture")

    # Test 2: Public clearance query trying to access confidential security
    query_graph_cypher(kg, "Security Protocol", user_clearance="public", taxonomy="security_governance")

    # Test 3: Confidential clearance query accessing confidential security
    query_graph_cypher(kg, "Security Protocol", user_clearance="confidential", taxonomy="security_governance")
