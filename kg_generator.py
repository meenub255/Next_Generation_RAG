import os
from dotenv import load_dotenv
from llama_index.core import (
    KnowledgeGraphIndex,
    ServiceContext,
    StorageContext,
    Document as LlamaDocument,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores import SimpleGraphStore

# Load environment variables (API Keys)
load_dotenv()

def generate_knowledge_graph():
    print("Initializing LLM for Knowledge Graph Extraction...")
    
    # Configure LLM (GPT-4o-mini is cost-effective for triplet extraction)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.llm = llm
    Settings.chunk_size = 512

    # 1. Define source documents (Reusing from main.py)
    documents = [
        LlamaDocument(text="Corrective RAG (CRAG) improves retrieval quality by adding a lightweight relevance grader that evaluates each retrieved document."),
        LlamaDocument(text="CRAG stands for Corrective Retrieval-Augmented Generation. It uses an LLM grader and dynamically corrects retrieval using web search."),
        LlamaDocument(text="HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to find semantically similar documents in the vector store."),
        LlamaDocument(text="Query decomposition breaks a complex question into multiple simpler sub-queries to improve recall."),
        LlamaDocument(text="The system uses Tavily Search as a fallback when local documents are deemed insufficient or irrelevant."),
    ]

    # 2. Setup Graph Store
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    print("Extracting entities and relations (this may take a moment)...")
    
    # 3. Build Knowledge Graph Index
    # This will use the LLM to identify (subject, predicate, object) triplets
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=3,
        storage_context=storage_context,
        include_embeddings=True,
    )

    print("\n✅ Knowledge Graph Generated successfully!")
    return index

def query_graph(index, query):
    # Query using the graph-based retrieval engine
    query_engine = index.as_query_engine(
        include_text=True, 
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=5
    )
    
    print(f"\nQuerying Graph: {query}")
    response = query_engine.query(query)
    print(f"Answer: {response}")
    
    # Optional: Print extracted triplets for transparency
    print("\nSample Triplets in Graph:")
    try:
        # Method 3: Use the NetworkX graph representation (very robust)
        g = index.get_networkx_graph()
        triplets = []
        for u, v, d in g.edges(data=True):
            rel = d.get('relationship', 'related to')
            triplets.append((u, rel, v))
    except Exception as e:
        triplets = []
        print(f"  (Note: Hidden debug - {e})")

    if triplets:
        for triplet in list(triplets)[:10]:
            print(f"  {triplet}")
    else:
        print("  (No triplets found in the graph store.)")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file.")
    else:
        kg_index = generate_knowledge_graph()
        
        # Example queries to test the graph
        query_graph(kg_index, "What is the relationship between CRAG and web search?")
        query_graph(kg_index, "How does HyDE work?")
