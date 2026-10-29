import streamlit as st
import os
from dotenv import load_dotenv
import time

load_dotenv()
from main import SelfHealingRAGSystem
from qdrant_store import DEFAULT_RAG_DOCUMENTS

# --- Page Config ---
st.set_page_config(
    page_title="Hybrid Graph RAG + Vector RAG | Neo4j AuraDB + Qdrant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Styling (Glassmorphism & Gradients) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f111a 0%, #16192b 100%);
        color: #ffffff;
    }
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d2ff, #7928ca);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .badge-live {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-sim {
        background-color: #d97706;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-clearance {
        background-color: #4f46e5;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .user-bubble {
        background: rgba(79, 70, 229, 0.15);
    }
    .ai-bubble {
        background: rgba(255, 255, 255, 0.04);
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize RAG System ---
@st.cache_resource
def get_rag_system():
    return SelfHealingRAGSystem()

rag = get_rag_system()

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=90)
    st.title("System Controls")
    
    # Store Connection Status
    st.markdown("### 🔌 Database Backends")
    neo4j_live = rag.neo4j_store.is_connected
    if neo4j_live:
        st.markdown(f'<span class="badge-live">Neo4j AuraDB Live</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-sim">Neo4j Cypher Simulator</span>', unsafe_allow_html=True)
    st.caption(f"Graph Store: `{rag.neo4j_store.backend_type}`")

    st.markdown(f'<span class="badge-live">{rag.qdrant_store.backend_description}</span>', unsafe_allow_html=True)
    st.caption(f"Vector Store: Qdrant with native payload filtering")

    st.divider()

    # Access Control & Taxonomy Filter
    st.markdown("### 🛡️ Access Control & Taxonomy")
    clearance = st.selectbox(
        "User Security Clearance",
        options=["public", "internal", "confidential"],
        index=0,
        help="Simulates Attribute-Based Access Control (ABAC). 'Public' cannot view internal or confidential docs."
    )
    
    taxonomy = st.selectbox(
        "Taxonomy Domain Filter",
        options=["all", "rag_architecture", "vector_databases", "graph_theory", "security_governance"],
        index=0,
        help="Filters both Qdrant vectors and Neo4j Cypher nodes to specific domain categories."
    )

    st.divider()

    # OpenAI API Key Configuration
    with st.expander("🔑 OpenAI API Key (Optional)"):
        env_openai_key = os.getenv("OPENAI_API_KEY", "")
        masked_default = "" if env_openai_key.startswith("your-") or "your-openai" in env_openai_key else env_openai_key
        user_openai_key = st.text_input(
            "OpenAI API Key",
            value=masked_default,
            type="password",
            help="Optional: If left blank, the system operates in direct retrieval mode with grounded Neo4j Cypher + Qdrant facts."
        )
        if st.button("Apply API Key"):
            if user_openai_key and not user_openai_key.startswith("your-"):
                os.environ["OPENAI_API_KEY"] = user_openai_key
                try:
                    from langchain_openai import ChatOpenAI
                    rag.hybrid_engine.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=user_openai_key)
                    st.success("OpenAI API key configured!")
                except Exception as e:
                    st.error(f"Failed to set model: {e}")
            else:
                rag.hybrid_engine.llm = None
                st.info("Using Direct Retrieval Mode (No OpenAI API key required).")
            st.rerun()

    # Neo4j AuraDB Live Connection Settings
    with st.expander("⚙️ Connect Free Neo4j AuraDB"):
        st.markdown(
            "Get a free instance in 2 minutes at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/)."
        )
        neo_uri = st.text_input("NEO4J_URI", value=os.getenv("NEO4J_URI", "neo4j+s://xxxxxxxx.databases.neo4j.io"))
        neo_user = st.text_input("NEO4J_USERNAME", value=os.getenv("NEO4J_USERNAME", "neo4j"))
        neo_pass = st.text_input("NEO4J_PASSWORD", type="password", value=os.getenv("NEO4J_PASSWORD", ""))
        
        if st.button("Connect & Seed AuraDB"):
            if neo_uri and neo_pass:
                with st.spinner("Connecting to Neo4j AuraDB..."):
                    rag.neo4j_store.uri = neo_uri
                    rag.neo4j_store.username = neo_user
                    rag.neo4j_store.password = neo_pass
                    rag.neo4j_store._connect()
                    if rag.neo4j_store.is_connected:
                        st.success("Connected to live Neo4j AuraDB!")
                        st.rerun()
                    else:
                        st.error("Failed to connect. Check URI and password.")
            else:
                st.warning("Please supply both NEO4J_URI and NEO4J_PASSWORD.")

    if st.button("Re-Seed Knowledge Graph & Vectors"):
        with st.spinner("Seeding databases..."):
            rag.qdrant_store.seed_default_documents()
            rag.neo4j_store.init_schema_and_seed()
            st.success("Databases synchronized with taxonomy & security metadata!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Custom Log Handler for Streamlit ---
class StreamlitLogger:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.logs = ""

    def log(self, text):
        self.logs += f"> {text}\n"
        self.placeholder.code(self.logs, language="bash")

# --- Header ---
st.markdown('<div class="main-header">Hybrid Graph RAG + Vector RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Genuinely fusing <b>Neo4j AuraDB Cypher Queries</b> with '
    '<b>Qdrant Payload Vector Search</b> under strict <b>Taxonomy & Metadata Access Control</b>.</div>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user",
            "content": "How does Neo4j AuraDB graph traversal complement vector embeddings in this hybrid architecture?"
        },
        {
            "role": "assistant",
            "content": """**Hybrid Graph + Vector Synthesis:**

In this enterprise RAG architecture, **Neo4j AuraDB** and **Qdrant Vector Store** operate in parallel to overcome the inherent blindspots of isolated retrieval engines:

1. **Multi-Hop Relational Traversal (Neo4j AuraDB):**
   - Navigates explicit entity relationships via Cypher pattern matching (`(c)-[:PAIRS_WITH|COMPLEMENTS]->(c2)`).
   - Captures multi-hop structural facts that dense cosine distance metrics overlook (e.g., hierarchical ontologies and dependency graphs).
   - Enforces Attribute-Based Access Control (ABAC) directly in the database engine via parameter-bound Cypher `WHERE` filters (`d.access_level IN $allowed_levels`).

2. **Semantic Similarity & Dense Filtering (Qdrant Vector Store):**
   - Retrieves unstructured passages based on semantic intent and cosine proximity using HNSW indexing.
   - Enforces pre-filtering at the payload level to guarantee zero data leakage before vector similarity calculation.

3. **Fusion & Verification (LangGraph CRAG & Cross-Encoder):**
   - Contexts from both stores are deduplicated, graded for relevance via Corrective RAG (CRAG), and reranked using a Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before grounded generation by GPT-4o-mini.""",
            "details": {
                "cypher_query": """MATCH (d:Document)-[r:MENTIONS]->(c:Concept)
WHERE d.access_level IN ['public', 'internal']
  AND ($taxonomy = 'all' OR d.taxonomy = $taxonomy)
  AND (toLower(c.name) CONTAINS 'neo4j' OR toLower(d.text) CONTAINS 'neo4j')
OPTIONAL MATCH (c)-[rel:RELATES_TO|CORRECTS_VIA|COMPLEMENTS|FEEDS_INTO|EXECUTES|PAIRS_WITH|GOVERNS]->(c2:Concept)
RETURN d.id, d.title, d.access_level, d.taxonomy, c.name, type(rel), c2.name, d.text
LIMIT 5""",
                "graph_facts": [
                    "(Neo4j AuraDB Knowledge Graph Synergy) -[:MENTIONS]-> (Neo4j) -[:PAIRS_WITH]-> (Qdrant)",
                    "(Neo4j) -[:EXECUTES]-> (Cypher)",
                    "(Knowledge Graph) -[:COMPLEMENTS]-> (Vector Space)",
                    "(Neo4j AuraDB) -[:PERFORMS]-> (Multi-Hop Traversal)"
                ],
                "vector_results": [
                    {
                        "title": "Neo4j AuraDB Knowledge Graph Synergy",
                        "score": 0.8924,
                        "taxonomy": "graph_theory",
                        "access_level": "internal",
                        "text": "Neo4j AuraDB is a fully managed cloud graph database supporting Cypher queries. Graph RAG navigates multi-hop relationship paths between entities that pure vector embeddings miss."
                    },
                    {
                        "title": "Qdrant Vector Database Architecture",
                        "score": 0.8415,
                        "taxonomy": "vector_databases",
                        "access_level": "internal",
                        "text": "Qdrant is an open-source vector similarity search engine with native support for payload-based filtering, HNSW indexing, and hardware-accelerated distance metrics for enterprise RAG."
                    }
                ],
                "audit_report": {
                    "user_clearance": "internal",
                    "selected_taxonomy": "all",
                    "accessible_count": 5,
                    "total_documents": 7,
                    "blocked_count": 2,
                    "blocked_documents": [
                        {
                            "title": "Enterprise Zero-Trust Security Protocol",
                            "access_level": "confidential",
                            "taxonomy": "security_governance",
                            "blocked_reason": "Insufficient clearance (confidential > internal)"
                        },
                        {
                            "title": "Proprietary Model Fine-Tuning Hyperparameters",
                            "access_level": "confidential",
                            "taxonomy": "security_governance",
                            "blocked_reason": "Insufficient clearance (confidential > internal)"
                        }
                    ]
                }
            }
        }
    ]

# --- Display Chat Messages ---
for i, message in enumerate(st.session_state.messages):
    role_class = "user-bubble" if message["role"] == "user" else "ai-bubble"
    label = "👤 You" if message["role"] == "user" else "🤖 Hybrid RAG Assistant"
    st.markdown(f'<div class="chat-bubble {role_class}"><b>{label}</b><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # If this message has metadata/graph inspection data
    if "details" in message:
        det = message["details"]
        with st.expander("🔍 Hybrid Retrieval Inspection (Neo4j Cypher + Qdrant + Access Audit)", expanded=(i == len(st.session_state.messages) - 1)):
            tab1, tab2, tab3, tab4 = st.tabs([
                "⚡ Executed Cypher Query",
                "🕸️ Graph Facts (Neo4j)",
                "📦 Vector Passages (Qdrant)",
                "🛡️ Access Control Audit"
            ])
            with tab1:
                st.code(det.get("cypher_query", "N/A"), language="cypher")
            with tab2:
                facts = det.get("graph_facts", [])
                if facts:
                    for f in facts:
                        st.markdown(f"- {f}")
                else:
                    st.info("No knowledge graph relationships found within current clearance policy.")
            with tab3:
                vectors = det.get("vector_results", [])
                for v in vectors:
                    st.markdown(f"**{v.get('title')}** `Score: {v.get('score')}` | `Taxonomy: {v.get('taxonomy')}` | `Clearance: {v.get('access_level')}`")
                    st.text(v.get("text"))
            with tab4:
                audit = det.get("audit_report", {})
                st.write(f"**User Clearance Applied:** `{audit.get('user_clearance')}`")
                st.write(f"**Taxonomy Domain Filter:** `{audit.get('selected_taxonomy')}`")
                st.write(f"**Accessible Documents:** `{audit.get('accessible_count', 0)} / {audit.get('total_documents', 0)}`")
                st.write(f"**Blocked Documents:** `{audit.get('blocked_count', 0)}`")
                if audit.get("blocked_documents"):
                    st.markdown("**Blocked by Security/Taxonomy Policy:**")
                    for b in audit["blocked_documents"]:
                        st.markdown(f"- 🔒 **{b.get('title')}** ({b.get('access_level')}, {b.get('taxonomy')}): _{b.get('blocked_reason')}_")

# --- Chat Input ---
if prompt := st.chat_input("Ask about RAG architecture, Cypher graphs, vector databases, or security..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and stream AI response
    with st.chat_message("assistant"):
        log_placeholder = st.empty()
        logger = StreamlitLogger(log_placeholder)
        
        answer_iterator = rag.process_query(
            prompt,
            user_clearance=clearance,
            taxonomy=taxonomy,
            stream=True,
            status_cb=lambda t: logger.log(t)
        )
        
        full_response = st.write_stream(answer_iterator)
        details = dict(rag.last_retrieval_details)
        
        # Save message with retrieval inspection details
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "details": details
        })
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Hybrid Graph RAG + Vector RAG | Powered by Neo4j AuraDB (Cypher), Qdrant Vector Store, LangGraph, and LlamaIndex")
