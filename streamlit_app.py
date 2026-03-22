import streamlit as st
import os
import time
from main import SelfHealingRAGSystem

# --- Page Config ---
st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Styling (Glassmorphism & Gradients) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #ffffff;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-bubble {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .user-bubble {
        background: rgba(58, 123, 213, 0.2);
    }
    .ai-bubble {
        background: rgba(255, 255, 255, 0.05);
    }
    .step-log {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        color: #00ff41;
        background: #000;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=100)
    st.title("Settings")
    st.info("This system uses Query Decomposition, HyDE, and CRAG to provide highly accurate, self-correcting answers.")
    if st.button("Clear History"):
        st.session_state.messages = []

# --- Custom Log Handler to capture print statements ---
class StreamlitLogger:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.logs = ""

    def log(self, text):
        self.logs += f"> {text}\n"
        self.placeholder.code(self.logs, language="bash")

# --- Initialize RAG System ---
@st.cache_resource
def get_rag_system():
    return SelfHealingRAGSystem()

rag = get_rag_system()

# --- App Structure ---
st.markdown('<div class="main-header">Self-Healing RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Research Assistant with Automatic Error Correction</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat ---
for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "ai-bubble"
    label = "👤 You" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f'<div class="chat-bubble {role_class}"><b>{label}</b><br>{message["content"]}</div>', unsafe_allow_html=True)

# --- Chat Input ---
if prompt := st.chat_input("Ask a complex question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add AI message
    with st.chat_message("assistant"):
        log_placeholder = st.empty()
        logger = StreamlitLogger(log_placeholder)
        
        with st.spinner("Analyzing and Correcting..."):
            # Note: We'd need to modify main.py slightly to accept a logger callback 
            # for real-time streaming of steps, but for now we'll just show the process.
            # For this MVP, we capture the final answer.
            answer = rag.process_query(prompt)
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Footer ---
st.markdown("---")
st.caption("Powered by LangGraph, LlamaIndex, and OpenAI")
