import json
import os
import joblib
import streamlit as st
import subprocess
import sys
from src.rag_engine import answer_query

ART = "artifacts"
ARTIFACT_FILES = (
    "messages.json",
    "topics.json",
    "checkpoints.json",
    "persona.json",
    "vectorizer.joblib",
    "message_matrix.joblib",
    "topic_matrix.joblib",
    "checkpoint_matrix.joblib",
)

st.set_page_config(page_title="Conversation RAG + Persona Bot", layout="wide")
st.title("Conversation RAG + Persona Chatbot")
st.caption("Chronological topic checkpoints, 100-message checkpoints, RAG retrieval, and persona extraction.")

def artifact_signature():
    if not os.path.exists(ART):
        return None
    signature = []
    for filename in ARTIFACT_FILES:
        path = os.path.join(ART, filename)
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        signature.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


@st.cache_resource
def load_artifacts(signature):
    if not os.path.exists(ART):
        return None
    with open(os.path.join(ART, "messages.json"), "r", encoding="utf-8") as f:
        messages = json.load(f)
    with open(os.path.join(ART, "topics.json"), "r", encoding="utf-8") as f:
        topics = json.load(f)
    with open(os.path.join(ART, "checkpoints.json"), "r", encoding="utf-8") as f:
        checkpoints = json.load(f)
    with open(os.path.join(ART, "persona.json"), "r", encoding="utf-8") as f:
        persona = json.load(f)
    return {
        "messages": messages,
        "topics": topics,
        "checkpoints": checkpoints,
        "persona": persona,
        "vectorizer": joblib.load(os.path.join(ART, "vectorizer.joblib")),
        "matrices": {
            "message_matrix": joblib.load(os.path.join(ART, "message_matrix.joblib")),
            "topic_matrix": joblib.load(os.path.join(ART, "topic_matrix.joblib")),
            "checkpoint_matrix": joblib.load(os.path.join(ART, "checkpoint_matrix.joblib")),
        }
    }

artifact_sig = artifact_signature()
art = load_artifacts(artifact_sig) if artifact_sig else None
if art is None:
    st.warning("Artifacts not found. Building index now. First run may take a few minutes...")
    with st.spinner("Parsing CSV, creating checkpoints, extracting persona, and building retrieval index..."):
        result = subprocess.run([sys.executable, "src/build_index.py", "--csv", "data/conversations.csv"], capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Index build failed. Check logs below.")
        st.code(result.stderr)
        st.stop()
    st.success("Index built successfully. Reloading artifacts...")
    st.cache_resource.clear()
    artifact_sig = artifact_signature()
    art = load_artifacts(artifact_sig) if artifact_sig else None
    if art is None:
        st.error("Artifacts still not found after build.")
        st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Messages", len(art["messages"]))
col2.metric("Topic checkpoints", len(art["topics"]))
col3.metric("100-message checkpoints", len(art["checkpoints"]))

tab_chat, tab_topics, tab_persona = st.tabs(["Chatbot", "Topic Checkpoints", "Persona JSON"])

with tab_chat:
    q = st.text_input("Ask a question", value="What kind of person is this user?")
    if st.button("Ask", type="primary"):
        result = answer_query(q, art["vectorizer"], art["matrices"], {
            "messages": art["messages"],
            "topics": art["topics"],
            "checkpoints": art["checkpoints"],
        }, art["persona"])

        st.markdown("---")
        st.header("🔍 AI Insight")
        st.markdown(result["answer"])

        st.markdown("---")
        st.subheader("🛠️ Retrieval Context")
        
        tab1, tab2 = st.tabs(["Topic Summaries", "Message Evidence"])
        
        with tab1:
            if result["retrieved_topics"]:
                for t in result["retrieved_topics"]:
                    st.markdown(f"**Topic {t['topic_id']}** (Score: {t['score']:.3f})")
                    st.info(t["summary"])
            else:
                st.write("No relevant topics found.")

        with tab2:
            if result["retrieved_messages"]:
                for m in result["retrieved_messages"]:
                    st.markdown(f"**Message {m['msg_id']}** - *{m['speaker']}* (Score: {m['score']:.3f})")
                    st.write(m["text"])
            else:
                st.write("No relevant messages found.")


with tab_topics:
    st.subheader("📂 Chronological Topic Segments")
    st.caption("Segments identified via embedding similarity shifts and size constraints.")
    for t in art["topics"][:100]:
        with st.expander(f"Topic {t['topic_id']} | {t['message_count']} messages | {t['start_msg_id']} to {t['end_msg_id']}"):
            st.write("**Keywords:**", ", ".join(t["keywords"]))
            st.write("**Automated Summary:**")
            st.info(t["summary"])
            st.write("**Sample Messages:**")
            st.json(t["sample_messages"])

with tab_persona:
    st.subheader("👤 Extracted User Persona")
    st.caption("Evidence-driven persona traits extracted from meaningful conversation segments.")
    
    p = art["persona"]
    c1, c2 = st.columns(2)
    c1.json(p)
    
    with c2:
        st.write("### Stats")
        stats = p.get('stats', {})
        st.write(f"- **Total Messages Analyzed:** {stats.get('total_messages', 0)}")
        st.write(f"- **Meaningful Messages:** {stats.get('meaningful_messages', 'N/A')}")
        st.write(f"- **Avg Length:** {stats.get('avg_message_length', 0)} chars")
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.stTextInput>div>div>input {
    background-color: #161b22;
    color: white;
    border-radius: 10px;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

.stMetric {
    background-color: #161b22;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
