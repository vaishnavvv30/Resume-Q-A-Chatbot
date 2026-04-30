import streamlit as st
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core import VectorStoreIndex
import tempfile, os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="DocChat AI", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; }

/* Hide sidebar collapse button */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161822 !important;
    border-right: 1px solid #2d2d3f;
}

/* File uploader cleanup */
[data-testid="stFileUploaderDropzone"] button span { display: none !important; }
[data-testid="stFileUploaderDropzone"] button::after { content: "Browse File"; color: #a5b4fc; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] small { display: none !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #1a1b2e !important;
    border: 1px solid #3730a3 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    padding: 14px 18px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,.3) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border-radius: 12px !important;
    color: #fff !important;
}

/* Custom chat bubbles */
.chat-wrap { display: flex; flex-direction: column; gap: 12px; margin-bottom: 1rem; }

.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    box-shadow: 0 2px 12px rgba(99,102,241,.35);
    line-height: 1.55;
    font-size: 0.95rem;
}
.bubble-bot {
    align-self: flex-start;
    background: #1e1e2e;
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    max-width: 75%;
    border: 1px solid #2d2d3f;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
    line-height: 1.55;
    font-size: 0.95rem;
}

.app-title {
    text-align: center; padding: 1.5rem 0 .5rem;
    background: linear-gradient(135deg,#6366f1,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;
}
.app-subtitle { text-align: center; color: #64748b; font-size: .95rem; margin-bottom: 1.5rem; }

.status-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: .8rem; font-weight: 500;
}
.badge-ready { background: #064e3b; color: #6ee7b7; }
.badge-wait  { background: #451a03; color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">DocChat AI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a document and start asking questions</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("Upload", type=["pdf", "docx"], label_visibility="collapsed")

    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            reader = PDFReader() if ext == ".pdf" else DocxReader()
            documents = reader.load_data(file=tmp_path)
            index = VectorStoreIndex.from_documents(documents)
            st.session_state.query_engine = index.as_query_engine()
            st.markdown(f'<span class="status-badge badge-ready">✓ {uploaded_file.name} loaded</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        st.markdown('<span class="status-badge badge-wait">No document uploaded</span>', unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history (plain markdown, no st.chat_message) ────────────────────────
bubbles_html = '<div class="chat-wrap">'
for msg in st.session_state.messages:
    cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
    bubbles_html += f'<div class="{cls}">{msg["content"]}</div>'
bubbles_html += '</div>'
st.markdown(bubbles_html, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.query_engine is None:
        reply = "Please upload a PDF or DOCX file first."
    else:
        with st.spinner("Thinking…"):
            reply = str(st.session_state.query_engine.query(prompt))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()