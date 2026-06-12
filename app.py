import streamlit as st
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core import VectorStoreIndex
import tempfile, os, requests
from dotenv import load_dotenv
load_dotenv()

# Simple Gemini (Google Generative API) LLM adapter for Llama-Index
# Try multiple possible import locations for ServiceContext (llama-index API varies)
ServiceContext = None
import importlib
try:
    mod = importlib.import_module("llama_index")
    ServiceContext = getattr(mod, "ServiceContext", None)
except Exception:
    try:
        mod = importlib.import_module("llama_index.service_context")
        ServiceContext = getattr(mod, "ServiceContext", None)
    except Exception:
        ServiceContext = None
import asyncio
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SimpleLLMPredictor:
    def __init__(self, llm):
        self._llm = llm

    def predict(self, prompt: str, **kwargs) -> str:
        if hasattr(self._llm, "predict"):
            return self._llm.predict(prompt, **kwargs)
        if hasattr(self._llm, "_generate"):
            return self._llm._generate(prompt)
        raise NotImplementedError("Underlying LLM does not implement predict")

    async def apredict(self, prompt: str, **kwargs) -> str:
        if hasattr(self._llm, "apredict"):
            return await self._llm.apredict(prompt, **kwargs)
        if hasattr(self._llm, "predict"):
            return await asyncio.to_thread(self._llm.predict, prompt, **kwargs)
        if hasattr(self._llm, "_generate"):
            return await asyncio.to_thread(self._llm._generate, prompt)
        raise NotImplementedError("Underlying LLM does not implement apredict")

class GeminiLLM:
    def __init__(self, api_key=None, model=None, temperature=0.0):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "models/text-bison-001")
        self.temperature = temperature

    def _generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")
        url = f"https://generativelanguage.googleapis.com/v1beta2/{self.model}:generate?key={self.api_key}"
        payload = {"prompt": {"text": prompt}, "temperature": self.temperature}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Try common response shapes
        if isinstance(data, dict):
            if "candidates" in data and len(data["candidates"]) > 0:
                cand = data["candidates"][0]
                return cand.get("output") or cand.get("content") or cand.get("text") or ""
            # fallback keys
            if "output" in data:
                return data.get("output")
        return ""

    # Llama-Index LLMPredictor expects an object with synchronous and async predict methods.
    def predict(self, prompt: str, **kwargs) -> str:
        return self._generate(prompt)

    async def apredict(self, prompt: str, **kwargs) -> str:
        return self._generate(prompt)


class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for local embeddings. Install it via `pip install sentence-transformers`.")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # returns list of vectors
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

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

            # Create a simple predictor backed by Gemini and attach via ServiceContext
            gemini_llm = GeminiLLM()
            llm_predictor = SimpleLLMPredictor(gemini_llm)

            # Prefer local embeddings to avoid OpenAI dependency
            local_emb = None
            try:
                local_emb = LocalEmbeddings()
            except Exception:
                local_emb = None

            if ServiceContext is not None:
                # Prefer explicit local embedding model if available to avoid OpenAI defaults
                try:
                    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model='local')
                except Exception:
                    # fallback without explicit embed_model
                    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            else:
                # Fall back to creating the index without ServiceContext.
                # Try to pass embed_model if VectorStoreIndex supports it.
                try:
                    index = VectorStoreIndex.from_documents(documents, embed_model='local')
                except Exception:
                    index = VectorStoreIndex.from_documents(documents)
                try:
                    if hasattr(index, "_service_context") and index._service_context is not None:
                        index._service_context.llm_predictor = llm_predictor
                        if local_emb is not None:
                            index._service_context.embed_model = local_emb
                except Exception:
                    pass
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
