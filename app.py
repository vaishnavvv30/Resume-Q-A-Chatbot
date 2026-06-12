import streamlit as st
import tempfile
import os

from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -----------------------------
# Gemini + Local Embeddings
# -----------------------------

Settings.llm = Gemini(
    model_name="models/gemini-1.5-flash",
    api_key=st.secrets["GEMINI_API_KEY"]
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="DocChat AI",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Choose PDF or DOCX",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:

        try:

            ext = os.path.splitext(uploaded_file.name)[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext
            ) as tmp:

                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            reader = PDFReader() if ext == ".pdf" else DocxReader()

            documents = reader.load_data(file=tmp_path)

            index = VectorStoreIndex.from_documents(documents)

            st.session_state.query_engine = index.as_query_engine()

            st.success("Document loaded successfully!")

            os.unlink(tmp_path)

        except Exception as e:
            st.error(str(e))

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Main Page
# -----------------------------

st.title("DocChat AI")
st.caption("Upload a document and ask questions about it.")

# Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input

prompt = st.chat_input(
    "Ask something about your document..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.query_engine is None:

        response = "Please upload a PDF or DOCX file first."

    else:

        with st.spinner("Thinking..."):

            response = str(
                st.session_state.query_engine.query(prompt)
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)
