"""
LexQA — Retrieval-Augmented Q&A over legal documents.

Pipeline: upload (PDF/TXT) -> split into chunks -> embed locally ->
store in an in-memory Chroma vector index -> retrieve top-k chunks for a
question -> answer grounded in those chunks via an LLM on OpenRouter.
"""

import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()  # loads OPENROUTER_API_KEY from a local .env file, if present

LEGAL_SYSTEM_PROMPT = """You are a legal research assistant helping an attorney review documents.
Answer the question using ONLY the context below. If the answer isn't in the context, say so plainly \
instead of guessing. Where possible, point to the specific clause or section you relied on.
This tool provides informational assistance only and is not legal advice.

Context:
{context}"""


def get_api_key() -> str | None:
    """Read the OpenRouter key from env (.env locally) or Streamlit Cloud secrets."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Runs locally (no API key, no per-call cost) — only the final question
    # and retrieved chunks are ever sent to the LLM.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)  # don't leave uploaded client documents sitting on disk


def build_chain(api_key: str, uploaded_files):
    docs = []
    for f in uploaded_files:
        docs.extend(load_document(f))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Unique collection per index so re-uploading in the same session
    # never collides with a previous in-memory Chroma collection.
    vectorstore = Chroma.from_documents(
        chunks, get_embeddings(), collection_name=f"session_{uuid.uuid4().hex[:8]}"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", LEGAL_SYSTEM_PROMPT), ("human", "{input}")]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


# ---------------------------------------------------------------- UI ----

st.set_page_config(page_title="LexQA — Legal Document Assistant", page_icon="⚖️")
st.title("⚖️ LexQA — Legal Document Assistant")
st.caption("Ask questions grounded in the contracts, briefs, or filings you upload. Not legal advice.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Document → chunks (LangChain text splitter)\n"
        "2. Chunks → embeddings (local, `all-MiniLM-L6-v2`)\n"
        "3. Embeddings → Chroma vector index\n"
        "4. Question → top-k relevant chunks retrieved\n"
        "5. Chunks + question → LLM (via OpenRouter) → grounded answer\n\n"
        "Every answer lists the source chunks it was built from, so you can verify it against the original text."
    )
    st.divider()
    st.caption("⚠️ Demo project — don't upload real client or privileged documents to a public deployment.")

api_key = get_api_key()
if not api_key:
    st.error(
        "No OpenRouter API key found. Add `OPENROUTER_API_KEY` to a local `.env` file "
        "(see `.env.example`) or to your Streamlit Cloud app secrets."
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if st.session_state.get("file_names") != current_names:
        with st.spinner("Indexing document(s)..."):
            st.session_state.chain = build_chain(api_key, uploaded_files)
            st.session_state.file_names = current_names
            st.session_state.history = []

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.chat_input("Ask a question about the uploaded document(s)")
    if query:
        with st.spinner("Searching..."):
            result = st.session_state.chain.invoke({"input": query})
        st.session_state.history.append((query, result["answer"], result["context"]))

    for question, answer, sources in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for doc in sources:
                    page = doc.metadata.get("page")
                    source = doc.metadata.get("source_file", "document")
                    label = f"**{source}**" + (f", page {page + 1}" if page is not None else "")
                    st.markdown(label)
                    snippet = doc.page_content[:400]
                    st.text(snippet + ("..." if len(doc.page_content) > 400 else ""))
else:
    st.info("Upload a document to get started — or try the sample in `sample_docs/sample_nda.txt`.")
