# app_streamlit_sih.py
"""
PaperSafe — Smart Document QA & Integrity Assistant (SIH 2025 demo)
Features:
 - Multi-PDF upload + per-file indexing (FAISS + SentenceTransformers)
 - Cross-document or per-document retrieval QA using Groq-hosted LLaMA
 - Structured JSON answers with "answer", "supporting_snippet", "source"
 - Simple plagiarism / duplicate detection via fuzzy overlap of chunks
 - Session Q/A history and CSV download
 - Sidebar controls: chunk size, overlap, k, tokens, temperature
Notes:
 - Put GROQ_API_KEY and GROQ_API_URL in Streamlit Secrets (or .env for local dev)
 - Keep temperature low (0.0) for deterministic extraction
"""
from dotenv import load_dotenv
load_dotenv()  # local dev only

import streamlit as st
import os, json, requests, hashlib, time
import pandas as pd
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from difflib import SequenceMatcher

# ------------------ Config / Util ------------------
st.set_page_config(page_title="PaperSafe (SIH2025)", layout="wide")
APP_TITLE = "📚 PaperSafe — Document QA & Integrity Assistant (SIH 2025)"
st.title(APP_TITLE)

# Helpers
def file_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def parse_json_safe(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        # attempt to find a JSON substring
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e+1])
            except Exception:
                return None
        return None

# ------------------ LLM wrapper (Groq) ------------------
class GroqRemoteLLM:
    def __init__(self, api_url=None, api_key=None, model=None, timeout=60, temperature=0.0):
        self.api_url = api_url or os.getenv("GROQ_API_URL")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout
        self.temperature = float(temperature)
    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        if not self.api_url or not self.api_key:
            raise RuntimeError("GROQ_API_URL or GROQ_API_KEY missing (set in env or Streamlit Secrets).")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages":[{"role":"system","content":"You are an information extraction assistant. Answer only from the context."},{"role":"user","content":prompt}], "max_tokens": max_tokens, "temperature": float(self.temperature)}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        out = resp.json()
        # robust extraction
        if isinstance(out, dict):
            choices = out.get("choices", [])
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    if "text" in first:
                        return first["text"]
@st.cache_resource
def get_embeddings(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)

@st.cache_resource
def build_faiss(_docs, embed_model_name="all-MiniLM-L6-v2"):
    emb = get_embeddings(embed_model_name)
    return FAISS.from_documents(_docs, emb)

    emb = get_embeddings(embed_model_name)
    return FAISS.from_documents(docs, emb)

# ------------------ Sidebar controls ------------------
st.sidebar.header("Settings")
CHUNK_SIZE = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=900, step=100)
CHUNK_OVERLAP = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=400, value=80, step=10)
RETRIEVER_K = st.sidebar.number_input("Retriever k (per doc)", min_value=1, max_value=10, value=3, step=1)
MAX_TOKENS = st.sidebar.number_input("Max tokens (LLM)", min_value=64, max_value=1024, value=256, step=32)
TEMPERATURE = st.sidebar.slider("Temperature (determinism)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
EMBED_MODEL = st.sidebar.selectbox("Embedding model", options=["all-MiniLM-L6-v2"], index=0)
st.sidebar.markdown("---")
st.sidebar.write("NOTE: Add GROQ_API_KEY and GROQ_API_URL in Streamlit Secrets (Manage app → Secrets).")

# ------------------ State init ------------------
if "index" not in st.session_state:
    st.session_state["index"] = {}  # cache_key -> {filename, path, docs, faiss}
if "qa_log" not in st.session_state:
    st.session_state["qa_log"] = []

# ------------------ UI: uploads ------------------
uploaded = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        try:
            b = f.read()
            fh = file_hash(b)
            cache_key = f"{fh}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{EMBED_MODEL}"
            if cache_key in st.session_state["index"]:
                st.info(f"{f.name} already indexed for current settings.")
                continue
            # save to disk
            path = f"uploaded_{fh}.pdf"
            with open(path, "wb") as out:
                out.write(b)
            st.info(f"Indexing {f.name} ... (this may take time)")
            loader = PyPDFLoader(path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            docs = splitter.split_documents(pages)
            faiss = build_faiss(docs, embed_model_name=EMBED_MODEL)
            st.session_state["index"][cache_key] = {"filename": f.name, "path": path, "docs": docs, "faiss": faiss}
            st.success(f"Indexed {f.name} — {len(docs)} chunks.")
        except Exception as e:
            st.error(f"Failed to index {f.name}: {e}")

# ------------------ Document selection ------------------
docs_list = list(st.session_state["index"].items())
if not docs_list:
    st.info("No documents indexed yet. Upload PDFs to begin.")
# Build selection options
doc_options = ["All documents"]
for ck, meta in docs_list:
    doc_options.append(f"{meta['filename']} ({len(meta['docs'])} chunks) | key:{ck[:8]}")

selected = st.selectbox("Select document (or All documents):", options=doc_options)

# ------------------ Querying ------------------
query = st.text_input("Ask a question about indexed documents")
if st.button("Run Query") and query:
    # build list of selected cache_keys
    if selected == "All documents":
        keys = list(st.session_state["index"].keys())
    else:
        # find matching key
        parts = selected.split("| key:")
        if len(parts) == 2:
            key_suffix = parts[1].strip()
            keys = [k for k in st.session_state["index"].keys() if k.startswith(key_suffix)]
            if not keys:
                # fallback try simpler match
                keys = [k for k, m in st.session_state["index"].items() if selected.startswith(m["filename"])]
        else:
            keys = list(st.session_state["index"].keys())

    if not keys:
        st.error("No matching documents selected or indexed.")
    else:
        # Gather top chunks from each doc
        aggregated = []
        for ck in keys:
            meta = st.session_state["index"][ck]
            faiss = meta["faiss"]
            retriever = faiss.as_retriever(search_kwargs={"k": RETRIEVER_K})
            docs_ret = retriever.get_relevant_documents(query)
            for i, d in enumerate(docs_ret):
                aggregated.append({"cache_key": ck, "filename": meta["filename"], "chunk_index": i+1, "doc": d, "text": d.page_content})

        # sort optionally by nothing (FAISS ordering kept). Limit to top N
        TOP_N = min(8, len(aggregated))
        top_chunks = aggregated[:TOP_N]
        # Combine into context with separators + chunk labels
        context = "\n\n----\n\n".join([f"DOCUMENT: {c['filename']} | CHUNK {c['chunk_index']}\n{c['text']}" for c in top_chunks])

        # Build JSON extraction prompt asking for supporting snippet & source
        prompt_template = """
You are an information extraction assistant. Use ONLY the context below (top retrieved chunks) to answer the question.

Context:
{context}

Question:
{question}

Task:
1) Provide a concise factual answer (1-2 sentences) to the question.
2) Provide the exact supporting sentence from the context (field: supporting_snippet).
3) Provide the source filename and chunk label in field: source (e.g. "paper.pdf | CHUNK 2").
Return ONLY a JSON object with keys: answer, supporting_snippet, source.
If the answer is not present in the context, set "answer" to "Not stated in the document" and other fields to null.
"""
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        prompt = QA_PROMPT.format(context=context, question=query)

        # Call LLM (Groq)
        llm = GroqRemoteLLM(temperature=TEMPERATURE)
        try:
            with st.spinner("Querying model..."):
                raw = llm(prompt, max_tokens=MAX_TOKENS)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            raw = None

        parsed = parse_json_safe(raw) if raw else None
        # fallback if parsing fails: ask simpler extraction
        if not parsed and raw:
            try:
                fallback_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer in two lines: first the answer, second the supporting sentence (prefixed 'Support:')."
                raw2 = llm(fallback_prompt, max_tokens=MAX_TOKENS)
                if "Support:" in raw2:
                    a, s = raw2.split("Support:", 1)
                    parsed = {"answer": a.strip(), "supporting_snippet": s.strip(), "source": None}
                else:
                    parsed = {"answer": raw2.strip(), "supporting_snippet": None, "source": None}
            except Exception:
                parsed = None

        # Show results
        st.subheader("Model output (structured)")
        if parsed:
            st.json(parsed)
            st.markdown("**Answer:**")
            st.write(parsed.get("answer"))
            if parsed.get("supporting_snippet"):
                with st.expander("Supporting snippet"):
                    st.write(parsed.get("supporting_snippet"))
                    if parsed.get("source"):
                        st.caption(f"Source: {parsed.get('source')}")
        else:
            st.error("Model did not return parseable JSON. Raw output shown below.")
            st.code(raw or "(no output)")

        # Simple plagiarism / duplicate detection: check if supporting_snippet appears (or similar) in other documents
        try:
            snippet = (parsed.get("supporting_snippet") or "").strip()
            duplicates = []
            if snippet:
                for ck, meta in st.session_state["index"].items():
                    # check all chunks for similarity
                    for d in meta["docs"]:
                        sim = similar(snippet, d.page_content[:len(snippet)+200])  # quick heuristic
                        if sim > 0.85 and meta["filename"] not in [x["filename"] for x in duplicates]:
                            duplicates.append({"filename": meta["filename"], "similarity": round(sim, 3)})
            if duplicates:
                st.warning("Possible duplicate content found in other documents:")
                st.table(pd.DataFrame(duplicates))
            else:
                st.success("No obvious duplicates found for the supporting snippet among indexed docs.")
        except Exception as e:
            st.error("Error while checking duplicates: " + str(e))

        # Save QA audit log
        st.session_state["qa_log"].append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": query,
            "answer": parsed.get("answer") if parsed else (raw or ""),
            "supporting_snippet": parsed.get("supporting_snippet") if parsed else None,
            "source": parsed.get("source") if parsed else None,
            "top_docs": ";".join({c["filename"] for c in top_chunks})
        })

# ------------------ History & export ------------------
st.markdown("---")
st.subheader("Session QA Audit Log")
if st.session_state["qa_log"]:
    df = pd.DataFrame(st.session_state["qa_log"])
    st.dataframe(df.sort_values(by="timestamp", ascending=False))
    b = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download audit log (CSV)", data=b, file_name="papersafe_audit_log.csv", mime="text/csv")
else:
    st.write("No QA events in this session yet.")

# Debug toggle
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("Indexed docs keys:", list(st.session_state["index"].keys()))
    st.sidebar.write("Env GROQ_API_URL set:", bool(os.getenv("GROQ_API_URL")))
    st.sidebar.write("Number of indexed documents:", sum(len(meta["docs"]) for meta in st.session_state["index"].values()) if st.session_state["index"] else 0)
