# PaperSafe — Smart Document QA & Integrity Assistant (SIH 2025)

## Project summary (SIH 2025)
PaperSafe helps conference organizers, reviewers and researchers quickly query uploaded papers, obtain concise factual answers with supporting text, and perform a light integrity check to detect duplicate content across submitted documents.

### Key features
- Multi-PDF upload and per-document FAISS indexing (embeddings cached)
- Retrieval-based QA using Groq-hosted LLaMA (model: `llama-3.3-70b-versatile`)
- Structured JSON answers with supporting sentence and source
- Simple duplicate detection across indexed documents
- Exportable audit log for review workflows

### Tech stack
- Python + Streamlit (UI)
- LangChain + FAISS + SentenceTransformers (embeddings + retrieval)
- Groq inference API (LLaMA 3.3 70B) for extraction
- Deployable on Streamlit Cloud, HuggingFace Spaces, or Docker

### Getting started (local)
1. Clone repo and create an environment:
   ```bash
   conda create -n papersafe python=3.10 -y
   conda activate papersafe
   pip install -r requirements.txt