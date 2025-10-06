import os, uuid
from typing import List, Dict
from rag.chunking import load_pdf_text, chunk_text
from tools.retriever import add_documents

def ingest_pdfs(paths: List[str], max_chars=1200, overlap=200):
    docs: List[Dict[str,str]] = []
    for p in paths:
        txt = load_pdf_text(p)
        for i, chunk in enumerate(chunk_text(txt, max_chars=max_chars, overlap=overlap)):
            docs.append({"id": f"{os.path.basename(p)}-{i}-{uuid.uuid4().hex[:6]}", "text": chunk, "source": p})
    if docs:
        add_documents(docs)
