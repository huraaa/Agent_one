from pypdf import PdfReader

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk.strip())
        i += max(1, max_chars - overlap)
    return [c for c in chunks if c]
