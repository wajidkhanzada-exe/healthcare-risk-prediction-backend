"""
build_index.py - Reads all PDFs in knowledge_base/, splits them into chunks,
generates embeddings for each chunk using Gemini, and saves everything to
knowledge_base/rag_index.json.

Run this ONCE (and again only if you add/change PDFs):
    python rag/build_index.py
(run from the project root, so the knowledge_base/ path resolves correctly)
"""

import os
import json
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "rag_index.json")

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
EMBED_BATCH_SIZE = 20


def extract_text_from_pdf(path):
    """Pulls all text out of a PDF, page by page."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Splits text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def main():
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    all_chunks = []  # each: {"text": ..., "source": filename}

    for filename in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        print(f"Extracting text from {filename}...")
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")
        for c in chunks:
            all_chunks.append({"text": c, "source": filename})

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    print("Generating embeddings (this calls the Gemini API)...")

    for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[i:i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

        for chunk, embedding in zip(batch, response.embeddings):
            chunk["embedding"] = embedding.values

        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

    with open(INDEX_PATH, "w") as f:
        json.dump(all_chunks, f)

    print(f"\nIndex saved to {INDEX_PATH}")
    print(f"File size: {os.path.getsize(INDEX_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()