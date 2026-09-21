"""
retriever.py - Given a query, finds the most relevant chunks from
knowledge_base/rag_index.json using cosine similarity on embeddings.
"""

import os
import json
import math
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "rag_index.json")

# Maps each source PDF filename to the disease(s) it's relevant to.
# Used to filter retrieval so a Diabetes query doesn't pull Heart/CKD chunks.
DISEASE_SOURCE_MAP = {
    "cdc_55506_DS1.pdf": "diabetes",
    "exercise_and_nutrition.pdf": "diabetes",
    "Healthy Eating for Adults with Chronic Kidney Disease - NIDDK.pdf": "ckd",
    "Preventing Heart Disease _ Heart Disease _ CDC.pdf": "heart",
}


class Retriever:
    def __init__(self, index_path=INDEX_PATH):
        with open(index_path) as f:
            self.chunks = json.load(f)
        print(f"[Retriever] Loaded {len(self.chunks)} chunks from {index_path}")

    def _embed_query(self, query):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

    @staticmethod
    def _cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)

    def retrieve(self, query, disease=None, top_k=3):
        """
        Returns the top_k most relevant chunks for the query.
        If `disease` is given (e.g. "diabetes"), only searches chunks
        from that disease's source documents.
        """
        query_embedding = self._embed_query(query)

        candidates = self.chunks
        if disease:
            candidates = [
                c for c in self.chunks
                if DISEASE_SOURCE_MAP.get(c["source"]) == disease
            ]

        scored = [
            (self._cosine_similarity(query_embedding, c["embedding"]), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {"text": c["text"], "source": c["source"], "score": round(score, 4)}
            for score, c in scored[:top_k]
        ]


# ----------------------------------------------------------------------
# Quick manual test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    retriever = Retriever()

    test_queries = [
        ("diabetes", "lifestyle advice for someone at high risk of diabetes"),
        ("heart", "how to prevent heart disease through diet and exercise"),
        ("ckd", "diet recommendations for chronic kidney disease patients"),
    ]

    for disease, query in test_queries:
        print(f"\n{'='*60}\nDisease: {disease} | Query: \"{query}\"\n{'='*60}")
        results = retriever.retrieve(query, disease=disease, top_k=2)
        for r in results:
            print(f"\n[score={r['score']}] from {r['source']}")
            print(r["text"][:300] + "...")