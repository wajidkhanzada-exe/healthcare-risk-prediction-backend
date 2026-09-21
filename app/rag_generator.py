"""
rag_generator.py

Combines ML prediction results with RAG-retrieved medical evidence
to generate a patient-friendly explanation and recommendations via Gemini.
"""

import os
import sys
import traceback
import time

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------------------------
# Load project-root .env reliably
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_FILE)


# ---------------------------------------------------------------------------
# Retriever import
# ---------------------------------------------------------------------------

RAG_DIR = os.path.join(BASE_DIR, "rag")

if RAG_DIR not in sys.path:
    sys.path.append(RAG_DIR)

from retriever import Retriever


# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from the backend .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Keep this configurable so you can change the model without editing code.
GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ---------------------------------------------------------------------------
# Retriever singleton
# ---------------------------------------------------------------------------

_retriever = None


def get_retriever():
    """
    Lazily load the Retriever once.
    """
    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(
    disease_key,
    disease_display_name,
    probability,
    risk_category,
    chunks
):
    evidence_text = "\n\n".join(
        f"[Source: {c.get('source', 'Unknown source')}]\n"
        f"{c.get('text', '')}"
        for c in chunks
    )

    if not evidence_text.strip():
        evidence_text = (
            "No retrieved medical evidence was available. "
            "Do not invent medical facts."
        )

    return f"""
You are a patient health communicator.

A machine learning model has already calculated the patient's risk.
You MUST NOT change, reinterpret, or invent the probability or risk category.

PATIENT'S ML RESULT
- Condition: {disease_display_name}
- Risk probability: {probability * 100:.1f}%
- Risk category: {risk_category}

TRUSTED MEDICAL EVIDENCE
{evidence_text}

TASK

Write exactly these two sections:

## Explanation

Write 2-3 short sentences in simple patient-friendly language explaining
what this model result means.

Do not change the probability.
Do not change the risk category.
Do not claim that the model result is a medical diagnosis.

## Recommendations

Write 4-6 concise bullet points containing practical lifestyle,
diet, physical activity, monitoring, or follow-up suggestions.

Use the trusted evidence above.
Do not invent citations.
Do not recommend medication changes.

End with one short sentence saying that this is not a medical diagnosis
and the patient should consult a qualified healthcare professional.
""".strip()


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------

def generate_explanation_and_plan(
    disease_key,
    disease_display_name,
    probability,
    risk_category
):
    """
    Runs RAG retrieval + Gemini generation.

    Returns:
        {
            "text": "...",
            "sources": [...]
        }
    """

    try:
        retriever = get_retriever()

        query = (
            f"lifestyle diet exercise monitoring advice for "
            f"{risk_category.lower()} risk of "
            f"{disease_display_name}"
        )

        chunks = retriever.retrieve(
            query,
            disease=disease_key,
            top_k=3
        )

        # Always preserve retrieved sources.
        sources = list(
            dict.fromkeys(
                c.get("source")
                for c in chunks
                if c.get("source")
            )
        )

        prompt = build_prompt(
            disease_key=disease_key,
            disease_display_name=disease_display_name,
            probability=probability,
            risk_category=risk_category,
            chunks=chunks
        )

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):

            try:
                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt
                )

                generated_text = getattr(
                    response,
                    "text",
                    None
                )

                if generated_text and generated_text.strip():

                    print(
                        f"[Gemini] Generated explanation for "
                        f"{disease_display_name}"
                    )

                    print(
                        f"[Gemini] Sources: {sources}"
                    )

                    return {
                        "text": generated_text.strip(),
                        "sources": sources
                    }

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            except Exception as e:

                print(
                    f"[Gemini] Attempt {attempt}/{max_attempts} "
                    f"failed for {disease_display_name}: {e}"
                )

                if attempt == max_attempts:
                    traceback.print_exc()

                    # IMPORTANT:
                    # Keep RAG sources even if Gemini fails.
                    return {
                        "text": (
                            "The AI-generated explanation is temporarily "
                            "unavailable. The prediction and contributing "
                            "factors shown above were generated by the "
                            "machine-learning model. Please consult a "
                            "qualified healthcare professional for "
                            "interpretation."
                        ),
                        "sources": sources,
                        "error": str(e)
                    }

                wait_seconds = 2 * attempt

                print(
                    f"[Gemini] Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    except Exception as e:

        traceback.print_exc()

        return {
            "text": (
                "The AI-generated explanation is temporarily unavailable. "
                "Please consult a qualified healthcare professional."
            ),
            "sources": [],
            "error": str(e)
        }