"""
rag_generator.py - Combines ML prediction results with RAG-retrieved
medical evidence to generate a patient-friendly explanation and
lifestyle/exercise plan via Gemini.
"""

import os
import sys
import traceback
import time
from dotenv import load_dotenv
from google import genai

# Allow importing Retriever from the rag/ folder at project root
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rag"
    )
)

from retriever import Retriever


load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GENERATION_MODEL = "gemini-3.6-flash"

_retriever = None


def get_retriever():
    """Lazily loads the Retriever once (it loads a ~1.2MB index file)."""
    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


def build_prompt(
    disease_key,
    disease_display_name,
    probability,
    risk_category,
    chunks
):
    evidence_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    )

    return f"""You are a patient health communicator. A machine learning model has
already calculated this patient's risk - DO NOT invent, adjust, or restate any
different probability or risk level. Use ONLY the numbers given below.

PATIENT'S ML RESULT (do not change these numbers):
- Condition: {disease_display_name}
- Risk probability: {probability * 100:.1f}%
- Risk category: {risk_category}

TRUSTED MEDICAL EVIDENCE (use this as your source of lifestyle advice):
{evidence_text}

TASK:
Write a response with exactly two sections, using these headers:

## Explanation
2-3 short sentences explaining, in plain language, what a "{risk_category}" risk
for {disease_display_name} means for this patient. Do not give a different
number or category than the one provided above.

## Recommendations
A short bulleted list (4-6 bullets) of practical diet/exercise/lifestyle
suggestions, grounded in the medical evidence above. Be specific but concise.

End with one sentence reminding the patient this is not a medical diagnosis
and they should consult a doctor.
"""


def generate_explanation_and_plan(
    disease_key,
    disease_display_name,
    probability,
    risk_category
):
    """
    Runs RAG retrieval + Gemini generation for one disease's result.
    Returns the generated text, or a fallback message if generation fails.
    """

    retriever = get_retriever()

    query = (
        f"lifestyle diet exercise advice for "
        f"{risk_category.lower()} risk of {disease_display_name}"
    )

    chunks = retriever.retrieve(
        query,
        disease=disease_key,
        top_k=3
    )

    prompt = build_prompt(
        disease_key,
        disease_display_name,
        probability,
        risk_category,
        chunks
    )

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )

            return {
                "text": response.text,
                "sources": list(
                    set(c["source"] for c in chunks)
                )
            }

        except Exception as e:
            is_last_attempt = attempt == max_attempts

            if is_last_attempt:
                traceback.print_exc()

                return {
                    "text": (
                        "Explanation could not be generated at this time. "
                        "Please consult a doctor."
                    ),
                    "sources": [],
                    "error": str(e)
                }

            wait_seconds = 2 * attempt  # 2s, then 4s

            print(
                f"Attempt {attempt} failed ({e}), "
                f"retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)