"""
extractor.py - Uses Gemini's multimodal capability to read an uploaded
lab report (PDF or image) and extract structured field values matching
the frontend's FIELD_CATALOG field ids.
"""

import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GENERATION_MODEL = "gemini-2.5-flash"

# Field ids MUST match the "id" values in the frontend's FIELD_CATALOG.
# Keep this list in sync if the frontend catalog changes.
EXTRACTABLE_FIELDS = """
- age (years)
- gender (Male/Female)
- bmi (Body Mass Index)
- systolicBP (systolic blood pressure, mmHg)
- diastolicBP (diastolic blood pressure, mmHg)
- bloodGlucose (random blood glucose, mg/dL)
- fastingBloodSugar (fasting blood sugar, mg/dL)
- hba1c (HbA1c %)
- totalCholesterol (mg/dL)
- ldl (LDL cholesterol, mg/dL)
- hdl (HDL cholesterol, mg/dL)
- triglycerides (mg/dL)
- creatinine (serum creatinine, mg/dL)
- bun (blood urea nitrogen, mg/dL)
- gfr (glomerular filtration rate, mL/min)
- proteinInUrine (g/day)
- acr (albumin-creatinine ratio, mg/g)
- sodium (serum sodium, mEq/L)
- potassium (serum potassium, mEq/L)
- calcium (serum calcium, mg/dL)
- phosphorus (serum phosphorus, mg/dL)
- hemoglobin (g/dL)
- hematocrit (%)"""

def _build_valid_ids():
    valid_ids = set()
    for line in EXTRACTABLE_FIELDS.strip().split("\n"):
        cleaned = line.strip().lstrip("-").strip()
        if cleaned:
            valid_ids.add(cleaned.split(" ")[0])
    return valid_ids


VALID_FIELD_IDS = _build_valid_ids()


def extract_fields_from_document(file_bytes, mime_type, max_attempts=3):
    """
    Sends the uploaded file to Gemini and asks it to extract lab values.
    Returns a dict of {field_id: value}. Only fields it actually found
    in the document are included - missing fields are simply omitted,
    NOT guessed or defaulted.

    Retries on transient server errors (e.g. 503 UNAVAILABLE) before
    giving up.
    """
    prompt = f"""You are extracting lab values from a medical report for a health
risk prediction form. Look at the attached document (PDF or image) and extract
ONLY the following fields, if present:

{EXTRACTABLE_FIELDS}

Rules:
- Return ONLY a JSON object, with field ids as keys exactly as listed above.
- Only include a field if you actually found its value in the document.
- Do NOT guess, estimate, or make up a value for anything not clearly stated.
- Numbers should be plain numbers (no units in the value).
- gender should be exactly "Male" or "Female" if found.
- If you find NOTHING usable in the document, return an empty JSON object {{}}.
"""

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            try:
                extracted = json.loads(response.text)
            except (json.JSONDecodeError, TypeError):
                return {}

            return {
                k: v for k, v in extracted.items()
                if k in VALID_FIELD_IDS and v is not None
            }

        except Exception as e:
            last_error = e
            is_last_attempt = attempt == max_attempts
            if is_last_attempt:
                break
            wait_seconds = 2 * attempt
            print(f"[extractor] Attempt {attempt} failed ({e}), "
                  f"retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    print(f"[extractor] All {max_attempts} attempts failed: {last_error}")
    return {}