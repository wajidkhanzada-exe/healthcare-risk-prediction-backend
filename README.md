# Healthcare Risk Prediction — Backend

A Flask-based backend that predicts a patient's risk of **Diabetes**, **Heart
Disease**, and **Chronic Kidney Disease (CKD)** from clinical data, then
generates a plain-language explanation and lifestyle/exercise recommendations
for each result using Retrieval-Augmented Generation (RAG) grounded in
trusted medical sources, powered by Google Gemini.

This is the API layer for the project. The React frontend lives in a
separate repository.

---

## Overview

- Three independently trained, calibrated ML models (one per disease),
  built with scikit-learn / XGBoost.
- A single `/report/full` endpoint accepts partial patient data, runs
  whichever predictions have enough information, imputes the rest using
  statistics learned from training, and returns a full report per disease.
- Each disease result includes a SHAP-based explanation of which patient
  features most influenced the prediction.
- A RAG pipeline retrieves the most relevant passages from a small
  knowledge base of official health guidelines (CDC, NIDDK, ADA) and feeds
  them to Gemini to generate a grounded, patient-friendly explanation and
  recommendations — Gemini never invents its own risk numbers, it only
  explains the numbers the ML models produced.
- A Gemini-vision-based extractor can read an uploaded lab report (PDF or
  image) and pre-fill known fields automatically.

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| ML models | scikit-learn, XGBoost, joblib |
| Explainability | SHAP |
| LLM / embeddings | Google Gemini (`google-genai` SDK) |
| PDF/image parsing | `pdfplumber`, Gemini multimodal input |
| Auth (verification) | Supabase JWT (verified per-request) |

## Project Structure

```
healthcare-risk-prediction/
├── app/
│   ├── __init__.py        # Flask app factory, loads HealthPredictor once at startup
│   ├── predictor.py       # Core inference: validation, imputation, prediction, SHAP
│   ├── extractor.py       # Gemini-based lab report field extraction
│   ├── rag_generator.py   # RAG retrieval + Gemini explanation/recommendations
│   └── routes.py          # All API endpoints
├── rag/
│   ├── build_index.py     # One-time script: chunks knowledge_base/ PDFs into embeddings
│   └── retriever.py       # Cosine-similarity retrieval over the embedded chunks
├── knowledge_base/        # Source PDFs (CDC, NIDDK, ADA guidelines) + generated rag_index.json
├── saved_models/          # Per-disease preprocessor + calibrated model (.joblib) + metadata.json
├── requirements.txt
├── .env                   # GEMINI_API_KEY, SUPABASE_*  (not committed)
└── run.py                 # Application entry point
```

## Getting Started

### Prerequisites
- Python 3.11
- A Gemini API key ([Google AI Studio](https://aistudio.google.com))
- A Supabase project (for auth token verification)

### Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
```

### Build the RAG index (first run only, or whenever `knowledge_base/` changes)

```bash
python rag/build_index.py
```

### Run the server

```bash
python run.py
```

By default this starts in production-safe mode (debug off). For local
development with auto-reload:

```bash
# Windows PowerShell
$env:FLASK_DEBUG="1"
python run.py
```

The API is served at `http://127.0.0.1:5000`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/diseases` | Returns the field schema (names, types, allowed values, ranges) for all three diseases — used by the frontend to build its form dynamically |
| `POST` | `/predict/<disease_key>` | Runs a single disease prediction. `disease_key` is `diabetes`, `heart`, or `ckd` |
| `POST` | `/predict/full` | Runs predictions for whichever diseases have enough data; skips the rest with a reason |
| `POST` | `/report/full` | Same as `/predict/full`, plus a Gemini-generated, RAG-grounded explanation and recommendations for each result |
| `POST` | `/extract-report` | Accepts a multipart file upload (PDF/image) and returns extracted field values |

### Example: `/report/full`

Request body (fields may be partial — anything omitted is imputed):

```json
{
  "diabetes": { "age": 45, "bmi": 27.3, "HbA1c_level": 6.1, "blood_glucose_level": 140, "gender": "Female", "smoking_history": "never", "hypertension": 0, "heart_disease": 0 },
  "heart": { "Age": 46, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 115, "Cholesterol": 237, "FastingBS": 0, "ExerciseAngina": "Y" },
  "ckd": { "Age": 66, "Gender": 1, "BMI": 32.4, "SystolicBP": 91, "SerumElectrolytesSodium": 141.4, "SerumElectrolytesPotassium": 3.8, "HbA1c": 6.6, "Smoking": 0, "FastingBloodSugar": 135.4, "CholesterolTotal": 183 }
}
```

Response (abridged):

```json
{
  "results": {
    "diabetes": {
      "disease": "Diabetes",
      "probability": 0.0117,
      "risk_category": "Low",
      "data_completeness": 100,
      "estimated_fields": [],
      "top_factors": [
        { "feature": "HbA1c Level", "direction": "decreased", "impact": 0.031 }
      ],
      "ai_explanation": "## Explanation\n...",
      "evidence_sources": ["cdc_55506_DS1.pdf"]
    }
  },
  "skipped": {},
  "summary": {
    "highest_risk_disease": "...",
    "highest_risk_category": "...",
    "diseases_flagged_high": []
  }
}
```

## Data Completeness & Partial Predictions

Each disease has a small set of **critical fields** (e.g. HbA1c and glucose
for diabetes; sodium/potassium for CKD). If those are missing, the disease
is skipped rather than predicted from irrelevant defaults. Any other field
is optional: if omitted, it is imputed using the median/most-frequent value
learned from the training data, exactly as the trained scikit-learn
pipeline does. Every response reports `data_completeness` and
`estimated_fields` so the caller knows how much of the result was
estimated versus provided.

## Model Details

| Disease | Final Model | Notes |
|---|---|---|
| Diabetes | XGBoost (calibrated) | Global thresholds: Low < 0.30, Medium < 0.70 |
| Heart Disease | Logistic Regression (calibrated) | Global thresholds: Low < 0.30, Medium < 0.70 |
| CKD | Random Forest (calibrated) | Disease-specific thresholds (Low < 0.84, Medium < 0.96) — derived from out-of-fold percentiles because the CKD dataset is ~92% positive prevalence |

All models are wrapped in `CalibratedClassifierCV` (sigmoid) so their
output probabilities are well-calibrated, and all predictions are
verified bit-exact against the training environment (see
`saved_models/metadata.json` for the exact library versions used at
training time).

## Disclaimers Built Into the System

- CKD results always include a note that the dataset has a weak
  specificity signal and that even a "Low" result should not be treated
  as reassurance.
- Every generated report ends with a reminder that this is not a medical
  diagnosis and a doctor should be consulted.
- Gemini is explicitly prompted to only explain the ML models' numbers,
  never to generate its own probability or risk category.

## Security Notes

- The Gemini API key and Supabase JWT secret are only ever read
  server-side from environment variables; they are never exposed to the
  frontend.
- CORS is restricted to the known frontend origins.
- Debug mode defaults to off and is only enabled via the `FLASK_DEBUG`
  environment variable.

## Known Limitations

- The CKD training dataset is heavily imbalanced (~92% positive), which
  limits how confidently the model can identify true negatives.
- Free-tier Gemini API quotas are limited; the RAG/explanation and
  extraction calls include automatic retries for transient errors.
