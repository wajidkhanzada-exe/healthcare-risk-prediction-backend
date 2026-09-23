# Healthcare Risk Prediction — Backend

A Flask API that predicts risk for three chronic diseases (Diabetes, Heart Disease, Chronic Kidney Disease) from partial patient data, explains each prediction with SHAP, and generates a grounded, patient-friendly report using Retrieval-Augmented Generation (RAG) over official medical sources and Gemini.

**Live API:** https://healthcare-risk-prediction-backend.vercel.app/
**Live App:** https://healthriskkk-ai.vercel.app/
**Frontend repo:** https://github.com/wajidkhanzada-exe/healthcare-risk-prediction-frontend

---

## Overview

Most "AI health risk" demos train a model on a clean, complete dataset and stop there. This project is built around a harder and more realistic constraint: **real patients almost never provide complete data.** A patient might upload one lab report, or none, and still expect a useful, honestly-caveated answer.

The system is designed around that constraint end-to-end — from how missing values are imputed, to how uncertainty is surfaced to the user, to how much a single disease's weak dataset is allowed to inflate its reported confidence.

```
Patient
  │
  ├── Uploads a lab report (PDF/image)  ──►  Gemini multimodal extraction  ──► partial field values
  ├── Fills in what they know (form)    ──────────────────────────────────►
  │
  ▼
Patient profile (partial, per-disease)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  For each disease (Diabetes / Heart / CKD):              │
│    1. Validate provided values                           │
│    2. Impute missing fields (trained SimpleImputer)      │
│    3. Predict probability (calibrated model)              │
│    4. Map probability → Low / Medium / High               │
│    5. Explain via SHAP → top contributing factors         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
RAG retrieval over CDC / ADA / NIDDK source documents
  │
  ▼
Gemini — writes explanation + lifestyle plan
(never invents a probability; only explains ML output)
  │
  ▼
JSON report → frontend
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| API framework | Flask |
| ML | scikit-learn 1.6.1, XGBoost 3.2.0 |
| Explainability | SHAP (model-agnostic `Explainer`) |
| LLM | Google Gemini (`google-genai`) |
| RAG | Custom chunking + `gemini-embedding-001` embeddings |
| PDF/image extraction | Gemini multimodal input |
| Auth | Supabase Auth (email OTP) |
| Hosting | Vercel (serverless) |
| Language / runtime | Python 3.11 |

---

## ML Pipeline

Trained and validated in a separate Colab notebook; artifacts (models, preprocessors, metadata, SHAP backgrounds) are exported via `joblib` and loaded once at server startup.

| Disease | Dataset | Rows (after cleaning) | Best model | ROC-AUC | Recall |
|---|---|---|---|---|---|
| Diabetes | [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle) | 96,146 | XGBoost | 0.978 | 0.92 |
| Heart Disease | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle) | 913 | Logistic Regression | 0.900 | 0.88 |
| Chronic Kidney Disease | [CKD Dataset](https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis) (Kaggle, synthetic) | 1,659 | Random Forest | ~0.83 | see limitations |

All three models are wrapped in `CalibratedClassifierCV` (sigmoid, 5-fold), so a reported probability is meant to reflect an actual frequency, not just a ranking score.

---

## Key Engineering Decisions

These are the decisions that mattered most, and why — not just "what the code does."

**1. Class imbalance breaks a single global risk threshold.**
Diabetes and Heart Disease use fixed 0.30 / 0.70 thresholds for Low/Medium/High. The CKD dataset is ~92% positive, so a global threshold pushes almost every patient into "High." CKD instead uses **percentile-based thresholds derived from out-of-fold cross-validated probabilities** (`low_max=0.8351`, `medium_max=0.9556`) — computed without ever touching the test set.

**2. A near-perfect CKD score was investigated, not trusted.**
An earlier CKD dataset produced ~100% accuracy. Instead of accepting that, the "healthy" class was inspected directly — every non-CKD patient had lab values sitting in a perfectly normal range with zero borderline cases, which is not realistic. The dataset was swapped for a larger one; even so, CKD's strongest feature correlation is weak (|r| ≈ 0.20), and this is documented rather than hidden.

**3. Missing data is expected, not an error state.**
`predictor.py` only requires a low floor (`MINIMUM_FIELDS_REQUIRED = 1`, i.e. *some* relevant signal) rather than a fixed list of mandatory fields. Anything beyond that is passed through as `NaN` and filled in by the same `SimpleImputer` fitted during training. Every response reports `data_completeness` and `estimated_fields`, so the frontend (and the patient) can see exactly how much of a given prediction was estimated.

**4. Leakage is checked for, not assumed absent.**
A Heart Disease preprocessing step (dropping ECG/stress-test columns that aren't obtainable from a blood report) caused 5 rows to become exact duplicates *after* the drop — a leak that wasn't visible before that step. Deduplication was moved to run after the column drop, and the model was retrained.

**5. Explainability is per-prediction, not global.**
SHAP explanations are computed against a small k-means background sample (25 points) per disease, using a model-agnostic `Explainer` around each calibrated model's `predict_proba`. Every prediction returns its own top-3 contributing factors (feature, direction, impact) — not a generic "feature importance" chart.

**6. External LLM calls are treated as unreliable by default.**
Both the Gemini explanation call and the Gemini extraction call retry on transient `503` errors (2s, then 4s backoff) before falling back to a safe message. Free-tier LLM APIs intermittently throttle concurrent requests; the system degrades gracefully instead of surfacing a raw failure.

**7. The LLM explains; it does not decide.**
Gemini is given the ML model's probability and the RAG-retrieved evidence, and is instructed to explain and recommend — never to state its own risk number. This keeps the one component prone to hallucination (the LLM) out of the one place where a wrong number is dangerous.

---

## API Reference

### `GET /health`
Liveness check.

### `GET /diseases`
Returns the input schema (numeric features, categorical options, valid ranges) for all three diseases — the frontend uses this to build its form dynamically.

### `POST /predict/<disease_key>`
`disease_key` is one of `diabetes`, `heart`, `ckd`.

```json
// Request
{
  "age": 55, "bmi": 31.2, "HbA1c_level": 6.8, "blood_glucose_level": 180,
  "gender": "Male", "smoking_history": "current",
  "hypertension": "1", "heart_disease": "0"
}
```

```json
// Response
{
  "disease": "Diabetes",
  "probability": 0.8712,
  "risk_category": "High",
  "disclaimer": null,
  "warnings": [],
  "estimated_fields": [],
  "data_completeness": 100.0,
  "explanation": {
    "top_factors": [
      { "feature": "HbA1c_level", "direction": "increased", "impact": 0.6197 },
      { "feature": "blood_glucose_level", "direction": "increased", "impact": 0.0476 },
      { "feature": "heart_disease: 0", "direction": "increased", "impact": 0.0373 }
    ]
  }
}
```

### `POST /predict/full`
Same idea, run across all three diseases at once. Body is `{ "diabetes": {...}, "heart": {...}, "ckd": {...} }`; any disease section may be partial or omitted. Returns per-disease results, a `skipped` object explaining any disease that couldn't run, and a `summary.highest_risk_disease`.

### `POST /report/full`
Everything `/predict/full` returns, plus a Gemini-generated `ai_explanation` and `evidence_sources` per disease, grounded in the RAG knowledge base (CDC Diabetes, ADA, NIDDK CKD, CDC Heart).

### `POST /extract-report`
Accepts a `multipart/form-data` upload (`report` field, PDF or image). Uses Gemini's multimodal input to extract lab values matching the frontend's field catalog, returning only fields it actually found — never a guessed value.

---

## Project Structure

```
app/
  __init__.py         # app factory, loads HealthPredictor once at startup
  routes.py            # all endpoints
  predictor.py          # HealthPredictor: validation, inference, SHAP explanations
  rag_generator.py       # RAG retrieval + Gemini report generation, with retry logic
  extractor.py            # Gemini-based PDF/image lab value extraction, with retry logic
saved_models/
  metadata.json                       # input schema, thresholds, categorical options, ranges
  {disease}_model.joblib               # calibrated classifier per disease
  {disease}_preprocessor.joblib         # fitted ColumnTransformer per disease
  {disease}_shap_background.joblib       # k-means background sample per disease
run.py
requirements.txt
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run locally:
```bash
python run.py
```

The server loads all three models, preprocessors, and SHAP explainers once at startup — check the terminal for:
```
[HealthPredictor] Loaded 3 disease models: ['diabetes', 'heart', 'ckd']
[HealthPredictor] Built SHAP explainers for: ['diabetes', 'heart', 'ckd']
```

---

## Known Limitations

- **CKD's `Gender` encoding is an assumption, not a verified fact.** The training dataset's Gender column had no published data dictionary; the frontend assumes `Female=0, Male=1` (scikit-learn's default `LabelEncoder` convention). This should be re-verified against the training notebook before the CKD model's gender-based behavior is trusted.
- **CKD's predictive signal is comparatively weak.** Even after switching to a larger dataset, correlations with the target are low, and probability distributions for the two classes overlap substantially. CKD's "Low" risk category in particular should not be read as strong reassurance — a disclaimer is surfaced to the user for that case specifically.
<<<<<<< HEAD
- **This is not a medical device.** Outputs are probabilistic estimates from models trained on public/synthetic datasets, not a diagnosis. A disclaimer to this effect is shown with every report.
=======
- **This is not a medical device.** Outputs are probabilistic estimates from models trained on public/synthetic datasets, not a diagnosis. A disclaimer to this effect is shown with every report.
>>>>>>> 271d6a49cdaff3b3a3c53ceabad88e56fe7829e1
