"""
predictor.py - Inference module for the Healthcare Risk Prediction System.

Responsible for:
  1. Loading all 3 diseases' preprocessors + calibrated models ONCE (at startup)
  2. Validating raw patient input against metadata.json
  3. Running the exact same preprocessing -> prediction pipeline used in training
  4. Converting probability -> risk category using disease-specific thresholds
  5. Explaining a prediction via SHAP (top contributing factors)

Design note on missing data:
  Each disease has a small set of CRITICAL_FIELDS that must be present for
  a prediction to be meaningful (e.g. HbA1c for diabetes, electrolytes for
  CKD). Any other feature in the disease's schema is treated as optional:
  if it is missing, it is passed through as NaN and filled in by the
  trained preprocessing pipeline's SimpleImputer, using statistics learned
  from the training data (median for numeric features, most frequent
  category for categorical features). This mirrors how real clinical risk
  calculators handle incomplete patient records rather than rejecting the
  whole submission outright.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models"
)


# Fields that must be present for a disease's prediction to be considered
# reliable. Any field in the disease's schema that is NOT listed here may
# be omitted - it is treated as missing and imputed by the trained
# preprocessing pipeline instead of blocking the prediction.
CRITICAL_FIELDS = {
    "diabetes": [
        "age", "bmi", "HbA1c_level", "blood_glucose_level",
        "gender", "smoking_history", "hypertension", "heart_disease"
    ],
    "heart": [
        "Age", "Sex", "ChestPainType", "RestingBP",
        "Cholesterol", "FastingBS", "ExerciseAngina"
    ],
    "ckd": [
        "Age", "Gender", "Smoking", "BMI", "SystolicBP", "HbA1c",
        "SerumElectrolytesSodium", "SerumElectrolytesPotassium",
        "FastingBloodSugar", "CholesterolTotal"
    ]
}


class ValidationError(Exception):
    """Raised when patient input fails validation. Safe to show message to user."""
    pass


class HealthPredictor:
    def __init__(self, artifact_dir: str = ARTIFACT_DIR):
        self.artifact_dir = artifact_dir

        # Load metadata once
        with open(os.path.join(artifact_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        # Load all preprocessors and models once, keep them in memory
        self.preprocessors = {}
        self.models = {}

        for disease_key, info in self.metadata["diseases"].items():
            self.preprocessors[disease_key] = joblib.load(
                os.path.join(artifact_dir, info["preprocessor_file"])
            )
            self.models[disease_key] = joblib.load(
                os.path.join(artifact_dir, info["model_file"])
            )

        print(f"[HealthPredictor] Loaded {len(self.models)} disease models: "
              f"{list(self.models.keys())}")

        
        self.explainers = {}
        for disease_key in self.models:
            bg_path = os.path.join(artifact_dir, f"{disease_key}_shap_background.joblib")
            if not os.path.exists(bg_path):
                print(f"[HealthPredictor] No SHAP background found for {disease_key}, "
                      f"skipping explainability for this disease.")
                continue

            background = joblib.load(bg_path)
            model = self.models[disease_key]

            def make_predict_fn(m):
                return lambda X: m.predict_proba(X)[:, 1]

            self.explainers[disease_key] = shap.Explainer(
                make_predict_fn(model), background
            )

        print(f"[HealthPredictor] Built SHAP explainers for: {list(self.explainers.keys())}")

    def _validate_critical_fields(self, disease_key: str, patient_data: dict) -> None:
        """
        Raises ValidationError if any field listed in CRITICAL_FIELDS for
        this disease is missing or empty. These are the fields without which
        a prediction would not be clinically meaningful. Any other field in
        the disease's schema may be omitted - it is imputed downstream.
        """
        critical_fields = CRITICAL_FIELDS.get(disease_key, [])
        missing = [
            field for field in critical_fields
            if field not in patient_data or patient_data[field] in (None, "")
        ]
        if missing:
            raise ValidationError(
                f"Missing required fields for a reliable {disease_key} prediction: {missing}"
            )

    def _validate_provided_values(self, disease_key: str, patient_data: dict) -> None:
        """
        Validates only the fields that WERE provided - categorical values
        must be one of the allowed options, numeric values must be numbers.
        Fields not present in patient_data are skipped here; they are
        handled as missing values further down the pipeline.
        """
        info = self.metadata["diseases"][disease_key]

        for col in info["categorical_features"]:
            if col not in patient_data:
                continue
            allowed = info["categorical_options"].get(col, [])
            value = str(patient_data[col])
            if allowed and value not in allowed:
                raise ValidationError(
                    f"Invalid value '{value}' for '{col}'. Allowed values: {allowed}"
                )

        for col in info["numeric_features"]:
            if col not in patient_data:
                continue
            value = patient_data[col]
            if not isinstance(value, (int, float)):
                raise ValidationError(f"'{col}' must be a number, got: {type(value).__name__}")

    def _check_numeric_ranges(self, disease_key: str, patient_data: dict) -> list:
        """
        Returns a list of warning strings for numeric values outside the
        training data's observed range. Only checks fields that were
        actually provided. Does NOT block the prediction.
        """
        info = self.metadata["diseases"][disease_key]
        warnings = []

        for col in info["numeric_features"]:
            if col not in patient_data:
                continue
            value = patient_data[col]
            rng = info["numeric_ranges"].get(col)
            if rng and (value < rng["min"] or value > rng["max"]):
                warnings.append(
                    f"'{col}' value {value} is outside the training data range "
                    f"({rng['min']} - {rng['max']}); prediction may be less reliable."
                )
        return warnings

    def _get_risk_category(self, disease_key: str, probability: float) -> str:
        thresholds = self.metadata["risk_thresholds_by_disease"][disease_key]
        if probability < thresholds["low_max"]:
            return "Low"
        elif probability < thresholds["medium_max"]:
            return "Medium"
        else:
            return "High"

    def _prettify_feature_name(self, raw_name: str) -> str:
        """
        Converts a preprocessed column name like 'numeric__HbA1c_level' or
        'categorical__gender_Male' into a readable label like 'HbA1c Level'
        or 'Gender: Male'.
        """
        name = raw_name.split("__", 1)[-1]  # drop the 'numeric__'/'categorical__' prefix
        if "_" in name and raw_name.startswith("categorical__"):
            base, _, option = name.rpartition("_")
            return f"{base}: {option}"
        return name

    # ------------------------------------------------------------------
    def predict(self, disease_key: str, patient_data: dict) -> dict:
        """
        Main entry point.

        Args:
            disease_key: one of "diabetes", "heart", "ckd"
            patient_data: dict of raw feature values. May be a PARTIAL
                dict - any field not listed in CRITICAL_FIELDS for this
                disease may be omitted.

        Returns:
            dict with probability, risk_category, disclaimer, warnings,
            estimated_fields (fields that were missing and imputed), and
            data_completeness (percentage of the schema that was provided).
        """
        if disease_key not in self.metadata["diseases"]:
            raise ValidationError(
                f"Unknown disease_key '{disease_key}'. "
                f"Must be one of: {list(self.metadata['diseases'].keys())}"
            )

        info = self.metadata["diseases"][disease_key]

        # Step 1: Validate
        self._validate_critical_fields(disease_key, patient_data)
        self._validate_provided_values(disease_key, patient_data)
        range_warnings = self._check_numeric_ranges(disease_key, patient_data)

        # Step 2: Track which fields were missing and will be imputed
        estimated_fields = [
            col for col in info["input_columns"]
            if col not in patient_data or patient_data[col] in (None, "")
        ]

        # Step 3: Build a single-row DataFrame, missing columns become NaN
        raw_df = pd.DataFrame([patient_data])
        raw_df = raw_df.reindex(columns=info["input_columns"])

        # Step 4: Preprocess (same transformer fitted during training;
        # its SimpleImputer fills any NaN using training-data statistics)
        preprocessor = self.preprocessors[disease_key]
        processed = preprocessor.transform(raw_df)
        processed_df = pd.DataFrame(
            processed,
            columns=preprocessor.get_feature_names_out(),
            index=raw_df.index
        )

        # Step 5: Predict probability
        model = self.models[disease_key]
        probability = float(model.predict_proba(processed_df)[:, 1][0])

        # Step 6: Categorize
        risk_category = self._get_risk_category(disease_key, probability)

        # Step 7: Disclaimer (only CKD has one, only shown for "Low")
        disclaimer = None
        if risk_category == "Low" and info.get("low_category_disclaimer"):
            disclaimer = info["low_category_disclaimer"]

        total_fields = len(info["input_columns"])
        completeness = round(
            100 * (total_fields - len(estimated_fields)) / total_fields, 1
        )

        return {
            "disease": info["display_name"],
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "disclaimer": disclaimer,
            "warnings": range_warnings,
            "estimated_fields": estimated_fields,
            "data_completeness": completeness
        }

    # ------------------------------------------------------------------
    def explain(self, disease_key: str, patient_data: dict, top_n: int = 3) -> dict:
        """
        Returns the top contributing factors behind this disease's most
        recent prediction for this patient. Must be called with the SAME
        patient_data that was passed to predict() for consistent results.

        Returns an empty list (not an error) if no explainer is available
        for this disease, so callers can treat explainability as optional.
        """
        if disease_key not in self.explainers:
            return {"top_factors": []}

        info = self.metadata["diseases"][disease_key]
        raw_df = pd.DataFrame([patient_data]).reindex(columns=info["input_columns"])
        preprocessor = self.preprocessors[disease_key]
        processed = preprocessor.transform(raw_df)
        processed_df = pd.DataFrame(
            processed, columns=preprocessor.get_feature_names_out(), index=raw_df.index
        )

        shap_values = self.explainers[disease_key](processed_df)
        contributions = shap_values.values[0]
        feature_names = shap_values.feature_names

        ranked = sorted(
            zip(feature_names, contributions),
            key=lambda pair: abs(pair[1]),
            reverse=True
        )[:top_n]

        top_factors = [
            {
                "feature": self._prettify_feature_name(name),
                "direction": "increased" if value > 0 else "decreased",
                "impact": round(abs(float(value)), 4)
            }
            for name, value in ranked
        ]

        return {"top_factors": top_factors}