from flask import Blueprint, request, jsonify, current_app
import os
import requests

from .predictor import ValidationError
from .rag_generator import generate_explanation_and_plan
from .extractor import extract_fields_from_document


bp = Blueprint("api", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Simple check to confirm the server is running."""
    return jsonify({"status": "ok"}), 200


@bp.route("/diseases", methods=["GET"])
def list_diseases():
    """
    Returns the schema for all 3 diseases - field names, allowed
    categorical values, numeric ranges. The frontend uses this to
    build the patient input form dynamically.
    """
    predictor = current_app.predictor
    diseases_info = {}

    for key, info in predictor.metadata["diseases"].items():
        diseases_info[key] = {
            "display_name": info["display_name"],
            "numeric_features": info["numeric_features"],
            "categorical_features": info["categorical_features"],
            "categorical_options": info["categorical_options"],
            "numeric_ranges": info["numeric_ranges"]
        }

    return jsonify(diseases_info), 200


@bp.route("/predict/<disease_key>", methods=["POST"])
def predict(disease_key):
    """
    Main prediction endpoint.
    Expects JSON body with the patient's raw field values.

    Example:
        POST /predict/diabetes
        {
            "age": 45,
            "bmi": 27.3,
            "gender": "Female"
        }
    """
    predictor = current_app.predictor

    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    patient_data = request.get_json()

    try:
        result = predictor.predict(disease_key, patient_data)

        # Same patient_data used here as in predict() above,
        # so the SHAP explanation lines up with the prediction.
        result["explanation"] = predictor.explain(
            disease_key,
            patient_data
        )

        return jsonify(result), 200

    except ValidationError as e:
        # Bad input from the client - safe to show the message.
        return jsonify({
            "error": str(e)
        }), 400

    except Exception:
        # Unexpected server-side problem - don't leak internals.
        return jsonify({
            "error": "Internal server error"
        }), 500


@bp.route("/predict/full", methods=["POST"])
def predict_full():
    """
    Combined report endpoint - runs a prediction for each disease that
    has enough data (its CRITICAL_FIELDS present), and skips the rest.

    Expects JSON body shaped like:

    {
        "diabetes": { ...diabetes fields... },
        "heart": { ...heart fields... },
        "ckd": { ...ckd fields... }
    }

    A disease's section may be partial or entirely absent - it will
    simply be skipped if its critical fields aren't present.

    Returns the successful results, a "skipped" section explaining why
    any disease was excluded, and an overall summary based only on the
    diseases that were successfully predicted.
    """
    predictor = current_app.predictor

    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    payload = request.get_json()
    disease_keys = ["diabetes", "heart", "ckd"]

    results = {}
    skipped = {}

    for disease_key in disease_keys:
        disease_data = payload.get(disease_key, {})

        try:
            results[disease_key] = predictor.predict(
                disease_key,
                disease_data
            )

            # Same disease_data used here as just above,
            # so the SHAP explanation lines up with the prediction.
            results[disease_key]["explanation"] = predictor.explain(
                disease_key,
                disease_data
            )

        except ValidationError as e:
            skipped[disease_key] = str(e)

        except Exception:
            skipped[disease_key] = "Internal server error"

    if not results:
        return jsonify({
            "error": "Not enough data was provided to run any prediction.",
            "skipped": skipped
        }), 400

    risk_order = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    highest_risk_disease = max(
        results.items(),
        key=lambda item: (
            risk_order[item[1]["risk_category"]],
            item[1]["probability"]
        )
    )

    report = {
        "results": results,
        "skipped": skipped,
        "summary": {
            "highest_risk_disease": highest_risk_disease[1]["disease"],
            "highest_risk_category": highest_risk_disease[1]["risk_category"],
            "diseases_flagged_high": [
                r["disease"]
                for r in results.values()
                if r["risk_category"] == "High"
            ]
        }
    }

    return jsonify(report), 200


@bp.route("/report/full", methods=["POST"])
def report_full():
    """
    Same as /predict/full, but each disease's result also gets a
    Gemini-generated explanation + lifestyle/exercise plan grounded
    in the RAG knowledge base.

    After the report is generated, it is saved to Supabase for the
    currently authenticated user.
    """

    # ---------------------------------------------------------
    # Get the original frontend payload
    # ---------------------------------------------------------

    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400

    # ---------------------------------------------------------
    # Reuse existing prediction logic
    # ---------------------------------------------------------

    prediction_response = predict_full()

    # predict_full() returns a (Response, status_code) tuple.
    response_obj, status_code = prediction_response

    # If prediction failed, pass that error straight through.
    if status_code != 200:
        return prediction_response

    report_data = response_obj.get_json()

    # ---------------------------------------------------------
    # Generate Gemini/RAG explanations
    # ---------------------------------------------------------

    for disease_key, result in report_data["results"].items():

        generated = generate_explanation_and_plan(
            disease_key=disease_key,
            disease_display_name=result["disease"],
            probability=result["probability"],
            risk_category=result["risk_category"]
        )

        result["ai_explanation"] = generated["text"]
        result["evidence_sources"] = generated["sources"]

    # ---------------------------------------------------------
    # Get Supabase authentication token
    # ---------------------------------------------------------

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return jsonify({
            "error": "Authentication required"
        }), 401

    access_token = auth_header.split(" ", 1)[1].strip()

    if not access_token:
        return jsonify({
            "error": "Invalid authentication token"
        }), 401

    # ---------------------------------------------------------
    # Get Supabase configuration
    # ---------------------------------------------------------

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_anon_key:
        return jsonify({
            "error": "Supabase configuration is missing on the server"
        }), 500

    # ---------------------------------------------------------
    # Verify the logged-in user with Supabase
    # ---------------------------------------------------------

    try:
        user_response = requests.get(
            f"{supabase_url}/auth/v1/user",
            headers={
                "apikey": supabase_anon_key,
                "Authorization": f"Bearer {access_token}"
            },
            timeout=10
        )
    except requests.RequestException:
        return jsonify({
            "error": "Could not connect to Supabase authentication service"
        }), 502

    if user_response.status_code != 200:
        return jsonify({
            "error": "Invalid or expired authentication session"
        }), 401

    try:
        user_data = user_response.json()
    except ValueError:
        return jsonify({
            "error": "Invalid response from Supabase authentication service"
        }), 502

    user_id = user_data.get("id")

    if not user_id:
        return jsonify({
            "error": "Could not identify authenticated user"
        }), 401

    # ---------------------------------------------------------
    # Get patient information from frontend
    # ---------------------------------------------------------

    patient_name = payload.get("fullName")
    patient_age = payload.get("age")

    # Convert age to integer for the database.
    if patient_age not in (None, ""):
        try:
            patient_age = int(patient_age)
        except (TypeError, ValueError):
            patient_age = None

    # ---------------------------------------------------------
    # Prepare database record
    # ---------------------------------------------------------

    database_row = {
        "user_id": user_id,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "report_data": report_data
    }

    # ---------------------------------------------------------
    # Save completed report to Supabase
    # ---------------------------------------------------------

    try:
        insert_response = requests.post(
            f"{supabase_url}/rest/v1/reports",
            headers={
                "apikey": supabase_anon_key,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json=database_row,
            timeout=15
        )
    except requests.RequestException as e:
        print("Supabase report insert connection error:")
        print(str(e))

        return jsonify({
            "error": "Report was generated but could not be saved"
        }), 502

    if insert_response.status_code not in (200, 201):
        print("Supabase report insert failed:")
        print("Status:", insert_response.status_code)
        print("Response:", insert_response.text)

        return jsonify({
            "error": "Report was generated but could not be saved"
        }), 500

    # ---------------------------------------------------------
    # Return the same report response to the frontend
    # ---------------------------------------------------------

    return jsonify(report_data), 200


@bp.route("/extract-report", methods=["POST"])
def extract_report():
    """
    Accepts an uploaded lab report (PDF or image) and returns
    extracted field values for pre-filling the patient form.
    """
    if "report" not in request.files:
        return jsonify({
            "error": "No file uploaded. Expected a 'report' field."
        }), 400

    file = request.files["report"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    allowed_types = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.mimetype not in allowed_types:
        return jsonify({
            "error": f"Unsupported file type: {file.mimetype}"
        }), 400

    try:
        file_bytes = file.read()

        extracted = extract_fields_from_document(
            file_bytes,
            file.mimetype
        )

        return jsonify({
            "extracted_fields": extracted,
            "field_count": len(extracted)
        }), 200

    except Exception as e:
        import traceback

        traceback.print_exc()

        return jsonify({
            "error": "Extraction failed",
            "details": str(e)
        }), 500