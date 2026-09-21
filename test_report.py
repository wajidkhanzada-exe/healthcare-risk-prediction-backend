"""
test_report.py - Sends a sample patient's data to /report/full and
prints the results in a readable way. Run this instead of using
curl/PowerShell to avoid command-line JSON quoting issues.
"""

import requests
import json

API_BASE = "http://127.0.0.1:5000"

payload = {
    "diabetes": {
        "age": 45.0,
        "bmi": 27.3,
        "HbA1c_level": 6.1,
        "blood_glucose_level": 140,
        "gender": "Female",
        "smoking_history": "never",
        "hypertension": 0,
        "heart_disease": 0
    },
    "heart": {
        "Age": 46,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 115.0,
        "Cholesterol": 237.0,
        "FastingBS": 0,
        "ExerciseAngina": "Y"
    },
    "ckd": {
        "Age": 66, "Gender": 1, "BMI": 32.44, "SystolicBP": 91, "DiastolicBP": 95,
        "FastingBloodSugar": 135.43, "HbA1c": 6.64, "SerumCreatinine": 3.64,
        "BUNLevels": 41.96, "GFR": 59.27, "ProteinInUrine": 4.53, "ACR": 298.64,
        "SerumElectrolytesSodium": 141.38, "SerumElectrolytesPotassium": 3.79,
        "SerumElectrolytesCalcium": 10.16, "SerumElectrolytesPhosphorus": 2.86,
        "HemoglobinLevels": 13.22, "CholesterolTotal": 183.05, "CholesterolLDL": 158.57,
        "CholesterolHDL": 70.86, "CholesterolTriglycerides": 109.28,
        "FamilyHistoryKidneyDisease": 0, "FamilyHistoryHypertension": 0,
        "FamilyHistoryDiabetes": 0, "PreviousAcuteKidneyInjury": 0,
        "UrinaryTractInfections": 0, "Smoking": 0, "Edema": 0,
        "FatigueLevels": 0.34, "NauseaVomiting": 2.47, "MuscleCramps": 2.92, "Itching": 9.94
    }
}

print("Sending request to /report/full ...\n")
response = requests.post(f"{API_BASE}/report/full", json=payload)

print(f"Status code: {response.status_code}\n")

if response.status_code != 200:
    print("Error response:")
    print(response.text)
else:
    data = response.json()

    for disease_key, result in data["results"].items():
        print("=" * 60)
        print(f"{result['disease']}")
        print("=" * 60)
        print(f"Probability: {result['probability']}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"\nAI Explanation:\n{result['ai_explanation']}")
        print(f"\nEvidence Sources: {result['evidence_sources']}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(data["summary"], indent=2))