# test_load.py - Verify all saved models load correctly on this machine
import joblib
import json
import os
import warnings

ARTIFACT_DIR = "saved_models"  # folder rename ke baad ye naam hona chahiye

print("Python environment check\n" + "="*50)

# Load metadata first
with open(os.path.join(ARTIFACT_DIR, "metadata.json")) as f:
    metadata = json.load(f)

print("Model trained with:")
for k, v in metadata["environment"].items():
    print(f"  {k}: {v}")

print("\n" + "="*50)
print("Attempting to load each preprocessor + model...\n")

# Catch warnings so version-mismatch warnings actually show up
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")

    for disease_key, info in metadata["diseases"].items():
        try:
            preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, info["preprocessor_file"]))
            model = joblib.load(os.path.join(ARTIFACT_DIR, info["model_file"]))
            print(f"[OK] {info['display_name']}: preprocessor + model loaded successfully")
        except Exception as e:
            print(f"[FAIL] {info['display_name']}: {type(e).__name__} - {e}")

    if caught_warnings:
        print("\n" + "="*50)
        print(f"{len(caught_warnings)} warning(s) raised during loading:\n")
        for w in caught_warnings:
            print(f"  - {w.category.__name__}: {w.message}")
    else:
        print("\nNo warnings raised during loading.")

print("\n" + "="*50)
print("Test complete.")