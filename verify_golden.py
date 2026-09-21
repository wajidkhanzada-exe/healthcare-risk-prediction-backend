# verify_golden.py - Compare local predictions against Colab's known-correct outputs
import joblib
import json
import os
import pandas as pd
import numpy as np

ARTIFACT_DIR = "saved_models"

with open(os.path.join(ARTIFACT_DIR, "metadata.json")) as f:
    metadata = json.load(f)

with open(os.path.join(ARTIFACT_DIR, "golden_test.json")) as f:
    golden = json.load(f)

print("GOLDEN TEST VERIFICATION\n" + "="*55)
all_passed = True

for disease_key, data in golden.items():
    info = metadata["diseases"][disease_key]

    preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, info["preprocessor_file"]))
    model = joblib.load(os.path.join(ARTIFACT_DIR, info["model_file"]))

    raw_df = pd.DataFrame(data["raw_input_rows"])
    raw_df = raw_df[info["input_columns"]]  # enforce exact column order

    processed = preprocessor.transform(raw_df)
    local_probs = model.predict_proba(processed)[:, 1]

    expected_probs = np.array(data["expected_probabilities"])
    max_diff = np.abs(local_probs - expected_probs).max()
    passed = max_diff < 1e-6

    all_passed = all_passed and passed

    print(f"\n{info['display_name']}")
    print(f"  Expected (Colab) : {[round(p, 6) for p in expected_probs]}")
    print(f"  Got (Local)      : {[round(p, 6) for p in local_probs]}")
    print(f"  Max difference   : {max_diff:.2e}")
    print(f"  Status           : {'PASS' if passed else 'FAIL'}")

print("\n" + "="*55)
print("ALL PASSED - safe to proceed" if all_passed else "MISMATCH FOUND - do not proceed, tell Claude")