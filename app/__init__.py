from flask import Flask
from flask_cors import CORS
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .predictor import HealthPredictor


# ============================================================
# LOAD PROJECT ROOT .ENV FILE
# ============================================================

# app.py is inside:
# healthcare-risk-prediction/app/app.py
#
# .env is inside:
# healthcare-risk-prediction/.env

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Load the project's .env file when python-dotenv is available
if load_dotenv:
    load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# MAP SUPABASE ENVIRONMENT VARIABLES FOR BACKEND
# ============================================================

# The .env currently uses VITE_ names because they are also
# used by the React frontend.
#
# Flask backend/routes may expect:
# SUPABASE_URL
# SUPABASE_ANON_KEY
#
# So we map the existing VITE_ variables to backend names.
# This does NOT change your existing .env values.

if not os.getenv("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = os.getenv("VITE_SUPABASE_URL", "")

if not os.getenv("SUPABASE_ANON_KEY"):
    os.environ["SUPABASE_ANON_KEY"] = os.getenv(
        "VITE_SUPABASE_ANON_KEY",
        ""
    )


# ============================================================
# CREATE FLASK APP
# ============================================================

def create_app():
    app = Flask(__name__)

    # ========================================================
    # ENVIRONMENT CONFIGURATION CHECK
    # ========================================================

    print("\n========================================")
    print("Backend Environment Configuration")
    print("========================================")

    print("Backend .env path:", ENV_FILE)
    print("Backend .env exists:", ENV_FILE.exists())

    print(
        "VITE_SUPABASE_URL loaded:",
        bool(os.getenv("VITE_SUPABASE_URL"))
    )

    print(
        "VITE_SUPABASE_ANON_KEY loaded:",
        bool(os.getenv("VITE_SUPABASE_ANON_KEY"))
    )

    print(
        "SUPABASE_URL loaded:",
        bool(os.getenv("SUPABASE_URL"))
    )

    print(
        "SUPABASE_ANON_KEY loaded:",
        bool(os.getenv("SUPABASE_ANON_KEY"))
    )

    print("========================================\n")

    # ========================================================
    # CORS CONFIGURATION
    # ========================================================

    CORS(
        app,
        origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://healthcare-risk-prediction-frontend.vercel.app",
            "https://healthriskkk-ai.vercel.app"
        ]
    )

    # ========================================================
    # LOAD MODELS ONCE WHEN APP STARTS
    # ========================================================

    app.predictor = HealthPredictor()

    # ========================================================
    # REGISTER ROUTES
    # ========================================================

    from .routes import bp
    app.register_blueprint(bp)

    return app