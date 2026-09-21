import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # FLASK_DEBUG=1 turns on debug mode; defaults to OFF (production-safe) if not set
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=5000)