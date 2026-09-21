import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # .env file se GEMINI_API_KEY load karta hai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Available models for this API key:\n")
for model in client.models.list():
    print(model.name)