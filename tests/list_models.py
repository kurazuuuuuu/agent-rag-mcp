import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("--- v1beta (default) Models ---")
client = genai.Client(api_key=api_key)
try:
    for m in client.models.list():
        print(f"{m.name}")
except Exception as e:
    print(f"Error: {e}")
    
print("\n--- v1alpha Models ---")
client_alpha = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
try:
    for m in client.models.list():
        print(f"{m.name}")
except Exception as e:
    print(f"Error: {e}")
