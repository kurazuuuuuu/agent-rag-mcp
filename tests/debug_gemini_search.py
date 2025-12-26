import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def test_file_search():
    client = genai.Client(api_key=api_key)
    
    print("🔍 Listing stores...")
    target_store = None
    for store in client.file_search_stores.list():
        print(f" - {store.display_name} ({store.name})")
        if "krz-tech-minecraft-project" in store.display_name:
            target_store = store.name
            
    if not target_store:
        print("❌ Store not found")
        return

    print(f"🎯 Using store: {target_store}")
    
    prompt = "What is the project summary?"
    model = "models/gemini-flash-latest" # Using the one we decided on
    
    print(f"🚀 Sending query to {model}...")
    start = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[target_store]
                        )
                    )
                ]
            ),
        )
        duration = time.time() - start
        print(f"✅ Response received in {duration:.2f}s")
        print(f"Answer: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_file_search()
