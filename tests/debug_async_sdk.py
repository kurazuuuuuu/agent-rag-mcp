"""Test native async SDK File Search."""
import asyncio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


async def test_async_file_search():
    client = genai.Client(api_key=api_key)
    
    print("🔍 Listing stores (native async)...")
    stores = await client.aio.file_search_stores.list()
    target_store = None
    for store in stores:
        print(f" - {store.display_name} ({store.name})")
        if "krz-tech-minecraft-project" in store.display_name:
            target_store = store.name
            
    if not target_store:
        print("❌ Store not found")
        return

    print(f"🎯 Using store: {target_store}")
    
    prompt = "What is the project summary?"
    model = "gemini-2.5-flash"
    
    print(f"🚀 Sending query via client.aio.models.generate_content({model})...")
    import time
    start = time.time()
    try:
        response = await client.aio.models.generate_content(
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
        if response.text:
            print(f"Answer: {response.text[:300]}...")
        else:
            print("⚠️ Empty response text")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_async_file_search())
