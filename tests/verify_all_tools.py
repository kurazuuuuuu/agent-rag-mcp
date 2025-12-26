import asyncio
import json
import os
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000/mcp" 
# Note: Docker Compose default AUTH_TOKEN might be empty or 'secret' depending on .env
# For test, we assume 'secret' or whatever provided. 
# If server started with empty AUTH_TOKEN, auth is disabled.
TOKEN = os.getenv("AUTH_TOKEN", "secret") 

async def run_tests():
    print(f"🔌 Connecting to {SERVER_URL} with token='{TOKEN}'...")
    
    transport = StreamableHttpTransport(SERVER_URL)
    # Using 'auth' kwarg directly on Client is the standard way now
    client = Client(transport, auth=TOKEN)

    async with client:
        print("✅ Connected!")
        
        # --- 1. Basic Info Tools ---
        print("\n--- [Test 1] get_store_info ---")
        try:
            info = await client.call_tool("get_store_info", {})
            print(f"Result:\n{info}")
        except Exception as e:
            print(f"❌ Failed: {e}")

        print("\n--- [Test 2] get_request_schema_template ---")
        try:
            schema_result = await client.call_tool("get_request_schema_template", {})
            schema = schema_result.content[0].text
            print(f"Result (First 100 chars): {schema[:100]}...")
            if "request:" in schema:
                print("✅ Validation: Contains 'request:' (TOON format detected)")
            else:
                print("⚠️ Validation: 'request:' not found in schema")
        except Exception as e:
            print(f"❌ Failed: {e}")

        # --- 3. Doc RAG ---
        # print("\n--- [Test 3] ask_project_document ---")
        # try:
        #     # We assume RAG is working or at least returns an answer (even if empty context)
        #     ans = await client.call_tool("ask_project_document", {"question": "What is this project?"})
        #     print(f"Result:\n{ans}")
        # except Exception as e:
        #     print(f"❌ Failed: {e}")

        # --- 3. Dynamic RAG (ask_code_pattern) ---
        print("\n--- [Test 4] ask_code_pattern (TOON Format) ---")
        toon_input = """
request:
  language: Python
  framework: FastMCP
  design_context:
    pattern: Tool Verification
  content:
    feature_details: Testing TOON support
        """.strip()
        try:
            ans = await client.call_tool("ask_code_pattern", {"request_data": toon_input})
            print(f"Result:\n{ans}")
        except Exception as e:
            print(f"❌ Failed: {e}")

        print("\n--- [Test 5] ask_code_pattern (JSON Format) ---")
        json_input = json.dumps({
            "request": {
                "language": "Python",
                "framework": "FastMCP",
                "design_context": {"pattern": "JSON Compatibility"},
                "content": {"feature_details": "Testing JSON fallback"}
            }
        })
        try:
            ans = await client.call_tool("ask_code_pattern", {"request_data": json_input})
            print(f"Result:\n{ans}")
        except Exception as e:
            print(f"❌ Failed: {e}")

        print("\n--- [Test 6] ask_code_pattern (Double-Encoded JSON) ---")
        # Simulating agent mistake: json.dumps(json_input)
        double_encoded = json.dumps(json_input) 
        try:
            ans = await client.call_tool("ask_code_pattern", {"request_data": double_encoded})
            print(f"Result:\n{ans}")
            if "Error" not in ans:
                 print("✅ Validation: Robustness fix worked (no error returned)")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_tests())
