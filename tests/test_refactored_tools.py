"""Test refactored code pattern tools."""
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

SERVER_URL = "http://127.0.0.1:8000/sse"

async def test_tools():
    print(f"🔌 Connecting to {SERVER_URL}...")
    transport = SSETransport(SERVER_URL)
    client = Client(transport)
    
    async with client:
        print("\n📝 1. Testing tell_code_pattern (Success Report)...")
        success_report = """
request:
  language: Python
  framework: FastAPI
  design_context:
    pattern: Singleton
  content:
    result: SUCCESS
    feature_details: Created a singleton database connection pool.
    code:
      success: |
        class DB:
          _instance = None
          def __new__(cls):
            if not cls._instance: cls._instance = super().__new__(cls)
            return cls._instance
"""
        result = await client.call_tool("tell_code_pattern", {"request_data": success_report})
        print(f"Result: {result.content[0].text if result.content else result}")

        print("\n🔍 2. Testing ask_code_pattern (Search for Singleton)...")
        ask_query = """
request:
  language: Python
  framework: FastAPI
  design_context:
    pattern: Singleton
  content:
    feature_details: How to implement a database singleton?
"""
        result = await client.call_tool("ask_code_pattern", {"request_data": ask_query})
        print(f"Result Preview: {result.content[0].text[:300] if result.content else result}...")

        print("\n❌ 3. Testing tell_code_pattern (Failure Report and Advice)...")
        failure_report = """
request:
  language: Python
  framework: FastAPI
  design_context:
    pattern: Singleton
  content:
    result: FAILED
    feature_details: Errors occur when using __init__ in Singleton.
"""
        result = await client.call_tool("tell_code_pattern", {"request_data": failure_report})
        print(f"Result Preview: {result.content[0].text[:500] if result.content else result}...")

if __name__ == "__main__":
    asyncio.run(test_tools())
