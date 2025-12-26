"""Test simple tool that should return quickly."""
import asyncio
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
import time

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def test_simple_tool():
    transport = StreamableHttpTransport(SERVER_URL)
    client = Client(transport)
    
    print(f"🔌 Connecting to {SERVER_URL}...")
    async with client:
        print("✅ Connected. Testing simple tools...")
        
        # Test 1: get_store_info (should be fast)
        print("\n📋 Testing get_store_info...")
        start = time.time()
        try:
            result = await asyncio.wait_for(
                client.call_tool("get_store_info", {}),
                timeout=10.0
            )
            duration = time.time() - start
            print(f"✅ get_store_info: {duration:.2f}s")
            if hasattr(result, 'content') and result.content:
                print(f"   Result: {result.content[0].text[:200]}...")
        except asyncio.TimeoutError:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Test 2: get_request_schema_template (should be fast)
        print("\n📋 Testing get_request_schema_template...")
        start = time.time()
        try:
            result = await asyncio.wait_for(
                client.call_tool("get_request_schema_template", {}),
                timeout=10.0
            )
            duration = time.time() - start
            print(f"✅ get_request_schema_template: {duration:.2f}s")
            if hasattr(result, 'content') and result.content:
                print(f"   Result: {result.content[0].text[:200]}...")
        except asyncio.TimeoutError:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_simple_tool())
