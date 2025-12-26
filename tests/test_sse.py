"""Test MCP with SSE transport."""
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

# Endpoint changed to /sse in legacy mode
SERVER_URL = "http://127.0.0.1:8000/sse"


async def test_sse():
    print(f"🔌 Connecting to {SERVER_URL} via SSE transport...")
    transport = SSETransport(SERVER_URL)
    client = Client(transport)
    
    async with client:
        print("✅ Connected. Testing ask_project_document (via REST bridge)...")
        
        import time
        start = time.time()
        try:
            # The tool currently calls the internal REST endpoint
            result = await asyncio.wait_for(
                client.call_tool("ask_project_document", {"question": "What is the project summary?"}),
                timeout=55.0
            )
            duration = time.time() - start
            print(f"✅ SUCCESS! Duration: {duration:.2f}s")
            
            if hasattr(result, 'content') and result.content:
                text = result.content[0].text
                print(f"📄 Response length: {len(text)}")
                print(f"📄 Preview: {text[:300]}...")
            else:
                print(f"✅ Result: {result}")
        except asyncio.TimeoutError:
            print(f"❌ Timeout after {time.time() - start:.2f}s")
        except Exception as e:
            print(f"❌ Failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_sse())
