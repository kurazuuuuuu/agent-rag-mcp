"""Test MCP from inside container (localhost)."""
import asyncio
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport

SERVER_URL = "http://localhost:8000/mcp"


async def test_local():
    transport = StreamableHttpTransport(SERVER_URL)
    client = Client(transport)
    
    print(f"🔌 Connecting to {SERVER_URL}...")
    async with client:
        print("✅ Connected. Testing ask_project_document...")
        
        import time
        start = time.time()
        try:
            result = await asyncio.wait_for(
                client.call_tool("ask_project_document", {"question": "What is the project summary?"}),
                timeout=25.0
            )
            duration = time.time() - start
            print(f"✅ Duration: {duration:.2f}s")
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else str(result)
                print(f"📄 Answer length: {len(content)}")
                print(f"📄 Answer preview: {content[:200]}...")
            else:
                print(f"✅ Result: {result}")
        except asyncio.TimeoutError:
            print(f"❌ Timeout after {time.time() - start:.2f}s")
        except Exception as e:
            print(f"❌ Failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_local())
