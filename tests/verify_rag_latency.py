import asyncio
import time
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
import os

SERVER_URL = "http://127.0.0.1:8000/mcp"
TOKEN = os.getenv("AUTH_TOKEN", "secret")

async def test_rag_latency():
    transport = StreamableHttpTransport(SERVER_URL)
    client = Client(transport, auth=TOKEN)
    
    print(f"🔌 Connecting to {SERVER_URL}...")
    async with client:
        print("✅ Connected. Testing ask_project_document latency...")
        
        start = time.time()
        try:
            # Simple question that should rely on static docs (which we have or not)
            # If no docs, it returns error/empty, which is fine, checking roundtrip time.
            result = await client.call_tool("ask_project_document", {"question": "What is the project summary?"})
            duration = time.time() - start
            print(f"⏱️ Duration: {duration:.2f}s")
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else str(result)
                print(f"✅ Answer length: {len(content)}")
                print(f"📄 Answer preview: {content[:300]}...")
            else:
                print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_rag_latency())
