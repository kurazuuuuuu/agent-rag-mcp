import os
import time
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is not set!")
    exit(1)

print(f"🔑 Key found: {api_key[:5]}...")

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

print("📡 Testing Gemini connectivity...")
start = time.time()
try:
    # Use a simple generation request
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp", 
        contents="Hello, ping."
    )
    print(f"✅ Success! Response: {response.text}")
except Exception as e:
    print(f"❌ Failed: {e}")
    # Print more details if available
    try:
        import socket
        print(f"Network check - googleapis.com IP: {socket.gethostbyname('generativelanguage.googleapis.com')}")
    except Exception as dns_e:
        print(f"DNS Check failed: {dns_e}")

end = time.time()
print(f"⏱️ Duration: {end - start:.2f}s")
