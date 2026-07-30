import urllib.request
import json

try:
    print("Testing /api/connect...")
    req = urllib.request.Request("http://127.0.0.1:5000/api/connect", method="POST")
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        print(f"Response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
