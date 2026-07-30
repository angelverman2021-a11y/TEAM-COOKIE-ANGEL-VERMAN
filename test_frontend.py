import urllib.request

try:
    print("Fetching http://127.0.0.1:5000/ ...")
    req = urllib.request.Request("http://127.0.0.1:5000/")
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        print(f"Status: {response.status}")
        
        if "id=\"btn-connect\"" in html:
            print("SUCCESS: btn-connect found in served HTML")
        else:
            print("ERROR: btn-connect NOT found in served HTML")
            
        if "id=\"btn-connect-card\"" in html:
            print("SUCCESS: btn-connect-card found in served HTML")
        else:
            print("ERROR: btn-connect-card NOT found in served HTML")
            
        if "script.js" in html:
            print("SUCCESS: script.js found in served HTML")
        else:
            print("ERROR: script.js NOT found in served HTML")
            
except Exception as e:
    print(f"Error: {e}")
