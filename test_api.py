
import os
import sys
import base64
import urllib.request
import urllib.error
import json
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in .env file.")
    sys.exit(1)

def test_document(file_path: str):
    """Reads a local file, converts it to base64, and sends it to the API."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    file_name = os.path.basename(file_path)
    ext = file_name.split('.')[-1].lower()

    # Map file extension to accepted API fileType
    if ext == 'pdf':
        file_type = 'pdf'
    elif ext in ['docx', 'doc']:
        file_type = 'docx'
    elif ext in ['png', 'jpg', 'jpeg']:
        file_type = 'image'
    else:
        print(f"Error: Unsupported file extension '{ext}'. Must be pdf, docx, or image.")
        sys.exit(1)

    print(f"Reading {file_name}...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    base64_encoded = base64.b64encode(file_bytes).decode("utf-8")

    payload = {
        "fileName": file_name,
        "fileType": file_type,
        "fileBase64": base64_encoded
    }
    
    data = json.dumps(payload).encode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    url = "http://localhost:8000/api/document-analyze"
    print(f"Sending API Request to {url}...")
    
    req = urllib.request.Request(url, data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"\nResponse Status: {response.status}")
            result = json.loads(response.read().decode('utf-8'))
            print("\nAnalysis Result:\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    except urllib.error.HTTPError as e:
        print(f"\nResponse Status: {e.code}")
        print(f"Error Message: {e.read().decode('utf-8')}")
    except urllib.error.URLError:
        print("Error: Could not connect to API. Is uvicorn running?")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_your_document>")
        print("Example: python test_api.py sample.pdf")
        sys.exit(1)
        
    # Join arguments so paths with spaces (e.g. sample doc.pdf) work smoothly
    file_path = " ".join(sys.argv[1:])
    test_document(file_path)
