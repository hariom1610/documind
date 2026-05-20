import os
import sys
import base64
import urllib.request
import urllib.error
import json
import time
import glob
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in .env file.")
    sys.exit(1)

SUPPORTED_EXTENSIONS = {
    'pdf': 'pdf',
    'docx': 'docx',
    'doc': 'docx',
    'png': 'image',
    'jpg': 'image',
    'jpeg': 'image'
}

def test_document(file_path: str) -> bool:
    """Reads a local file, converts it to base64, sends it to the API, and prints result metrics."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File '{file_path}' does not exist.")
        return False

    file_name = os.path.basename(file_path)
    ext = file_name.split('.')[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Unsupported file extension '{ext}' for file '{file_name}'. Must be pdf, docx, or image.")
        return False

    file_type = SUPPORTED_EXTENSIONS[ext]

    print(f"\n==============================================================")
    print(f"[*] Testing Document: {file_name} ({file_type.upper()})")
    print(f"    Path: {file_path}")
    print(f"==============================================================")
    
    print("Reading file...")
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        print(f"[ERROR] Reading file failed: {e}")
        return False

    file_size_kb = len(file_bytes) / 1024
    print(f"File size: {file_size_kb:.2f} KB")

    print("Encoding to base64...")
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
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            elapsed = time.time() - start_time
            print(f"[OK] Response Status: {response.status}")
            print(f"[TIME] Time taken: {elapsed:.2f} seconds")
            
            result = json.loads(response.read().decode('utf-8'))
            print("\n[RESULT] Analysis Output:\n")
            print(f"  * Status: {result.get('status')}")
            print(f"  * Summary: {result.get('summary')}")
            
            entities = result.get('entities', {})
            print(f"  * Entities Extracted:")
            print(f"    - Names: {', '.join(entities.get('names', [])) or 'None'}")
            print(f"    - Dates: {', '.join(entities.get('dates', [])) or 'None'}")
            print(f"    - Organizations: {', '.join(entities.get('organizations', [])) or 'None'}")
            print(f"    - Locations: {', '.join(entities.get('locations', [])) or 'None'}")
            print(f"    - Amounts: {', '.join(entities.get('amounts', [])) or 'None'}")
            
            print(f"  * Sentiment: {result.get('sentiment')}")
            return True
            
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"[FAIL] Response Status: {e.code}")
        print(f"[TIME] Time taken: {elapsed:.2f} seconds")
        try:
            err_msg = e.read().decode('utf-8')
            print(f"[ERROR] Message: {err_msg}")
        except Exception:
            print(f"[ERROR] Could not read error response.")
        return False
    except urllib.error.URLError as e:
        print(f"[ERROR] Could not connect to API. Is the server running on port 8000? ({e.reason})")
        return False

def discover_and_run_tests(targets):
    """Discovers targets (files, folders, globs) and runs tests on them."""
    files_to_test = []
    
    for target in targets:
        # Check if it is a directory
        if os.path.isdir(target):
            print(f"Searching directory '{target}' for supported documents...")
            for entry in os.listdir(target):
                full_path = os.path.join(target, entry)
                if os.path.isfile(full_path):
                    ext = entry.split('.')[-1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        files_to_test.append(full_path)
        # Check if it contains glob characters
        elif any(char in target for char in ['*', '?']):
            glob_matches = glob.glob(target, recursive=True)
            for match in glob_matches:
                if os.path.isfile(match):
                    ext = match.split('.')[-1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        files_to_test.append(match)
        # Otherwise treat as a single file
        elif os.path.isfile(target):
            files_to_test.append(target)
        else:
            # Check if joining arguments works (e.g. for spaces)
            joined_target = target
            if not os.path.exists(joined_target):
                print(f"[WARN] Target '{target}' not found.")
            else:
                files_to_test.append(joined_target)

    # De-duplicate files list while preserving order
    seen = set()
    files_to_test = [f for f in files_to_test if not (f in seen or seen.add(f))]

    if not files_to_test:
        print("[ERROR] No supported documents found to test.")
        return

    print(f"[INFO] Found {len(files_to_test)} file(s) to analyze.")
    
    success_count = 0
    start_all = time.time()
    
    for idx, file_path in enumerate(files_to_test, 1):
        print(f"\n[Test {idx}/{len(files_to_test)}]")
        if test_document(file_path):
            success_count += 1
            
    total_time = time.time() - start_all
    print(f"\n==============================================================")
    print(f"[SUMMARY] Test Execution Scoreboard")
    print(f"==============================================================")
    print(f"  * Total files tested: {len(files_to_test)}")
    print(f"  * Successful tests:  {success_count} / {len(files_to_test)}")
    print(f"  * Failed tests:      {len(files_to_test) - success_count}")
    print(f"  * Total duration:    {total_time:.2f} seconds")
    print(f"==============================================================")

if __name__ == "__main__":
    # If no arguments provided, automatically test all files in 'samples/'
    if len(sys.argv) < 2:
        default_dir = "samples"
        if os.path.exists(default_dir):
            print(f"[INFO] No arguments provided. Automatically running test suite on '{default_dir}/' directory...")
            discover_and_run_tests([default_dir])
        else:
            print("Usage: python test_api.py <path_to_your_document_or_directory>")
            print("Example: python test_api.py sample.pdf")
            print("Example: python test_api.py samples/")
            sys.exit(1)
    else:
        # If there are arguments, check if we can join them as a single path containing spaces
        # e.g., python test_api.py samples/my document.pdf
        full_args = sys.argv[1:]
        joined_path = " ".join(full_args)
        
        if os.path.exists(joined_path):
            discover_and_run_tests([joined_path])
        else:
            # Otherwise, treat as individual arguments/files
            discover_and_run_tests(full_args)
