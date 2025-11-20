# test_request.py
# Helper script to generate and send dummy data to the prediction API

import requests
import json
import os

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY", "changeme")

def get_feature_length():
    """Get the required feature length from the health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        return data.get("feature_len", 100)  # fallback to 100
    except Exception as e:
        print(f"Error getting feature length: {e}")
        return 100  # fallback

def load_json_data(json_file="./output.json"):
    """Load data from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return None

def generate_dummy_data(feature_length=100):
    """Generate dummy JSON data with the correct feature length"""
    import random
    
    # Generate random gene features
    gene_features = [random.uniform(-2.0, 2.0) for _ in range(feature_length)]
    
    payload = {
        "gene_features": gene_features
    }
    return payload

def send_prediction_request(payload):
    """Send the prediction request to the API"""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        headers=headers,
        json=payload
    )
    
    # Always return the response object and status for better error handling
    return response

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Breast Cancer Relapse Prediction API")
    print("=" * 60)
    
    print("\n[1] Getting feature length from API...")
    try:
        feature_len = get_feature_length()
        print(f"    ✓ Feature length: {feature_len}")
    except Exception as e:
        print(f"    ✗ Failed to connect to API: {e}")
        print("    Make sure the server is running: uvicorn server:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    print("\n[2] Loading or generating gene features...")
    # Try to load from JSON file first
    dummy_data = load_json_data("dummy_data.json")
    if dummy_data and 'gene_features' in dummy_data:
        # Validate length matches
        if len(dummy_data['gene_features']) == feature_len:
            print(f"    ✓ Loaded {len(dummy_data['gene_features'])} gene features from dummy_data.json")
        else:
            print(f"    ⚠ JSON file has {len(dummy_data['gene_features'])} features, but API expects {feature_len}")
            print(f"    ✓ Generating new data with correct length...")
            dummy_data = generate_dummy_data(feature_len)
            print(f"    ✓ Generated {len(dummy_data['gene_features'])} gene features")
    else:
        print(f"    ✓ Generating random gene features...")
        dummy_data = generate_dummy_data(feature_len)
        print(f"    ✓ Generated {len(dummy_data['gene_features'])} gene features")
    
    print("\n[3] Sending prediction request...")
    try:
        response = send_prediction_request(dummy_data)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"    ✗ API returned status code: {response.status_code}")
            print(f"    Response: {response.text}")
            exit(1)
        
        result = response.json()
        
        # Validate response structure
        if 'risk_percent' not in result or 'will_relapse' not in result:
            print("    ✗ Unexpected response structure:")
            print(json.dumps(result, indent=2))
            exit(1)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Risk Percent:   {result['risk_percent']:.2f}%")
        print(f"Will Relapse:   {'Yes' if result['will_relapse'] else 'No'}")
        print("=" * 60)
        
        print("\nFull JSON response:")
        print(json.dumps(result, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"    Status code: {e.response.status_code}")
            print(f"    Response: {e.response.text}")
        exit(1)
    except KeyError as e:
        print(f"    ✗ Missing key in response: {e}")
        try:
            print(f"    Response received: {json.dumps(result, indent=2)}")
        except:
            print(f"    Could not parse response")
        exit(1)
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

