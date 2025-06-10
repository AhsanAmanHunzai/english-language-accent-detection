#!/usr/bin/env python3
"""
Test script for the Flask accent detection API
"""
import requests
import json

# Base URL for your Flask app
BASE_URL = "http://localhost:5000"

def test_url_analysis():
    """Test analyzing audio from URL"""
    print("Testing URL analysis...")
    
    # Example audio URL (replace with a real audio file URL)
    test_url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"link": test_url},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_file_upload():
    """Test analyzing uploaded audio file"""
    print("Testing file upload analysis...")
    
    # You would need to have an actual audio file for this test
    # Uncomment and modify the path below:
    
    # file_path = "path/to/your/audio/file.wav"
    # try:
    #     with open(file_path, 'rb') as audio_file:
    #         files = {'file': audio_file}
    #         response = requests.post(f"{BASE_URL}/analyze", files=files)
    #         
    #         print(f"Status Code: {response.status_code}")
    #         print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except FileNotFoundError:
    #     print(f"File not found: {file_path}")
    
    print("File upload test skipped - no test file specified")
    print("-" * 50)

def test_error_cases():
    """Test error handling"""
    print("Testing error cases...")
    
    # Test with invalid URL
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"link": "not-a-url"},
        headers={"Content-Type": "application/json"}
    )
    print(f"Invalid URL - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test with missing data
    response = requests.post(f"{BASE_URL}/analyze")
    print(f"No data - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    print("-" * 50)

if __name__ == "__main__":
    print("Flask Accent Detection API Test")
    print("Make sure your Flask app is running on localhost:5000")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✓ Flask app is running")
        else:
            print("✗ Flask app connectivity issue")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Flask app. Make sure it's running on localhost:5000")
        exit(1)
    
    print()
    test_url_analysis()
    test_file_upload()
    test_error_cases()
    
    print("Tests completed!") 