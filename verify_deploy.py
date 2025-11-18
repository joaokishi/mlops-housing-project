import requests

API_URL = "https://joaokishi-mlops-project.hf.space" 

endpoint = f"{API_URL}/predict"

# Example data (same as before)
payload = {
    "features": [8.32, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]
}

print(f"Sending request to: {endpoint}...")
try:
    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        price = data['predicted_price']
        print("✅ SUCCESS!")
        print(f"The AI predicts this house is worth: ${price:,.2f}")
    else:
        print("❌ ERROR:", response.status_code)
        print(response.text)
        
except Exception as e:
    print("❌ CONNECTION FAILED")
    print(e)