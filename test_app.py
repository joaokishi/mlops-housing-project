from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Housing Price API is ACTIVE"}

def test_prediction():
    # Send fake data to the API
    fake_house = {
        "features": [8.32, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]
    }
    response = client.post("/predict", json=fake_house)
    
    # Check if we get a success code
    assert response.status_code == 200
    
    # Check if the price is a number greater than 0
    json_data = response.json()
    assert "predicted_price" in json_data
    assert json_data["predicted_price"] > 0