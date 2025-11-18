from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. LOAD THE TRAINED MODEL
model = joblib.load("model.joblib")

# 2. INITIALIZE THE API
app = FastAPI()

# 3. DEFINE INPUT DATA FORMAT
class HouseFeatures(BaseModel):
    features: list[float]

# 4. DEFINE ROUTES (Endpoints)
@app.get("/")
def home():
    return {"message": "Housing Price API is ACTIVE"}

# The prediction endpoint
@app.post("/predict")
def predict(house: HouseFeatures):
    # Convert the list of features into a format the model understands (numpy array)
    data = np.array(house.features).reshape(1, -1)
    
    # Ask the model for a prediction
    prediction = model.predict(data)
    
    # Return the result formatted as currency
    return {"predicted_price": float(prediction[0]) * 100000}