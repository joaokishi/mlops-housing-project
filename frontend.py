import streamlit as st
import requests
import json

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

API_URL = "https://joaokishi-mlops-project.hf.space" 

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡")

st.title("Housing Price Regression")
st.markdown(f"""
This app uses a **Random Forest** model deployed on **Hugging Face Spaces** to predict housing prices in California.
The system is powered by a **CI/CD Pipeline** using GitHub Actions.
""")

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
st.sidebar.header("Input Features")
med_inc = st.sidebar.slider("Median Income ($10k)", 0.0, 15.0, 8.32)
house_age = st.sidebar.slider("House Age (Years)", 0, 52, 41)
ave_rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 10.0, 6.98)
ave_bedrooms = st.sidebar.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.02)
population = st.sidebar.slider("Population in Block", 100, 3000, 322)
ave_occup = st.sidebar.slider("Avg Occupancy", 1.0, 6.0, 2.55)
lat = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
lon = st.sidebar.slider("Longitude", -125.0, -114.0, -122.23)

# ---------------------------------------------------------
# PREPARE DATA (Moved Outside)
# ---------------------------------------------------------

payload = {
    "features": [
        med_inc, 
        house_age, 
        ave_rooms, 
        ave_bedrooms, 
        population, 
        ave_occup, 
        lat, 
        lon
    ]
}

# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------
if st.button("💰 Estimate Price"):
    with st.spinner("Connecting to AI Cloud..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                price = result['predicted_price']
                st.success(f"Estimated Value: **${price:,.2f}**")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Failed. Is the API URL correct? \n\nError: {e}")

# ---------------------------------------------------------
# DEBUG INFO
# ---------------------------------------------------------
with st.expander("Technical Details"):
    st.write("Current Data being sent to API:")
    st.json(payload)
    st.write(f"API Endpoint: {API_URL}/predict")