import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.set_page_config(
    page_title="Landslide Risk Prediction",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Landslide Risk Prediction in Hilly Areas")
st.markdown("A machine learning model to assess landslide risk based on environmental factors like rainfall, slope, soil, and more.")

# Load dataset
@st.cache_data
def load_data():
    file_path = "landslide_dataset.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data.dropna()  # Optional: Drop missing rows
    else:
        st.error("Dataset not found. Please upload 'landslide_data.csv'.")
        return pd.DataFrame()  # Return empty DataFrame

data = load_data()

# Proceed only if data is loaded
if not data.empty:

    # Define features & target
    features = [
        'Rainfall_mm',
        'Slope_Angle',
        'Soil_Saturation',
        'Vegetation_Cover',
        'Earthquake_Activity',
        'Proximity_to_Water',
        'Soil_Type_Gravel',
        'Soil_Type_Sand',
        'Soil_Type_Silt'
    ]
    target = 'Landslide'

    # Train model
    try:
        X = data[features]
        y = data[target]
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
    except Exception as e:
        st.error(f"Model training failed: {e}")



    # Input sliders
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
    slope = st.slider("Slope Angle (°)", 0.0, 90.0, 30.0)
    soil_saturation = st.slider("Soil Saturation (0-1)", 0.0, 1.0, 0.5)
    vegetation = st.slider("Vegetation Cover (0-1)", 0.0, 1.0, 0.5)
    earthquake = st.slider("Earthquake Activity (Richter)", 0.0, 10.0, 4.0)
    water_proximity = st.slider("Proximity to Water (0-1)", 0.0, 1.0, 0.5)
    soil_type = st.radio("Soil Type", ['Gravel', 'Sand', 'Silt'])

    # One-hot encode soil type
    soil_gravel = int(soil_type == 'Gravel')
    soil_sand = int(soil_type == 'Sand')
    soil_silt = int(soil_type == 'Silt')

    # Prepare input data
    input_data = pd.DataFrame([[
        rainfall, slope, soil_saturation, vegetation,
        earthquake, water_proximity,
        soil_gravel, soil_sand, soil_silt
    ]], columns=features)

    # Predict
    if st.button("Predict Landslide"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ High risk of landslide!")
        else:
            st.success("✅ Low risk of landslide.")

    # Optional: Show dataset
    if st.checkbox("Show dataset"):
        st.write(data.head())

else:
    st.stop()
