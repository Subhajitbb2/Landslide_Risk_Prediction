# Install required libraries (if not already installed)
!pip install scikit-learn pandas

# Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "landslide_dataset.csv"
if not os.path.exists(file_path):
    from google.colab import files
    print("Please upload 'landslide_dataset.csv' file.")
    uploaded = files.upload()
    file_path = next(iter(uploaded))

# Read data
data = pd.read_csv(file_path)
data = data.dropna()

# Show available columns
print("\nAvailable columns:", data.columns.tolist())

# Optional: Rename columns if needed
column_mapping = {
    'Rainfall': 'Rainfall_mm',
    'Slope': 'Slope_Angle',
    'Earthquake': 'Earthquake_Activity',
    'Water_Proximity': 'Proximity_to_Water'
}
data.rename(columns=column_mapping, inplace=True)

# Check if Soil_Type is present
if 'Soil_Type' in data.columns:
    # One-hot encode Soil_Type
    soil_dummies = pd.get_dummies(data['Soil_Type'], prefix='Soil_Type')
    data = pd.concat([data.drop('Soil_Type', axis=1), soil_dummies], axis=1)

# Set features and target
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

# Check if all required features are present
missing_features = [col for col in features if col not in data.columns]
if missing_features:
    print("\nMissing columns:", missing_features)
    raise ValueError("Please check your dataset columns.")

# Split data
X = data[features]
y = data[target]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Prediction input
print("\n--- Landslide Risk Prediction ---")
rainfall = float(input("Enter Rainfall (mm): "))
slope = float(input("Enter Slope Angle (degrees): "))
soil_saturation = float(input("Enter Soil Saturation (0-1): "))
vegetation = float(input("Enter Vegetation Cover (0-1): "))
earthquake = float(input("Enter Earthquake Activity (Richter scale): "))
water_proximity = float(input("Enter Proximity to Water (0-1): "))
soil_type = input("Enter Soil Type (Gravel / Sand / Silt): ").strip().capitalize()

# One-hot encoding user input
soil_gravel = int(soil_type == 'Gravel')
soil_sand = int(soil_type == 'Sand')
soil_silt = int(soil_type == 'Silt')

input_data = pd.DataFrame([[
    rainfall, slope, soil_saturation, vegetation,
    earthquake, water_proximity,
    soil_gravel, soil_sand, soil_silt
]], columns=features)

# Predict
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
risk_probability = probabilities[1]

# Output
print("\n--- Prediction Result ---")
if prediction == 1:
    print(f"⚠️ High Risk of Landslide Detected!")
else:
    print(f"✅ Low Risk of Landslide.")
print(f"Probability of Landslide: {risk_probability:.2%}")
