# 🌧️ Landslide Risk Prediction in Hilly Areas

This Streamlit web app predicts the risk of landslides based on environmental data like rainfall, slope, soil saturation, and more. It leverages a trained **Random Forest Classifier** to assess whether a location is at **high or low risk** of a landslide.

---

## 📊 Features

- Interactive sliders for environmental inputs
- Soil type selection (Gravel, Sand, Silt)
- Predicts landslide risk using a machine learning model
- Displays sample dataset (optional)
- Clean UI built with Streamlit

---

## 🧠 ML Model

- **Model**: Random Forest Classifier
- **Library**: Scikit-learn
- **Training Data**: Includes features like:
  - Rainfall (mm)
  - Slope Angle (degrees)
  - Soil Saturation
  - Vegetation Cover
  - Earthquake Activity (Richter scale)
  - Proximity to Water
  - Soil Type (One-hot encoded)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/landslide-risk-predictor.git
cd landslide-risk-predictor
