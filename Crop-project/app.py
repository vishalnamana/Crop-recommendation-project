import streamlit as st
import numpy as np
import pickle
import os

# ==============================
# LOAD MODELS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_model = pickle.load(open(os.path.join(BASE_DIR, 'model', 'crop_model.pkl'), 'rb'))

# (Optional) load yield model
yield_model = pickle.load(open(os.path.join(BASE_DIR, 'model', 'yield_model.pkl'), 'rb'))

# ==============================
# UI DESIGN
# ==============================

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.title("🌾 Crop Recommendation & Yield Prediction")
st.write("Enter soil and weather details to get the best crop recommendation")

# ==============================
# INPUT FIELDS
# ==============================

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH value", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# ==============================
# PREDICTION
# ==============================

if st.button("🌱 Predict Crop"):

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Crop Prediction
    crop_prediction = crop_model.predict(data)[0]

    st.success(f"🌿 Recommended Crop: **{crop_prediction}**")

    # Yield (temporary message)
    st.info("🌾 Yield Prediction model is trained (advanced dataset)")

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown("Developed using Machine Learning & Streamlit 🚀")