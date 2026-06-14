import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 House Price Prediction Dashboard")

# 1. Load pipeline artifacts
if not os.path.exists('HousePrice.pkl'):
    st.error("Error: 'HousePrice.pkl' not found!")
    st.stop()

try:
    with open('HousePrice.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts['model']
    preprocessor = artifacts['preprocessor']
    scaler = artifacts['scaler']
except Exception as e:
    st.error("❌ Failed to load pipeline artifacts.")
    st.exception(e)
    st.stop()

# 2. Form interface
with st.form("prediction_form"):
    st.subheader("📋 Enter Property Specifications")
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Total Area (in SqFt)", min_value=100, max_value=20000, value=4000, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=1)
        stories = st.slider("Stories / Floors", min_value=1, max_value=4, value=2)
        parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1)
        furnishingstatus = st.selectbox("Furnishing Status", options=["furnished", "semi-furnished", "unfurnished"])

    with col2:
        mainroad = st.selectbox("Is it on a Main Road?", options=["yes", "no"])
        guestroom = st.selectbox("Does it have a Guestroom?", options=["yes", "no"])
        basement = st.selectbox("Does it have a Basement?", options=["yes", "no"])
        hotwaterheating = st.selectbox("Has Hot Water Heating?", options=["yes", "no"])
        airconditioning = st.selectbox("Has Air Conditioning?", options=["yes", "no"])
        prefarea = st.selectbox("Is it in a Preferred Area?", options=["yes", "no"])
        
    submit_button = st.form_submit_button("Calculate Estimated Price")

# 3. Predict & Plot Visualizations
if submit_button:
    input_data = {
        'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories,
        'mainroad': mainroad, 'guestroom': guestroom, 'basement': basement,
        'hotwaterheating': hotwaterheating, 'airconditioning': airconditioning,
        'parking': parking, 'prefarea': prefarea, 'furnishingstatus': furnishingstatus
    }
    user_inputs = pd.DataFrame([input_data])
    
    try:
        user_inputs_processed = preprocessor.transform(user_inputs)
        user_inputs_scaled = scaler.transform(user_inputs_processed)
        predicted_raw = model.predict(user_inputs_scaled)
        
        predicted_price = float(predicted_raw)
        st.success(f"### 🎉 Estimated Evaluation Value: ${predicted_price:,.2f}")
        
        # Market context chart
        if os.path.exists('experiment_data.csv'):
            st.write("---")
            st.subheader("📊 Market Valuation Context")
            df = pd.read_csv('experiment_data.csv')
            price_col = [col for col in df.columns if col.lower() == 'price']
            
            if price_col:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df[price_col[0]], kde=True, color="skyblue", ax=ax)
                plt.axvline(predicted_price, color="red", linestyle="--", linewidth=2.5, label=f"Your Prediction: ${predicted_price:,.0f}")
                plt.title("Where your property stands in the market distribution")
                plt.xlabel("Price")
                plt.ylabel("Property Count")
                plt.legend()
                st.pyplot(fig)
                
    except Exception as e:
        st.error("Error processing pipeline inputs.")
        st.exception(e)
