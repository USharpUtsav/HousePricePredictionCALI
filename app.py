import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the best model
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI: input fields for prediction
st.title("House Price Prediction")

# Input fields for user to enter house features with default values
longitude = st.number_input('Longitude', value=-119.0, format="%.6f")
latitude = st.number_input('Latitude', value=36.0, format="%.6f")
house_age = st.number_input('House Age (years)', value=25.0)
total_rooms = st.number_input('Total Rooms', value=5.0, format="%.1f")
total_bedrooms = st.number_input('Total Bedrooms', value=4.0, format="%.1f")
population = st.number_input('Population', value=5.0, format="%.1f")
households = st.number_input('Households With Neighbouring Area', value=4.0, format="%.1f")
median_income = st.number_input('Median income per 100,000 dollars ($)', value=7.5, format="%.2f")

# Create input array from user input
input_data = np.array([[longitude, latitude, house_age, total_rooms, total_bedrooms,
                        population, households, median_income]])

# Convert input data to DataFrame with appropriate column names
input_df = pd.DataFrame(input_data, columns=['longitude', 'latitude', 'house_age',
                                             'total_rooms', 'total_bedrooms',
                                             'population', 'households', 'median_income'])

# Scale the input data using the loaded scaler
scaled_input = scaler.transform(input_df)

# Predict house price
if st.button("Predict"):
    predicted_price = model.predict(scaled_input)[0]

    # Conversion rate from USD to NPR
    conversion_rate = 135.1933  # Example conversion rate, please update as necessary
    predicted_price_npr = predicted_price * conversion_rate

    # Display the prediction
    st.write(f"Predicted House Price: ${predicted_price:,.2f}")
    st.write(f"Predicted House Price in Nepali Rupees: रू{predicted_price_npr:,.2f}")