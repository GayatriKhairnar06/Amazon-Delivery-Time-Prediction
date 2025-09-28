import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# =========================
# Load trained model + processed dataset structure
# =========================
model = joblib.load("best_model.pkl")
processed_df = pd.read_csv("amazon_processed.csv")   # just to keep same column order
expected_columns = processed_df.drop("Delivery_Time", axis=1).columns

# =========================
# Helper functions (from main.py)
# =========================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    curvature_distance = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(curvature_distance), sqrt(1-curvature_distance))
    R = 6371  # Earth radius (km)
    return R * c

def rush_hour(hour):
    return 1 if hour in [8, 9, 10, 17, 18, 19, 20] else 0

# =========================
# Streamlit UI
# =========================
st.title("🚚 Amazon Delivery Time Prediction")

st.header("Enter Delivery Details:")

# Agent info
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)

# Location details
store_lat = st.number_input("Store Latitude", value=19.0760, format="%.6f")
store_lon = st.number_input("Store Longitude", value=72.8777, format="%.6f")
drop_lat = st.number_input("Drop Latitude", value=19.2183, format="%.6f")
drop_lon = st.number_input("Drop Longitude", value=72.9781, format="%.6f")

# Date and Time
order_date = st.date_input("Order Date", datetime.today().date())
order_time = st.time_input("Order Time", datetime.now().time())
pickup_time = st.time_input("Pickup Time", datetime.now().time())

# Categorical features
weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
vehicle = st.selectbox("Vehicle", ["Bike", "Car", "Van"])
area = st.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
category = st.selectbox("Category", ["Electronics", "Clothing", "Groceries", "Others"])

# =========================
# Process Features
# =========================
if st.button("Predict Delivery Time"):
    # Derived features
    distance = np.sqrt((store_lat - drop_lat)**2 + (store_lon - drop_lon)**2)
    distance_km = haversine(store_lat, store_lon, drop_lat, drop_lon)

    order_datetime = pd.to_datetime(str(order_date) + " " + str(order_time))
    pickup_datetime = pd.to_datetime(str(order_date) + " " + str(pickup_time))

    order_year = order_datetime.year
    order_month = order_datetime.month
    order_day = order_datetime.day
    order_weekday = order_datetime.weekday()
    is_weekend = 1 if order_weekday >= 5 else 0
    order_hour = order_datetime.hour
    pickup_hour = pickup_datetime.hour
    rush = rush_hour(order_hour)

    pickup_delay_minutes = max((pickup_datetime - order_datetime).total_seconds() / 60, 0)

    # Build input row
    input_data = {
        "Agent_Age": [agent_age],
        "Agent_Rating": [agent_rating],
        "Store_Latitude": [store_lat],
        "Store_Longitude": [store_lon],
        "Drop_Latitude": [drop_lat],
        "Drop_Longitude": [drop_lon],
        "Weather": [weather],
        "Traffic": [traffic],
        "Vehicle": [vehicle],
        "Area": [area],
        "Category": [category],
        "Distance": [distance],
        "Distance_km": [distance_km],
        "Order_Date": [str(order_date)],    # keep as string
        "Order_Time": [str(order_time)],
        "Pickup_Time": [str(pickup_time)],
        "Order_DateTime": [str(order_datetime)],
        "Pickup_DateTime": [str(pickup_datetime)],
        "Order_Year": [order_year],
        "Order_Month": [order_month],
        "Order_Day": [order_day],
        "Order_Weekday": [order_weekday],
        "Is_Weekend": [is_weekend],
        "Order_Hour": [order_hour],
        "Pickup_Hour": [pickup_hour],
        "Rush_Hour": [rush],
        "Pickup_Delay_Minutes": [pickup_delay_minutes]
    }

    input_df = pd.DataFrame(input_data)

    # Align with training columns
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} minutes")
