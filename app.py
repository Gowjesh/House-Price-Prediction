import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("house_price_model.pkl")

columns = joblib.load("columns.pkl")

# Load scaler only if available
scaler = None

if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")


# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")

st.write("Enter House Details")


# -----------------------------
# User Inputs
# -----------------------------

area = st.number_input(
    "Area",
    min_value=20,
    max_value=320,
    value=20
)

year = st.number_input(
    "Year",
    min_value=1370,
    max_value=1402,
    value=1395
)

rooms = st.number_input(
    "Rooms",
    min_value=1,
    max_value=10,
    value=2
)

floor = st.number_input(
    "Floor",
    min_value=1,
    max_value=8,
    value=1
)

locations = []

for col in columns:
    if col.startswith("Location_"):
        locations.append(col.replace("Location_", ""))

location = st.selectbox(
    "Location",
    sorted(locations)
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Price"):

    house_age = 2026 - year

    input_df = pd.DataFrame(
        {
            "area": [area],
            "Rooms": [rooms],
            "Floor": [floor],
            "house_age": [house_age]
        }
    )

    # Create all training columns

    for col in columns:

        if col not in input_df.columns:

            input_df[col] = 0


    # Set selected location

    selected_location = "Location_" + location

    if selected_location in input_df.columns:

        input_df[selected_location] = 1


    # Arrange columns exactly like training

    input_df = input_df[columns]


    # Scale only if scaler exists

    if scaler is not None:

        input_df = scaler.transform(input_df)


    prediction = model.predict(input_df)

    st.success(
        f"Predicted House Price : ₹ {prediction[0]:,.2f}"
    )