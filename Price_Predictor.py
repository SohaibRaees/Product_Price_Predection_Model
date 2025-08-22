import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("price_predictor.pkl")

st.set_page_config(page_title="Product Price Predictor", page_icon="💰", layout="centered")

st.title("💰 Product Price Predictor")
st.write("Enter product details to predict the price.")

# ------------------------------
# Input features
# ------------------------------
status = st.selectbox("Status", ["Ok", "Cancelled", "Returned"])
category = st.selectbox("Category", ["Electronics", "Clothing", "Home", "Other"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "Cash", "Bank Transfer", "Other"])

qty_ordered = st.number_input("Quantity Ordered", min_value=1, value=1)
discount_amount = st.number_input("Discount Amount", min_value=0.0, value=0.0)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
dayofweek = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
is_weekend = st.selectbox("Is Weekend", [0, 1])
quarter = st.number_input("Quarter (1-4)", min_value=1, max_value=4, value=1)

# ------------------------------
# Create dataframe for model
# ------------------------------
input_dict = {
    "status": [status],
    "category_name_1": [category],
    "payment_method": [payment_method],
    "qty_ordered": [qty_ordered],
    "discount_amount": [discount_amount],
    "Year": [year],
    "Month": [month],
    "dayofweek": [dayofweek],
    "is_weekend": [is_weekend],
    "quarter": [quarter],
}

input_df = pd.DataFrame(input_dict)

# Apply the same dummy encoding as training
input_encoded = pd.get_dummies(input_df)

# Align with training columns
model_features = model.feature_names_in_  # columns used in training
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💲 Predicted Price: {prediction:.2f}")
