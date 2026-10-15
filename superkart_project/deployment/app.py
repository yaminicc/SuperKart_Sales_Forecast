import json
import os

import joblib
import pandas as pd
import streamlit as st

# Load the model committed by the pipeline (sits next to this file)
HERE = os.path.dirname(__file__)
model = joblib.load(os.path.join(HERE, "best_superkart_sales_model_v1.joblib"))

metrics = {}
metrics_path = os.path.join(HERE, "model_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

st.title("SuperKart Sales Forecast")
st.write(
    "Enter the product and store details to forecast the total sales revenue "
    "of that product at that store."
)

if metrics:
    t = metrics["test"]
    st.caption(
        f"Current production model — test RMSE: {t['rmse']:.0f}, "
        f"MAE: {t['mae']:.0f}, R²: {t['r2']:.3f} "
        f"(baseline RMSE: {metrics['baseline']['rmse']:.0f})"
    )

# ---- Product details ----
st.subheader("Product")
Product_Weight = st.slider("Product Weight (kg)", 4.0, 22.0, 12.5, step=0.1)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.slider(
    "Allocated Shelf Area (share of store)", 0.004, 0.30, 0.06, step=0.001, format="%.3f"
)
Product_Type = st.selectbox(
    "Product Type",
    [
        "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
        "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks",
        "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood",
    ],
)
Product_MRP = st.number_input("Product MRP", min_value=30.0, max_value=270.0, value=147.0)

# ---- Store details ----
st.subheader("Store")
Store_Id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
)

input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
}])

if st.button("Forecast Sales"):
    pred = float(model.predict(input_data)[0])
    st.success(f"Forecasted total sales: {pred:,.2f}")
    if metrics:
        mae = metrics["test"]["mae"]
        st.write(f"Typical error (test MAE): ± {mae:,.0f}")
