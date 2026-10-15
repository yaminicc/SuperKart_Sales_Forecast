"""
Data registration step.

Validates that the raw SuperKart dataset has the expected schema before the
rest of the pipeline is allowed to treat it as "registered".  This is the
pipeline's first gate: a renamed or missing column fails the run here, loudly,
instead of producing a silently wrong model three jobs later.
"""
import pandas as pd

RAW_PATH = "superkart_project/data/SuperKart.csv"

EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content",
    "Product_Allocated_Area", "Product_Type", "Product_MRP",
    "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]
TARGET = "Product_Store_Sales_Total"

df = pd.read_csv(RAW_PATH)

# 1. Schema check: every expected column must be present
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# 2. Basic sanity checks on the target
if df[TARGET].isna().any():
    raise ValueError("Target column contains missing values")
if (df[TARGET] < 0).any():
    raise ValueError("Target column contains negative sales totals")

# 3. Report a summary so the Actions log shows what was registered
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print(f"\n{TARGET} summary:")
print(df[TARGET].describe().round(2))
print("\nStores:")
print(df["Store_Id"].value_counts())
print("\nSugar content categories (note the dirty 'reg' label, fixed in prep.py):")
print(df["Product_Sugar_Content"].value_counts())
