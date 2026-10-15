"""
Data preparation step.

Cleans the registered dataset and writes train/test splits that the training
job picks up as a GitHub Actions artifact.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "superkart_project/data/SuperKart.csv"
TARGET = "Product_Store_Sales_Total"

df = pd.read_csv(RAW_PATH)

# --- Cleaning -----------------------------------------------------------
# 1. Fix the dirty category label: 'reg' is a typo of 'Regular'
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# 2. Drop columns that carry no predictive signal
#    - Product_Id is a unique identifier (8763 unique values for 8763 rows)
#    - Store_Establishment_Year is fully determined by Store_Id (4 stores, 3 years),
#      so it duplicates information the model already gets from Store_Id
df = df.drop(columns=["Product_Id", "Store_Establishment_Year"])

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here would make training and serving use
# different representations and silently break predictions.

# --- Split --------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# This is a regression target so there is nothing to stratify on.
# The dataset has no time column, so a random split is the correct choice;
# a time-based split would only make sense if sales were dated.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train rows: {len(Xtrain)}, Test rows: {len(Xtest)}")
print(f"Features: {list(X.columns)}")
print("\nSugar content after cleaning:")
print(X["Product_Sugar_Content"].value_counts())
