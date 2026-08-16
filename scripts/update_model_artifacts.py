# ============================================================
# UPDATE MODEL ARTIFACTS
# Supply Chain Late Delivery Prediction
# ============================================================

import os
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# ARTIFACT PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "late_delivery_lgbm.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

CATEGORICAL_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "categorical_columns.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("MODEL ARTIFACT UPDATE")
print("=" * 60)

print("\nLoading LightGBM model...")

model = joblib.load(
    MODEL_PATH
)

print("✓ Model loaded")


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

print(
    f"✓ Feature columns loaded: "
    f"{len(feature_columns)}"
)


# ============================================================
# FINAL CATEGORICAL COLUMNS
# ============================================================
#
# These MUST match the final column names used by
# the trained LightGBM model.
# ============================================================

categorical_columns = [

    "Type",

    "Category_Name",

    "Customer_City",

    "Customer_Segment",

    "Department_Name",

    "Market",

    "Order_Country",

    "Order_Region",

    "Order_State",

    "Order_Status",

    "Product_Name",

    "Shipping_Mode",

    "Order_Day_of_Week"
]


# ============================================================
# VERIFY COLUMNS EXIST
# ============================================================

missing_columns = [
    col
    for col in categorical_columns
    if col not in feature_columns
]


if missing_columns:

    raise ValueError(
        "\nThe following categorical columns "
        "are not present in feature_columns.pkl:\n"
        f"{missing_columns}"
    )


# ============================================================
# SAVE UPDATED ARTIFACT
# ============================================================

joblib.dump(
    categorical_columns,
    CATEGORICAL_COLUMNS_PATH
)


print(
    "\n✓ categorical_columns.pkl updated"
)


# ============================================================
# VERIFY SAVED ARTIFACT
# ============================================================

saved_categorical_columns = joblib.load(
    CATEGORICAL_COLUMNS_PATH
)


print(
    "\nFinal categorical columns:"
)

for i, col in enumerate(
    saved_categorical_columns,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# VERIFY LIGHTGBM MODEL FEATURES
# ============================================================

print(
    "\nChecking LightGBM model features..."
)


if list(model.feature_name_) != list(
    feature_columns
):

    raise ValueError(
        "\nLightGBM model features do not match "
        "feature_columns.pkl."
    )


print(
    "✓ LightGBM feature names match "
    "feature_columns.pkl"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "✓ MODEL ARTIFACT UPDATE COMPLETED"
)

print(
    "=" * 60
)
