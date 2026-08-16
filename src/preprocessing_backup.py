# ============================================================
# PREPROCESSING PIPELINE
# Supply Chain Late Delivery Prediction
# LightGBM Deployment Pipeline
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
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
# 2. MODEL ARTIFACT PATHS
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
# 3. LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

categorical_columns = joblib.load(
    CATEGORICAL_COLUMNS_PATH
)


# ============================================================
# 4. FINAL MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Type",
    "Days_for_shipment_(scheduled)",
    "Category_Name",
    "Customer_City",
    "Customer_Segment",
    "Department_Name",
    "Market",
    "Order_Country",
    "Order_Item_Discount",
    "Order_Item_Discount_Rate",
    "Order_Item_Profit_Ratio",
    "Order_Item_Quantity",
    "Sales",
    "Order_Item_Total",
    "Order_Profit_Per_Order",
    "Order_Region",
    "Order_State",
    "Order_Status",
    "Product_Name",
    "Product_Price",
    "Shipping_Mode",
    "Order_Year",
    "Order_Month",
    "Order_Day",
    "Order_Day_of_Week",
    "is_weekend_order",
    "is_cross_border"
]


# ============================================================
# 5. VERIFY SAVED FEATURE COLUMNS
# ============================================================

if list(feature_columns) != EXPECTED_FEATURES:

    raise ValueError(
        "\nSaved feature_columns.pkl does not match "
        "the expected training features.\n\n"
        f"Saved features:\n{list(feature_columns)}\n\n"
        f"Expected features:\n{EXPECTED_FEATURES}"
    )


# ============================================================
# 6. VERIFY LIGHTGBM INTERNAL FEATURE NAMES
# ============================================================

if hasattr(model, "feature_name_"):

    if list(model.feature_name_) != EXPECTED_FEATURES:

        raise ValueError(
            "\nLightGBM model feature names do not match "
            "the expected training features.\n\n"
            f"Model features:\n{list(model.feature_name_)}\n\n"
            f"Expected features:\n{EXPECTED_FEATURES}"
        )


# ============================================================
# 7. RAW → MODEL COLUMN NAME MAPPING
# ============================================================

COLUMN_RENAME_MAP = {

    "Days for shipment (scheduled)":
        "Days_for_shipment_(scheduled)",

    "Category Name":
        "Category_Name",

    "Customer City":
        "Customer_City",

    "Customer Segment":
        "Customer_Segment",

    "Department Name":
        "Department_Name",

    "Order Country":
        "Order_Country",

    "Order Item Discount":
        "Order_Item_Discount",

    "Order Item Discount Rate":
        "Order_Item_Discount_Rate",

    "Order Item Profit Ratio":
        "Order_Item_Profit_Ratio",

    "Order Item Quantity":
        "Order_Item_Quantity",

    "Order Item Total":
        "Order_Item_Total",

    "Order Profit Per Order":
        "Order_Profit_Per_Order",

    "Order Region":
        "Order_Region",

    "Order State":
        "Order_State",

    "Order Status":
        "Order_Status",

    "Product Name":
        "Product_Name",

    "Product Price":
        "Product_Price",

    "Shipping Mode":
        "Shipping_Mode",

    "Order Year":
        "Order_Year",

    "Order Month":
        "Order_Month",

    "Order Day":
        "Order_Day",

    "Order Day of Week":
        "Order_Day_of_Week"
}


# ============================================================
# 8. CATEGORICAL MODEL COLUMNS
# ============================================================
#
# We use the FINAL model column names here.
#
# This avoids depending on possibly outdated names in
# categorical_columns.pkl.
# ============================================================

MODEL_CATEGORICAL_COLUMNS = [
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
# 9. VERIFY CATEGORICAL COLUMNS
# ============================================================

missing_categorical_features = [
    col
    for col in MODEL_CATEGORICAL_COLUMNS
    if col not in EXPECTED_FEATURES
]

if missing_categorical_features:

    raise ValueError(
        "\nThe following categorical columns are not "
        "present in the model features:\n"
        f"{missing_categorical_features}"
    )


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Create the features that were generated during
    model training.

    Creates:

        Order Year
        Order Month
        Order Day
        is_weekend_order
        is_cross_border
    """

    df = df.copy()


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "order date (DateOrders)",
        "Order Day of Week",
        "Customer Country",
        "Order Country"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing columns required for "
            "feature engineering:\n"
            f"{missing_columns}"
        )


    # ========================================================
    # CONVERT ORDER DATE
    # ========================================================

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="raise"
    )


    # ========================================================
    # DATE FEATURES
    # ========================================================

    df["Order Year"] = (
        df["order date (DateOrders)"].dt.year
    )

    df["Order Month"] = (
        df["order date (DateOrders)"].dt.month
    )

    df["Order Day"] = (
        df["order date (DateOrders)"].dt.day
    )


    # ========================================================
    # WEEKEND ORDER
    # ========================================================
    #
    # Your training data contains day names:
    #
    # Monday
    # Tuesday
    # Wednesday
    # Thursday
    # Friday
    # Saturday
    # Sunday
    #
    # ========================================================

    df["is_weekend_order"] = (
        df["Order Day of Week"]
        .astype(str)
        .str.strip()
        .isin(["Saturday", "Sunday"])
        .astype(int)
    )


    # ========================================================
    # CROSS-BORDER ORDER
    # ========================================================

    df["is_cross_border"] = (
        df["Customer Country"]
        .astype(str)
        .str.strip()
        !=
        df["Order Country"]
        .astype(str)
        .str.strip()
    ).astype(int)


    return df


# ============================================================
# 11. RENAME COLUMNS
# ============================================================

def rename_columns(df):
    """
    Rename raw dataset columns to the exact column names
    used during LightGBM training.
    """

    df = df.copy()

    df.rename(
        columns=COLUMN_RENAME_MAP,
        inplace=True
    )

    return df


# ============================================================
# 12. PREPROCESS INPUT
# ============================================================

def preprocess_input(input_data):
    """
    Convert raw order input into the exact 27-feature
    dataframe expected by the trained LightGBM model.

    Input:
        Dictionary or pandas DataFrame containing raw
        dataset-style column names.

    Output:
        DataFrame with exactly 27 model features.
    """

    # ========================================================
    # CONVERT INPUT TO DATAFRAME
    # ========================================================

    if isinstance(input_data, dict):

        df = pd.DataFrame(
            [input_data]
        )

    elif isinstance(input_data, pd.DataFrame):

        df = input_data.copy()

    else:

        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame."
        )


    # ========================================================
    # REQUIRED RAW INPUT COLUMNS
    # ========================================================

    required_raw_columns = [

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        "Type",
        "Category Name",
        "Customer City",
        "Customer Country",
        "Customer Segment",
        "Department Name",
        "Market",
        "Order Country",
        "Order Region",
        "Order State",
        "Order Status",
        "Product Name",
        "Shipping Mode",
        "Order Day of Week",

        # ----------------------------------------------------
        # Numerical
        # ----------------------------------------------------

        "Days for shipment (scheduled)",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Product Price",

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        "order date (DateOrders)"
    ]


    # ========================================================
    # CHECK REQUIRED RAW COLUMNS
    # ========================================================

    missing_columns = [
        col
        for col in required_raw_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required input fields:\n"
            f"{missing_columns}"
        )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = create_features(df)


    # ========================================================
    # RENAME RAW COLUMNS
    # ========================================================

    df = rename_columns(df)


    # ========================================================
    # CHECK MODEL FEATURES
    # ========================================================

    missing_model_features = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing_model_features:

        raise ValueError(
            "\nUnable to create required model features:\n"
            f"{missing_model_features}"
        )


    # ========================================================
    # SELECT EXACT MODEL FEATURES
    # ========================================================

    df = df[
        list(feature_columns)
    ].copy()


    # ========================================================
    # CONVERT CATEGORICAL COLUMNS
    # ========================================================

    for col in MODEL_CATEGORICAL_COLUMNS:

        df[col] = df[col].astype("category")


    # ========================================================
    # FINAL FEATURE ORDER CHECK
    # ========================================================

    if list(df.columns) != list(feature_columns):

        raise ValueError(
            "\nFinal feature order does not match "
            "the training feature order.\n\n"
            f"Expected:\n{list(feature_columns)}\n\n"
            f"Received:\n{list(df.columns)}"
        )


    # ========================================================
    # FINAL FEATURE COUNT CHECK
    # ========================================================

    if df.shape[1] != 27:

        raise ValueError(
            f"\nExpected 27 features, "
            f"but received {df.shape[1]} features."
        )


    return df


# ============================================================
# 13. PREDICTION
# ============================================================

def predict(input_data):
    """
    Generate late-delivery prediction and probability.

    Returns:

        prediction:
            0 = Low risk
            1 = High risk

        probability:
            Probability of late delivery.
    """

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_data = preprocess_input(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        processed_data
    )[0]


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        processed_data
    )[0, 1]


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

