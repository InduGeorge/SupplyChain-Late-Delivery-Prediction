# ============================================================
# PREPROCESSING PIPELINE
# Supply Chain Late Delivery Prediction
# LightGBM Deployment Pipeline
#
# Model:
#   25 features
#   13 categorical features
#
# Removed features:
#   - Order Item Discount Rate
#   - Order Item Profit Ratio
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

CATEGORY_VALUES_PATH = os.path.join(
    MODEL_DIR,
    "category_values.pkl"
)


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

print("Loading model artifacts...")

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_COLUMNS_PATH
)

categorical_columns = joblib.load(
    CATEGORICAL_COLUMNS_PATH
)

category_values = joblib.load(
    CATEGORY_VALUES_PATH
)

print(
    f"Model loaded successfully. "
    f"Number of features: {len(feature_columns)}"
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
# 5. VERIFY FEATURE COLUMNS
# ============================================================

if list(feature_columns) != EXPECTED_FEATURES:

    raise ValueError(
        "\nSaved feature_columns.pkl does not match "
        "the expected 25-feature model schema.\n\n"

        f"Saved features:\n"
        f"{list(feature_columns)}\n\n"

        f"Expected features:\n"
        f"{EXPECTED_FEATURES}"
    )


# ============================================================
# 6. VERIFY FEATURE COUNT
# ============================================================

if len(feature_columns) != 25:

    raise ValueError(
        "\nExpected 25 model features, "
        f"but feature_columns.pkl contains "
        f"{len(feature_columns)} features."
    )


# ============================================================
# 7. VERIFY LIGHTGBM INTERNAL FEATURE NAMES
# ============================================================

if hasattr(model, "feature_name_"):

    model_features = list(
        model.feature_name_
    )

    if model_features != EXPECTED_FEATURES:

        raise ValueError(
            "\nLightGBM model feature names do not "
            "match the expected training features.\n\n"

            f"Model features:\n"
            f"{model_features}\n\n"

            f"Expected features:\n"
            f"{EXPECTED_FEATURES}"
        )


# ============================================================
# 8. RAW → MODEL COLUMN NAME MAPPING
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
# 9. FINAL MODEL CATEGORICAL COLUMNS
# ============================================================
#
# IMPORTANT:
#
# categorical_columns.pkl contains the raw dataset names:
#
#   Category Name
#   Customer City
#   ...
#
# But the model uses renamed names:
#
#   Category_Name
#   Customer_City
#   ...
#
# Therefore we explicitly define the model categorical columns
# here instead of directly using categorical_columns.pkl.
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
# 10. VERIFY CATEGORICAL COLUMNS
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
# 11. VERIFY CATEGORICAL COUNT
# ============================================================

if len(MODEL_CATEGORICAL_COLUMNS) != 13:

    raise ValueError(
        "\nExpected 13 categorical features, "
        f"but found "
        f"{len(MODEL_CATEGORICAL_COLUMNS)}."
    )


# ============================================================
# 12. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    """
    Create the engineered features required by the model.

    Creates:

        Order Year
        Order Month
        Order Day
        Order Day of Week
        is_weekend_order
        is_cross_border
    """

    df = df.copy()


    # ========================================================
    # ORDER DATE FEATURES
    # ========================================================

    if "order date (DateOrders)" in df.columns:

        df["order date (DateOrders)"] = pd.to_datetime(
            df["order date (DateOrders)"],
            errors="coerce"
        )


        # ----------------------------------------------------
        # Check invalid dates
        # ----------------------------------------------------

        if df["order date (DateOrders)"].isna().any():

            raise ValueError(
                "\nInvalid value found in "
                "'order date (DateOrders)'."
            )


        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

        df["Order Year"] = (
            df["order date (DateOrders)"].dt.year
        )


        # ----------------------------------------------------
        # Month
        # ----------------------------------------------------

        df["Order Month"] = (
            df["order date (DateOrders)"].dt.month
        )


        # ----------------------------------------------------
        # Day
        # ----------------------------------------------------

        df["Order Day"] = (
            df["order date (DateOrders)"].dt.day
        )


        # ----------------------------------------------------
        # Day of week
        # ----------------------------------------------------

        df["Order Day of Week"] = (
            df["order date (DateOrders)"].dt.day_name()
        )


    # ========================================================
    # WEEKEND FEATURE
    # ========================================================

    if "Order Day of Week" not in df.columns:

        raise ValueError(
            "\nUnable to create "
            "'Order Day of Week'."
        )


    df["is_weekend_order"] = (

        df["Order Day of Week"]

        .isin([
            "Saturday",
            "Sunday"
        ])

        .astype(int)
    )


    # ========================================================
    # CROSS-BORDER FEATURE
    # ========================================================

    if (
        "Customer Country" not in df.columns
        or
        "Order Country" not in df.columns
    ):

        raise ValueError(
            "\nCustomer Country and Order Country "
            "are required to create "
            "'is_cross_border'."
        )


    df["is_cross_border"] = (

        df["Customer Country"]
        !=
        df["Order Country"]

    ).astype(int)


    return df


# ============================================================
# 13. RENAME COLUMNS
# ============================================================

def rename_columns(df):

    """
    Rename raw dataset columns to the exact
    model feature names.
    """

    df = df.copy()


    df.rename(
        columns=COLUMN_RENAME_MAP,
        inplace=True
    )


    return df


# ============================================================
# 14. PREPROCESS INPUT
# ============================================================

def preprocess_input(input_data):

    """
    Convert raw order input into the exact
    25-feature dataframe expected by LightGBM.
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


        # ----------------------------------------------------
        # Numerical
        # ----------------------------------------------------

        "Days for shipment (scheduled)",

        "Order Item Discount",

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
    # EXPLICITLY REJECT REMOVED FEATURES
    # ========================================================

    removed_features = [

        "Order Item Discount Rate",

        "Order Item Profit Ratio"
    ]


    present_removed_features = [

        col

        for col in removed_features

        if col in df.columns
    ]


    if present_removed_features:

        print(
            "\nWarning: The following features were "
            "removed from the model and will be ignored:"
        )

        print(
            present_removed_features
        )


        df.drop(
            columns=present_removed_features,
            inplace=True
        )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = create_features(
        df
    )


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    df = rename_columns(
        df
    )


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

        df[col] = df[col].astype(
            "category"
        )


    # ========================================================
    # FINAL FEATURE ORDER CHECK
    # ========================================================

    if (
        list(df.columns)
        !=
        list(feature_columns)
    ):

        raise ValueError(

            "\nFinal feature order does not "
            "match training feature order.\n\n"

            f"Expected:\n"
            f"{list(feature_columns)}\n\n"

            f"Received:\n"
            f"{list(df.columns)}"
        )


    # ========================================================
    # FINAL FEATURE COUNT CHECK
    # ========================================================

    if df.shape[1] != 25:

        raise ValueError(
            "\nExpected 25 features, "
            f"but received {df.shape[1]} features."
        )


    return df


# ============================================================
# 15. PREDICTION
# ============================================================

def predict(input_data):

    """
    Generate late-delivery prediction.

    Returns:

        prediction:
            0 = Low risk
            1 = High risk

        probability:
            Probability of late delivery.
    """


    # ========================================================
    # PREPROCESS
    # ========================================================

    processed_data = preprocess_input(
        input_data
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        processed_data
    )[0]


    # ========================================================
    # PROBABILITY
    # ========================================================

    probability = model.predict_proba(
        processed_data
    )[0, 1]


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "prediction":
            int(prediction),

        "probability":
            float(probability)
    }


# ============================================================
# 16. FINAL STARTUP VALIDATION
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING PIPELINE LOADED")
print("=" * 60)

print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

print(
    f"Number of categorical features: "
    f"{len(MODEL_CATEGORICAL_COLUMNS)}"
)

print()
print("Categorical model features:")

for column in MODEL_CATEGORICAL_COLUMNS:

    print(
        f"  - {column}"
    )

print()
print("✓ Preprocessing pipeline ready")
print("=" * 60)