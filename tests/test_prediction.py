# ============================================================
# TEST COMPLETE INFERENCE PIPELINE
# Supply Chain Late Delivery Prediction
# ============================================================

import sys
import os


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# IMPORT PREPROCESSING PIPELINE
# ============================================================

from src.preprocessing import (
    preprocess_input,
    predict,
    feature_columns,
    MODEL_CATEGORICAL_COLUMNS
)


# ============================================================
# TEST ORDER
# ============================================================

test_order = {

    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    "Type": "DEBIT",

    "Category Name": "Sporting Goods",

    "Customer City": "Caguas",

    "Customer Country": "Puerto Rico",

    "Customer Segment": "Consumer",

    "Department Name": "Fan Shop",

    "Market": "USCA",

    "Order Country": "Puerto Rico",

    "Order Region": "Caribbean",

    "Order State": "PR",

    "Order Status": "COMPLETE",

    "Product Name":
        "Nike Men's Dri-FIT Victory Golf Polo",

    "Shipping Mode": "Standard Class",

    "Order Day of Week": "Saturday",


    # --------------------------------------------------------
    # NUMERICAL INPUTS
    # --------------------------------------------------------

    "Days for shipment (scheduled)": 4,

    "Order Item Discount": 20.0,

    "Order Item Quantity": 1,

    "Sales": 200.0,

    "Order Item Total": 180.0,

    "Order Profit Per Order": 60.0,

    "Product Price": 200.0,


    # --------------------------------------------------------
    # RAW DATE
    # --------------------------------------------------------

    "order date (DateOrders)": "2017-07-15"
}


# ============================================================
# START TEST
# ============================================================

print()
print("=" * 60)
print("SUPPLY CHAIN LATE DELIVERY")
print("INFERENCE PIPELINE TEST")
print("=" * 60)


# ============================================================
# PREPROCESSING
# ============================================================

print()
print("Running preprocessing...")


processed_data = preprocess_input(
    test_order
)


print(
    "✓ Preprocessing completed"
)


# ============================================================
# DISPLAY SHAPE
# ============================================================

print()
print("Processed dataframe shape:")

print(
    processed_data.shape
)


# ============================================================
# TEST FEATURE COUNT
# ============================================================

EXPECTED_FEATURE_COUNT = 25

assert (
    processed_data.shape[1]
    == EXPECTED_FEATURE_COUNT
), (
    f"Expected {EXPECTED_FEATURE_COUNT} features, "
    f"received {processed_data.shape[1]}"
)


print(
    f"✓ Correct number of features: "
    f"{EXPECTED_FEATURE_COUNT}"
)


# ============================================================
# DISPLAY FEATURES
# ============================================================

print()
print("Final model features:")

for i, column in enumerate(
    processed_data.columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


# ============================================================
# TEST FEATURE ORDER
# ============================================================

assert (
    processed_data.columns.tolist()
    == list(feature_columns)
), (
    "\nFeature order mismatch.\n"
    f"Expected:\n{list(feature_columns)}\n\n"
    f"Received:\n"
    f"{processed_data.columns.tolist()}"
)


print()
print(
    "✓ Feature order matches training"
)


# ============================================================
# TEST CATEGORICAL COLUMNS
# ============================================================

print()
print("Checking categorical columns...")


for column in MODEL_CATEGORICAL_COLUMNS:

    # --------------------------------------------------------
    # Column exists
    # --------------------------------------------------------

    assert column in processed_data.columns, (
        f"Categorical column missing: {column}"
    )


    # --------------------------------------------------------
    # Correct dtype
    # --------------------------------------------------------

    assert (
        str(processed_data[column].dtype)
        == "category"
    ), (
        f"Incorrect dtype for "
        f"categorical column: {column}"
    )


print(
    "✓ All categorical columns have "
    "correct 'category' dtype"
)


# ============================================================
# TEST THAT REMOVED FEATURES ARE NOT PRESENT
# ============================================================

print()
print("Checking removed features...")


REMOVED_FEATURES = [
    "Order_Item_Discount_Rate",
    "Order_Item_Profit_Ratio"
]


for column in REMOVED_FEATURES:

    assert column not in processed_data.columns, (
        f"Removed feature is still present: {column}"
    )


print(
    "✓ Order Item Discount Rate removed"
)

print(
    "✓ Order Item Profit Ratio removed"
)


# ============================================================
# TEST DATE FEATURES
# ============================================================

print()
print("Checking date feature engineering...")


assert (
    processed_data["Order_Year"].iloc[0]
    == 2017
)

assert (
    processed_data["Order_Month"].iloc[0]
    == 7
)

assert (
    processed_data["Order_Day"].iloc[0]
    == 15
)


print(
    "✓ Order Year correctly generated"
)

print(
    "✓ Order Month correctly generated"
)

print(
    "✓ Order Day correctly generated"
)


# ============================================================
# TEST WEEKEND FEATURE
# ============================================================

print()
print("Checking weekend feature...")


assert (
    processed_data[
        "is_weekend_order"
    ].iloc[0]
    == 1
)


print(
    "✓ Saturday correctly identified "
    "as weekend"
)


# ============================================================
# TEST CROSS-BORDER FEATURE
# ============================================================

print()
print("Checking cross-border feature...")


# Customer Country = Puerto Rico
# Order Country    = Puerto Rico
#
# Therefore:
# is_cross_border = 0

assert (
    processed_data[
        "is_cross_border"
    ].iloc[0]
    == 0
)


print(
    "✓ Same origin/destination country "
    "correctly identified as non-cross-border"
)


# ============================================================
# RUN MODEL
# ============================================================

print()
print("Running LightGBM prediction...")


result = predict(
    test_order
)


print(
    "✓ Prediction completed"
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("MODEL PREDICTION")
print("=" * 60)


print()

print(
    f"Prediction: "
    f"{result['prediction']}"
)


print(
    f"Probability of late delivery: "
    f"{result['probability']:.4f}"
)


# ============================================================
# HUMAN-READABLE RESULT
# ============================================================

print()


if result["prediction"] == 1:

    print(
        "Prediction: "
        "HIGH RISK OF LATE DELIVERY"
    )

else:

    print(
        "Prediction: "
        "LOW RISK OF LATE DELIVERY"
    )


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print()
print("=" * 60)
print(
    "✓ INFERENCE PIPELINE TEST PASSED"
)
print("=" * 60)

