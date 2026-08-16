# ============================================================
# STREAMLIT APPLICATION
# Supply Chain Late Delivery Prediction
# ============================================================

import streamlit as st
from datetime import date
import sys
import os
import joblib


# ============================================================
# 1. PROJECT ROOT
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
# 2. PATHS
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)


# ============================================================
# 3. LOAD PREPROCESSING / MODEL
# ============================================================

from src.preprocessing import predict


# ============================================================
# 4. LOAD ORDER LOCATION MAPPING
# ============================================================

LOCATION_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "location_mapping.pkl"
)

if not os.path.exists(LOCATION_MAPPING_PATH):

    raise FileNotFoundError(
        f"""
Order location mapping not found.

Expected file:
{LOCATION_MAPPING_PATH}
"""
    )


location_mapping = joblib.load(
    LOCATION_MAPPING_PATH
)


# ============================================================
# 5. LOAD CUSTOMER LOCATION MAPPING
# ============================================================

CUSTOMER_LOCATION_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "customer_location_mapping.pkl"
)

if not os.path.exists(
    CUSTOMER_LOCATION_MAPPING_PATH
):

    raise FileNotFoundError(
        f"""
Customer location mapping not found.

Expected file:
{CUSTOMER_LOCATION_MAPPING_PATH}
"""
    )


customer_location_mapping = joblib.load(
    CUSTOMER_LOCATION_MAPPING_PATH
)


# ============================================================
# 6. LOAD PRODUCT PRICE MAPPING
# ============================================================

PRODUCT_PRICE_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "product_price_mapping.pkl"
)

if not os.path.exists(
    PRODUCT_PRICE_MAPPING_PATH
):

    raise FileNotFoundError(
        f"""
Product price mapping not found.

Expected file:
{PRODUCT_PRICE_MAPPING_PATH}
"""
    )


product_price_mapping = joblib.load(
    PRODUCT_PRICE_MAPPING_PATH
)


# ============================================================
# 7. LOAD PRODUCT DETAILS MAPPING
#
# Product Name
#       ↓
# Category Name
# Department Name
# ============================================================

PRODUCT_DETAILS_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "product_details_mapping.pkl"
)


if not os.path.exists(
    PRODUCT_DETAILS_MAPPING_PATH
):

    raise FileNotFoundError(
        f"""
Product details mapping not found.

Expected file:
{PRODUCT_DETAILS_MAPPING_PATH}

Create the artifact using:

python scripts\\create_product_details_mapping.py
"""
    )


product_details_mapping = joblib.load(
    PRODUCT_DETAILS_MAPPING_PATH
)


# ============================================================
# 8. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Late Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# 9. TITLE
# ============================================================

st.title(
    "🚚 Supply Chain Late Delivery Prediction"
)

st.write(
    "Enter the order details below to predict the "
    "risk of late delivery."
)


# ============================================================
# 10. ORDER INFORMATION
# ============================================================

st.header("📦 Order Information")


col1, col2, col3 = st.columns(3)


with col1:

    order_type = st.selectbox(
        "Order Type",
        [
            "CASH",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )


with col2:

    shipping_mode = st.selectbox(
        "Shipping Mode",
        [
            "Standard Class",
            "Second Class",
            "First Class",
            "Same Day"
        ]
    )


with col3:

    scheduled_days = st.number_input(
        "Scheduled Shipping Days",
        min_value=0,
        max_value=30,
        value=4,
        step=1
    )


# ============================================================
# 11. ORDER DATE
# ============================================================

st.header("📅 Order Date")


order_date = st.date_input(
    "Order Date",
    value=date(2017, 7, 15)
)


# ============================================================
# 12. CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")


customer_cities = sorted(
    customer_location_mapping.keys()
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# CUSTOMER CITY
# ------------------------------------------------------------

with col1:

    customer_city = st.selectbox(
        "Customer City",
        customer_cities
    )


# ------------------------------------------------------------
# CUSTOMER COUNTRY - AUTO FILLED
# ------------------------------------------------------------

with col2:

    customer_country = (
        customer_location_mapping[
            customer_city
        ]
    )

    st.text_input(
        "Customer Country",
        value=customer_country,
        disabled=True
    )


# ------------------------------------------------------------
# CUSTOMER SEGMENT
# ------------------------------------------------------------

with col3:

    customer_segment = st.selectbox(
        "Customer Segment",
        [
            "Consumer",
            "Corporate",
            "Home Office"
        ]
    )


# ============================================================
# 13. PRODUCT INFORMATION
# ============================================================

st.header("🛍️ Product Information")


product_names = sorted(
    product_price_mapping.keys()
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# PRODUCT NAME
# ------------------------------------------------------------

with col1:

    product_name = st.selectbox(
        "Product Name",
        product_names
    )


# ------------------------------------------------------------
# GET PRODUCT DETAILS
# ------------------------------------------------------------

product_price = float(
    product_price_mapping[
        product_name
    ]
)


product_details = (
    product_details_mapping[
        product_name
    ]
)


category_name = (
    product_details[
        "Category Name"
    ]
)


department_name = (
    product_details[
        "Department Name"
    ]
)


# ------------------------------------------------------------
# PRODUCT PRICE - AUTO FILLED
# ------------------------------------------------------------

with col2:

    st.text_input(
        "Product Price",
        value=f"${product_price:,.2f}",
        disabled=True
    )


# ------------------------------------------------------------
# QUANTITY
# ------------------------------------------------------------

with col3:

    quantity = st.number_input(
        "Order Item Quantity",
        min_value=1,
        value=1,
        step=1
    )


# ============================================================
# 14. CATEGORY / DEPARTMENT - AUTO FILLED
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.text_input(
        "Category Name",
        value=category_name,
        disabled=True
    )


with col2:

    st.text_input(
        "Department Name",
        value=department_name,
        disabled=True
    )


# ============================================================
# 15. ORDER LOCATION
# ============================================================

st.header("🌍 Order Location")


# ------------------------------------------------------------
# ORDER STATE
# ------------------------------------------------------------

order_states = sorted(
    location_mapping.keys()
)


selected_order_state = st.selectbox(
    "Order State",
    order_states
)


# ------------------------------------------------------------
# GET LOCATION
# ------------------------------------------------------------

selected_location = (
    location_mapping[
        selected_order_state
    ]
)


# ============================================================
# 15.1 NORMAL STATE
# ============================================================

if isinstance(
    selected_location,
    dict
):

    order_country = (
        selected_location[
            "Order Country"
        ]
    )

    order_region = (
        selected_location[
            "Order Region"
        ]
    )

    market = (
        selected_location.get(
            "Market",
            ""
        )
    )


# ============================================================
# 15.2 AMBIGUOUS STATE
# ============================================================

elif isinstance(
    selected_location,
    list
):

    location_options = []


    for item in selected_location:

        option = (
            f"{item['Order Country']} | "
            f"{item['Order Region']} | "
            f"{item.get('Market', '')}"
        )

        location_options.append(
            option
        )


    selected_option = st.selectbox(
        "Select Location",
        location_options
    )


    selected_index = (
        location_options.index(
            selected_option
        )
    )


    selected_mapping = (
        selected_location[
            selected_index
        ]
    )


    order_country = (
        selected_mapping[
            "Order Country"
        ]
    )


    order_region = (
        selected_mapping[
            "Order Region"
        ]
    )


    market = (
        selected_mapping.get(
            "Market",
            ""
        )
    )


# ============================================================
# 15.3 DISPLAY AUTO-FILLED LOCATION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.text_input(
        "Order Country",
        value=order_country,
        disabled=True
    )


with col2:

    st.text_input(
        "Order Region",
        value=order_region,
        disabled=True
    )


with col3:

    st.text_input(
        "Market",
        value=market,
        disabled=True
    )


# ============================================================
# 16. ORDER STATUS
# ============================================================

st.header("📋 Order Status")


order_status = st.selectbox(
    "Order Status",
    [
        "COMPLETE",
        "PENDING",
        "PROCESSING",
        "CANCELED",
        "SUSPECTED_FRAUD"
    ]
)


# ============================================================
# 17. ORDER FINANCIAL INFORMATION
# ============================================================

st.header("💰 Order Financial Information")


# ============================================================
# SALES - AUTO FILLED
#
# Sales = Product Price × Quantity
# ============================================================

sales = (
    product_price *
    quantity
)


st.text_input(
    "Sales",
    value=f"${sales:,.2f}",
    disabled=True
)


# ============================================================
# ORDER ITEM DISCOUNT
# ============================================================

order_item_discount = st.number_input(
    "Order Item Discount ($)",
    min_value=0.0,
    value=20.0,
    step=1.0,
    format="%.2f"
)


# ============================================================
# ORDER ITEM TOTAL - AUTO FILLED
#
# Order Item Total = Sales − Discount
# ============================================================

order_item_total = (
    sales -
    order_item_discount
)


st.text_input(
    "Order Item Total",
    value=f"${order_item_total:,.2f}",
    disabled=True
)


# ============================================================
# ORDER PROFIT PER ORDER
# ============================================================

order_profit_per_order = st.number_input(
    "Order Profit Per Order ($)",
    value=60.0,
    step=1.0,
    format="%.2f"
)


# ============================================================
# 18. PREDICTION BUTTON
# ============================================================

st.divider()


predict_button = st.button(
    "🔮 Predict Late Delivery Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# 19. PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # CREATE RAW INPUT
    # ========================================================

    input_data = {

        # ----------------------------------------------------
        # ORDER
        # ----------------------------------------------------

        "Type":
            order_type,

        "Days for shipment (scheduled)":
            scheduled_days,

        "Shipping Mode":
            shipping_mode,


        # ----------------------------------------------------
        # CUSTOMER
        # ----------------------------------------------------

        "Customer City":
            customer_city,

        "Customer Country":
            customer_country,

        "Customer Segment":
            customer_segment,


        # ----------------------------------------------------
        # PRODUCT
        # ----------------------------------------------------

        "Category Name":
            category_name,

        "Department Name":
            department_name,

        "Product Name":
            product_name,

        "Product Price":
            product_price,

        "Order Item Quantity":
            quantity,


        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        "Market":
            market,

        "Order Country":
            order_country,

        "Order Region":
            order_region,

        "Order State":
            selected_order_state,


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        "Order Status":
            order_status,


        # ----------------------------------------------------
        # FINANCIAL
        #
        # IMPORTANT:
        # Order Item Discount Rate is NOT included.
        # Order Item Profit Ratio is NOT included.
        # ----------------------------------------------------

        "Sales":
            sales,

        "Order Item Discount":
            order_item_discount,

        "Order Item Total":
            order_item_total,

        "Order Profit Per Order":
            order_profit_per_order,


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        "order date (DateOrders)":
            order_date.strftime(
                "%Y-%m-%d"
            )
    }


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    try:

        result = predict(
            input_data
        )


        prediction = result[
            "prediction"
        ]


        probability = result[
            "probability"
        ]


        probability_percent = (
            probability * 100
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()


        st.header(
            "📊 Prediction Result"
        )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ HIGH RISK OF LATE DELIVERY"
            )

        else:

            st.success(
                "✅ LOW RISK OF LATE DELIVERY"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.metric(
            label="Probability of Late Delivery",
            value=(
                f"{probability_percent:.2f}%"
            )
        )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.progress(
            float(probability)
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander(
            "🔍 View Input Summary"
        ):

            # ------------------------------------------------
            # ORDER
            # ------------------------------------------------

            st.write(
                "### 📦 Order"
            )

            st.write(
                f"**Order Type:** "
                f"{order_type}"
            )

            st.write(
                f"**Shipping Mode:** "
                f"{shipping_mode}"
            )

            st.write(
                f"**Scheduled Shipping Days:** "
                f"{scheduled_days}"
            )

            st.write(
                f"**Order Date:** "
                f"{order_date}"
            )


            # ------------------------------------------------
            # CUSTOMER
            # ------------------------------------------------

            st.write(
                "### 👤 Customer"
            )

            st.write(
                f"**Customer City:** "
                f"{customer_city}"
            )

            st.write(
                f"**Customer Country:** "
                f"{customer_country}"
            )

            st.write(
                f"**Customer Segment:** "
                f"{customer_segment}"
            )


            # ------------------------------------------------
            # PRODUCT
            # ------------------------------------------------

            st.write(
                "### 🛍️ Product"
            )

            st.write(
                f"**Product:** "
                f"{product_name}"
            )

            st.write(
                f"**Category:** "
                f"{category_name}"
            )

            st.write(
                f"**Department:** "
                f"{department_name}"
            )

            st.write(
                f"**Product Price:** "
                f"${product_price:,.2f}"
            )

            st.write(
                f"**Quantity:** "
                f"{quantity}"
            )


            # ------------------------------------------------
            # LOCATION
            # ------------------------------------------------

            st.write(
                "### 🌍 Location"
            )

            st.write(
                f"**Order State:** "
                f"{selected_order_state}"
            )

            st.write(
                f"**Order Country:** "
                f"{order_country}"
            )

            st.write(
                f"**Order Region:** "
                f"{order_region}"
            )

            st.write(
                f"**Market:** "
                f"{market}"
            )


            # ------------------------------------------------
            # FINANCIAL
            # ------------------------------------------------

            st.write(
                "### 💰 Financial Information"
            )

            st.write(
                f"**Sales:** "
                f"${sales:,.2f}"
            )

            st.write(
                f"**Order Item Discount:** "
                f"${order_item_discount:,.2f}"
            )

            st.write(
                f"**Order Item Total:** "
                f"${order_item_total:,.2f}"
            )

            st.write(
                f"**Profit Per Order:** "
                f"${order_profit_per_order:,.2f}"
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(e)

        