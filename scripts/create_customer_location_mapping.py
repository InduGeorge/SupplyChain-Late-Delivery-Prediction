# ============================================================
# CREATE CUSTOMER CITY → CUSTOMER COUNTRY MAPPING
# Supply Chain Late Delivery Prediction
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "DataCoSupplyChainDataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_PATH = os.path.join(
    MODEL_DIR,
    "customer_location_mapping.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Customer City",
    "Customer Country"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

location_df = df[
    required_columns
].dropna().copy()


# ============================================================
# CHECK CITY → COUNTRY CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("CHECKING CUSTOMER LOCATION CONSISTENCY")
print("=" * 60)

city_country_counts = (
    location_df
    .groupby("Customer City")["Customer Country"]
    .nunique()
)

ambiguous_cities = (
    city_country_counts[
        city_country_counts > 1
    ]
)

print(
    f"\nNumber of unique Customer Cities: "
    f"{location_df['Customer City'].nunique()}"
)

print(
    f"Cities mapped to multiple countries: "
    f"{len(ambiguous_cities)}"
)


# ============================================================
# DISPLAY AMBIGUOUS CITIES
# ============================================================

if len(ambiguous_cities) > 0:

    print(
        "\nWARNING: Some Customer Cities map "
        "to multiple countries."
    )

    print(
        ambiguous_cities.to_frame(
            name="country_count"
        ).head(30)
    )


# ============================================================
# CREATE MAPPING
# ============================================================

customer_location_mapping = {}

for city, group in location_df.groupby(
    "Customer City"
):

    countries = (
        group["Customer Country"]
        .dropna()
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # If only one country exists, store it directly
    # --------------------------------------------------------

    if len(countries) == 1:

        customer_location_mapping[
            city
        ] = countries[0]

    # --------------------------------------------------------
    # If multiple countries exist, store all possibilities
    # --------------------------------------------------------

    else:

        customer_location_mapping[
            city
        ] = countries


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MAPPING
# ============================================================

joblib.dump(
    customer_location_mapping,
    OUTPUT_PATH
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER LOCATION MAPPING CREATED")
print("=" * 60)

print(
    f"\nNumber of mapped cities: "
    f"{len(customer_location_mapping)}"
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)


# ============================================================
# SAMPLE MAPPINGS
# ============================================================

print("\nSample mappings:")

for city, country in list(
    customer_location_mapping.items()
)[:10]:

    print(
        f"{city} → {country}"
    )


print(
    "\n✓ Customer city → country mapping "
    "created successfully."
)
