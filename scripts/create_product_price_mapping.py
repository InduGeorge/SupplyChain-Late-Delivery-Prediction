import os
import joblib
import pandas as pd

# ============================================================
# PROJECT PATH
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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "product_price_mapping.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

# ============================================================
# CHECK PRODUCT → PRICE CONSISTENCY
# ============================================================

price_counts = (
    df.groupby("Product Name")["Product Price"]
    .nunique()
)

multiple_prices = price_counts[
    price_counts > 1
]

print(
    f"\nTotal products: "
    f"{df['Product Name'].nunique()}"
)

print(
    f"Products with multiple prices: "
    f"{len(multiple_prices)}"
)

if len(multiple_prices) > 0:

    print(
        "\nWARNING: Some products have multiple prices."
    )

    print(
        multiple_prices.to_string()
    )

    raise ValueError(
        "Product price mapping cannot be created "
        "because some products have multiple prices."
    )

# ============================================================
# CREATE MAPPING
# ============================================================

product_price_mapping = (
    df.groupby("Product Name")["Product Price"]
    .first()
    .to_dict()
)

# ============================================================
# SAVE
# ============================================================

joblib.dump(
    product_price_mapping,
    OUTPUT_PATH
)

print("\n" + "=" * 60)
print("PRODUCT PRICE MAPPING CREATED")
print("=" * 60)

print(
    f"\nNumber of products mapped: "
    f"{len(product_price_mapping)}"
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\nSample mappings:")

for product, price in list(
    product_price_mapping.items()
)[:10]:

    print(
        f"{product} → {price}"
    )

print(
    "\n✓ Product Name → Product Price mapping "
    "created successfully."
)