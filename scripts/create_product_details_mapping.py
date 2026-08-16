# ============================================================
# CREATE PRODUCT DETAILS MAPPING
# Product Name → Category Name + Department Name
# ============================================================

import os
import sys
import joblib
import pandas as pd


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

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "DataCoSupplyChainDataset.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_PATH = os.path.join(
    MODEL_DIR,
    "product_details_mapping.pkl"
)


# ============================================================
# 3. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print()
print("Dataset shape:")
print(df.shape)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Product Name",
    "Category Name",
    "Department Name"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"""
The following required columns are missing:

{missing_columns}
"""
    )


# ============================================================
# 6. CHECK PRODUCT CONSISTENCY
# ============================================================

print()
print("=" * 60)
print("CHECKING PRODUCT CONSISTENCY")
print("=" * 60)


check = (
    df.groupby("Product Name")[
        [
            "Category Name",
            "Department Name"
        ]
    ]
    .nunique()
)


inconsistent = check[
    (check["Category Name"] > 1) |
    (check["Department Name"] > 1)
]


print()
print(
    "Number of unique products:",
    df["Product Name"].nunique()
)

print(
    "Products with inconsistent mapping:",
    len(inconsistent)
)


# ============================================================
# 7. STOP IF INCONSISTENCIES EXIST
# ============================================================

if len(inconsistent) > 0:

    print()
    print(
        "WARNING: Some products have "
        "multiple Category/Department values."
    )

    print()
    print(inconsistent)

    raise ValueError(
        """
Product → Category/Department mapping is not unique.

Artifact was NOT created.
"""
    )


print()
print(
    "✓ Every product has exactly one "
    "Category Name and Department Name."
)


# ============================================================
# 8. CREATE MAPPING
# ============================================================

print()
print("=" * 60)
print("CREATING PRODUCT DETAILS MAPPING")
print("=" * 60)


product_details_mapping = {}


for _, row in (
    df[
        [
            "Product Name",
            "Category Name",
            "Department Name"
        ]
    ]
    .drop_duplicates()
    .iterrows()
):

    product_name = row[
        "Product Name"
    ]

    category_name = row[
        "Category Name"
    ]

    department_name = row[
        "Department Name"
    ]


    product_details_mapping[
        product_name
    ] = {

        "Category Name":
            category_name,

        "Department Name":
            department_name
    }


# ============================================================
# 9. SAVE ARTIFACT
# ============================================================

joblib.dump(
    product_details_mapping,
    OUTPUT_PATH
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("PRODUCT DETAILS MAPPING CREATED")
print("=" * 60)

print()
print(
    "Number of mapped products:",
    len(product_details_mapping)
)

print()
print(
    "Saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 11. DISPLAY SAMPLE MAPPINGS
# ============================================================

print()
print("Sample mappings:")

sample_count = 10

for i, (
    product,
    details
) in enumerate(
    product_details_mapping.items()
):

    if i >= sample_count:
        break

    print(
        f"{product} → "
        f"{details['Category Name']} → "
        f"{details['Department Name']}"
    )


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print()
print(
    "✓ Product Name → Category Name → "
    "Department Name mapping created successfully."
)

