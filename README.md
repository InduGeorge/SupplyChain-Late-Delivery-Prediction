# Supply Chain Late Delivery Prediction

## Overview

This project predicts whether a supply-chain order is at risk of late delivery using machine learning. It includes EDA, preprocessing, feature engineering, model training/evaluation, and an interactive Streamlit dashboard.
Project Structure
```text
SupplyChain_LateDelivery_Prediction/
├── app/
│   └── app.py
├── models/
│   └── trained model and preprocessing artifacts
├── notebooks/
│   └── EDA and modelling notebooks
├── scripts/
│   └── artifact-generation scripts
├── src/
│   └── preprocessing.py
├── tests/
│   └── test_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Installation
1. Open the project folder
Open PowerShell in the project root:
```powershell
cd "C:\Users\indup\Desktop\IIT Palakkad Advanced AI\Final Project\SupplyChain_LateDelivery_Prediction"
```
2. Create a virtual environment
```powershell
python -m venv venv
```
3. Activate it
```powershell
.\venv\Scripts\Activate.ps1
```
4. Install dependencies
```powershell
pip install -r requirements.txt
```
## Model Artifacts
The `models` folder contains the trained model and supporting artifacts required for inference.
The current model uses 25 features.
The following two original dataset columns were removed from the modelling pipeline:
`Order Item Profit Ratio`
`Order Item Discount Rate`
Important engineered features include:
`Order_Year`
`Order_Month`
`Order_Day`
`is_weekend_order`
`is_cross_border`
## Test the Inference Pipeline
Run the complete preprocessing and prediction test before launching the dashboard:
```powershell
python tests\test_prediction.py
```
The test verifies the feature count, feature order, categorical columns, date engineering, engineered features, and model prediction.
## Run the Streamlit Dashboard
From the project root:
```powershell
streamlit run app\app.py
```
Open the URL shown by Streamlit, normally:
```text
http://localhost:8501
```
## Dashboard Functionality
The dashboard allows users to enter order information and predict late-delivery risk.
Order Information
Order Type
Shipping Mode
Scheduled Shipping Days
Order Date
Customer Information
Customer City
Customer Country
Customer Segment
Customer Country is automatically populated from the customer-location mapping.
Product Information
Product Name
Product Price
Category Name
Department Name
Order Quantity
Product Price, Category Name, and Department Name are automatically populated based on the selected Product Name.
Order Location
Order State
Order Country
Order Region
Market
Location information is automatically populated using the location mapping artifact.
Financial Information
Sales is automatically calculated using:
```text
Sales = Product Price × Order Item Quantity
```
Order Item Total is automatically calculated using:
```text
Order Item Total = Sales − Order Item Discount
```
Money values are displayed with a dollar sign in the dashboard.
Prediction Output
The dashboard displays:
Late-delivery prediction
Probability of late delivery
Risk status
Order input summary
Prediction interpretation:
```text
0 = Low Risk / On Time
1 = High Risk / Late Delivery
```
End-to-End Workflow
```text
Raw Dataset
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation & Tuning
     ↓
Model Artifacts
     ↓
Inference Pipeline
     ↓
Streamlit Dashboard
     ↓
Late Delivery Prediction
```
## Troubleshooting
Git is not recognized
Check Git:
```powershell
git --version
```
If unavailable, install Git and enable the option to add Git to PATH, then restart PowerShell.

Streamlit is not recognized

Activate the virtual environment and install Streamlit:
```powershell
pip install streamlit
```
Then run:
```powershell
streamlit run app\app.py
```

Model feature mismatch

If the model reports a feature-count or feature-order mismatch, make sure the model artifacts and `feature_columns.pkl` were generated from the same training run.
Then run:
```powershell
python tests\test_prediction.py
```
Scikit-learn version warning

If an `InconsistentVersionWarning` appears, the saved preprocessing artifact was created with a different scikit-learn version. Using the same package versions used during training is recommended.

Reproducibility

For reliable predictions:
Keep the model and preprocessing artifacts from the same training run.
Do not manually change the feature order.
Keep preprocessing and inference logic synchronized.
Use the versions specified in `requirements.txt`.
Run `tests/test_prediction.py` after changing preprocessing or retraining the model.

Final Application

The completed application provides an interactive machine-learning based decision-support dashboard for identifying orders with a higher risk of late delivery.
