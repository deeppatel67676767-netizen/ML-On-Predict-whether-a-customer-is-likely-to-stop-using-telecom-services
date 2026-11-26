"""
Data preprocessing pipeline for Customer Churn Prediction.
Handles loading, cleaning, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV and return a cleaned DataFrame."""
    df = pd.read_csv(filepath)
    return clean_dataset(df)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataframe."""
    df = df.copy()

    # Drop customerID if present
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # TotalCharges may contain blank strings — coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Encode target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].mode()[0])

    # SeniorCitizen is already 0/1 in the IBM dataset — keep it
    return df


def get_feature_columns(df: pd.DataFrame):
    """Return categorical and numerical column lists that exist in df."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    return cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """Build a ColumnTransformer preprocessing pipeline."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Split features and target, build preprocessor, fit on train.
    Returns X_train, X_test, y_train, y_test, preprocessor, feature_cols.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    cat_cols, num_cols = get_feature_columns(df)
    feature_cols = cat_cols + num_cols

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_cols


def preprocess_single_input(input_dict: dict, preprocessor, feature_cols: list) -> np.ndarray:
    """
    Convert a single customer input dict to a preprocessed numpy array.
    Handles missing columns gracefully.
    """
    row = {}
    for col in feature_cols:
        row[col] = input_dict.get(col, np.nan)

    df_row = pd.DataFrame([row])

    # TotalCharges must be numeric
    if "TotalCharges" in df_row.columns:
        df_row["TotalCharges"] = pd.to_numeric(df_row["TotalCharges"], errors="coerce")

    return preprocessor.transform(df_row)
