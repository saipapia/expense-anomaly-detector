import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation


def detect_anomalies_zscore_robust(df, threshold=3.0):
    df = df.copy()

    # Calculate robust z-score using MAD
    median = df["Amount"].median()
    mad = median_abs_deviation(df["Amount"], scale='normal')
    df["Robust_Z_Score"] = (df["Amount"] - median) / mad

    # Detect anomalies based on threshold
    df["Anomaly_Z"] = df["Robust_Z_Score"].abs() > threshold

    return df

def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.4) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.
    Encodes 'Category' and uses 'Amount' and difference from category median.
    Args:
        df: Input DataFrame with 'Amount' and 'Category'.
        contamination: expected proportion of outliers in the data.
    Returns:
        DataFrame with 'Anomaly_IF' column added.
    """
    df = df.copy()

    # One-hot encode Category
    category_encoded = pd.get_dummies(df['Category'], prefix='Category')

    # Feature: Amount_diff_from_cat_median
    df['Amount_diff_from_cat_median'] = df['Amount'] - df.groupby('Category')['Amount'].transform('median')

    # Prepare features
    features = pd.concat([df[['Amount', 'Amount_diff_from_cat_median']], category_encoded], axis=1)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(features_scaled)

    df['Anomaly_IF'] = preds == -1
    print(f"detect_anomalies_isolation_forest detected {df['Anomaly_IF'].sum()} anomalies")  # Debug
    return df
