import pandas as pd

def convert_df(df: pd.DataFrame) -> bytes:
    # Converts DataFrame to CSV bytes for download
    return df.to_csv(index=False).encode("utf-8")

def prepare_trend(df, anomaly_col):
    trend = df.groupby("Date")["Amount"].sum().reset_index()
    anomalies = df[df[anomaly_col]].groupby("Date")["Amount"].sum().reset_index()
    anomalies["Is_Anomaly"] = True
    trend = trend.merge(anomalies, on="Date", how="left", suffixes=("", "_Anomaly"))
    trend["Is_Anomaly"] = trend["Is_Anomaly"].fillna(False)
    return trend
