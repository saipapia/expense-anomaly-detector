import streamlit as st
import pandas as pd
import plotly.express as px
from detector import detect_anomalies_zscore_robust, detect_anomalies_isolation_forest
from utils import convert_df, prepare_trend

st.set_page_config(page_title="Expense Anomaly Detector", layout="wide", page_icon="💸")
st.title("💸 AI-Powered Expense Anomaly Detector")

uploaded_file = st.file_uploader("Upload your expense CSV", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        st.error("Missing 'Date' column in your CSV.")
        st.stop()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    return df

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        required_columns = {"Date", "Amount", "Category", "Payment Method"}
        if not required_columns.issubset(df.columns):
            st.error(f"Missing required columns: {', '.join(required_columns)}")
            st.stop()

        # Sidebar filters
        with st.sidebar:
            st.markdown("## 📅 Filter by Date Range")
            min_date, max_date = df["Date"].min(), df["Date"].max()
            start_date = st.date_input("From Date", min_value=min_date, max_value=max_date, value=min_date)
            end_date = st.date_input("To Date", min_value=min_date, max_value=max_date, value=max_date)
            if start_date > end_date:
                st.warning("⚠️ 'From Date' cannot be after 'To Date'")
                st.stop()

            st.markdown("## 🏷️ Filter by Category")
            categories = df["Category"].dropna().unique().tolist()
            selected_categories = st.multiselect("Choose Categories", options=categories, default=categories)

            st.markdown("## ⚙️ Anomaly Threshold (Z-Score) 🎯")
            zscore_threshold = st.slider(
                "Anomaly Sensitivity (Z-Score Threshold)",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Lower value = more sensitive (more anomalies); Higher value = less sensitive"
            )
            st.session_state['threshold'] = zscore_threshold  # Track the current threshold
            st.sidebar.write(f"Current Threshold: {st.session_state['threshold']}")  # Optional debug display

        # Apply filters
        df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
        df_filtered = df_filtered[df_filtered["Category"].isin(selected_categories)]

        if df_filtered.empty:
            st.error("No data after applying filters.")
            st.stop()

        # Detect anomalies
        df_z = detect_anomalies_zscore_robust(df_filtered.copy(), threshold=zscore_threshold)
        df_if = detect_anomalies_isolation_forest(df_filtered.copy())

        # Recalculate trend with updated anomalies
        trend_z = prepare_trend(df_z, "Anomaly_Z")
        trend_if = prepare_trend(df_if, "Anomaly_IF")

        st.write(f"Z-Score anomalies detected: {df_z['Anomaly_Z'].sum()}")
        st.write(df_z[df_z["Anomaly_Z"]][["Date", "Amount", "Robust_Z_Score"]]
                 .sort_values(by="Robust_Z_Score", ascending=False).head())

        tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Detection", "📊 Insights", "📂 Download"])

        with tab1:
            st.subheader("🔍 Anomaly Detection")
            st.markdown("- Z-Score: Based on deviation from median.")
            st.markdown("- Isolation Forest: Machine-learning-based anomaly detection.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Z-Score Detected Anomalies")
                fig_z = px.line(trend_z, x="Date", y="Amount", title="Z-Score - Daily Expenses", markers=True)
                fig_z.add_scatter(
                    x=trend_z[trend_z["Is_Anomaly"]]["Date"],
                    y=trend_z[trend_z["Is_Anomaly"]]["Amount"],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Anomaly'
                )
                st.plotly_chart(fig_z, use_container_width=True)
                st.markdown("#### Z-Score Anomaly Details")
                st.dataframe(df_z[df_z["Anomaly_Z"]].sort_values(by="Date"))

            with col2:
                st.markdown("### Isolation Forest Detected Anomalies")
                fig_if = px.line(trend_if, x="Date", y="Amount", title="Isolation Forest - Daily Expenses", markers=True)
                fig_if.add_scatter(
                    x=trend_if[trend_if["Is_Anomaly"]]["Date"],
                    y=trend_if[trend_if["Is_Anomaly"]]["Amount"],
                    mode='markers',
                    marker=dict(color='orange', size=10),
                    name='Anomaly'
                )
                st.plotly_chart(fig_if, use_container_width=True)
                st.markdown("#### Isolation Forest Anomaly Details")
                st.dataframe(df_if[df_if["Anomaly_IF"]].sort_values(by="Date"))

        with tab2:
            st.subheader("📊 Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Spending by Category")
                pie1 = df_filtered.groupby("Category")["Amount"].sum().reset_index()
                fig_cat = px.pie(pie1, names="Category", values="Amount")
                st.plotly_chart(fig_cat, use_container_width=True)

            with col2:
                st.markdown("### Spending by Payment Method")
                pie2 = df_filtered.groupby("Payment Method")["Amount"].sum().reset_index()
                fig_pay = px.pie(pie2, names="Payment Method", values="Amount")
                st.plotly_chart(fig_pay, use_container_width=True)

            st.subheader("🏆 Top 5 Highest Expenses")
            top5 = df_filtered.nlargest(5, "Amount").copy()
            top5["Z_Anomaly"] = df_z.loc[top5.index, "Anomaly_Z"]
            top5["Z_Score"] = df_z.loc[top5.index, "Robust_Z_Score"]
            st.dataframe(top5[["Date", "Category", "Payment Method", "Amount", "Z_Anomaly", "Z_Score"]])

        with tab3:
            st.subheader("📂 Download Data")
            df_filtered["Anomaly_Z"] = df_z["Anomaly_Z"]
            df_filtered["Anomaly_IF"] = df_if["Anomaly_IF"]

            st.download_button(
                "Download Z-Score Anomalies CSV",
                convert_df(df_z[df_z["Anomaly_Z"]].reset_index(drop=True)),
                file_name="anomalies_zscore.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Full Dataset CSV",
                convert_df(df_filtered.reset_index(drop=True)),
                file_name="full_dataset.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to get started.")
