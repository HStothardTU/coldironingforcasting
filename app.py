
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from io import BytesIO

st.title(" Cold Ironing Demand Forecasting")

st.sidebar.header("Options")
data_source = st.sidebar.radio("Choose data source:", ("Upload CSV", "Generate Synthetic Data"))

def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=180, freq='D')
    vessel_counts = np.random.poisson(lam=5, size=len(dates))
    avg_power_kw = np.random.normal(loc=500, scale=50, size=len(dates))
    df = pd.DataFrame({
        "date": dates,
        "vessel_count": vessel_counts,
        "avg_power_kw": avg_power_kw
    })
    df["daily_demand_kw"] = df["vessel_count"] * df["avg_power_kw"]
    return df

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        if "daily_demand_kw" not in df.columns:
            df["daily_demand_kw"] = df["vessel_count"] * df["avg_power_kw"]
else:
    df = generate_synthetic_data()

if 'df' in locals():
    st.subheader("Input Data")
    st.write(df.head())

    prophet_df = df[["date", "daily_demand_kw"]].rename(columns={"date": "ds", "daily_demand_kw": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.subheader("Forecast")
    fig = model.plot(forecast)
    st.pyplot(fig)

    forecast_download = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    csv = forecast_download.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Forecast CSV", data=csv, file_name="cold_ironing_forecast.csv", mime="text/csv")
