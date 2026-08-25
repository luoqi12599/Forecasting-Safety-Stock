"""
Streamlit demo:
- 上传或使用 sample_sales.csv
- 选择 SKU & horizon
- 计算并展示预测（简单指数平滑）与安全库存（可调整服务水平与提前期）
"""
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.safety_stock import safety_stock_from_demand_std, safety_stock_from_forecast_errors

st.set_page_config(page_title="Forecast & Safety Stock Demo", layout="wide")
st.title("Demand Forecasting & Safety Stock Demo")

uploaded = st.file_uploader("Upload sales CSV (date, sku, quantity) or use sample", type="csv")
if uploaded is None:
    df = pd.read_csv("data/sample_sales.csv", parse_dates=["date"])
else:
    df = pd.read_csv(uploaded, parse_dates=["date"])

sku_list = sorted(df["sku"].unique())
sku = st.selectbox("Select SKU", sku_list)
lead_time = st.number_input("Lead time (days)", min_value=1, max_value=180, value=14)
service_level = st.slider("Service level", 0.8, 0.999, 0.95, step=0.01)
horizon_weeks = st.slider("Forecast horizon (weeks)", 1, 16, 4)

df_sku = df[df["sku"] == sku].groupby("date")["quantity"].sum().asfreq("D").fillna(0)
daily = df_sku
weekly = daily.resample("W").sum()

st.subheader(f"Recent daily sales for {sku}")
st.line_chart(daily.tail(90))

# Build a simple weekly forecast
train = weekly[:-max(4, horizon_weeks)]
model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=52).fit()
pred = model.forecast(horizon_weeks)
st.subheader("Weekly forecast (next {} weeks)".format(horizon_weeks))
st.line_chart(pd.concat([weekly, pred], axis=0))

# Safety stock using daily std
avg_daily = daily.mean()
std_daily = daily.std(ddof=1)
ss = safety_stock_from_demand_std(avg_daily, std_daily, lead_time, service_level)
rop = avg_daily * lead_time + ss

st.metric("Avg daily demand", f"{avg_daily:.1f}")
st.metric("Safety stock", f"{ss:.1f}")
st.metric("Reorder point (ROP)", f"{rop:.1f}")

st.write("Export replenishment suggestion:")
if st.button("Export CSV"):
    out = pd.DataFrame([{
        "sku": sku,
        "avg_daily": float(avg_daily),
        "std_daily": float(std_daily),
        "lead_time_days": int(lead_time),
        "service_level": float(service_level),
        "safety_stock": float(ss),
        "reorder_point": float(rop)
    }])
    out.to_csv(f"data/replenishment_{sku}.csv", index=False)
    st.success(f"Saved data/replenishment_{sku}.csv")
