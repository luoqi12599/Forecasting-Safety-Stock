from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from src.safety_stock import safety_stock_from_forecast_errors, safety_stock_from_demand_std

ROOT = Path('.')
DATA_PATH = ROOT / 'data' / 'sample_sales.csv'
MODELS_DIR = ROOT / 'models'
XGB_MODELS_DIR = ROOT / 'models_xgb'

app = FastAPI(title="Forecast & Safety Stock API")

class PredictRequest(BaseModel):
    sku: str
    horizon_days: int = 14
    lead_time_days: int = 14
    service_level: float = 0.95

def load_series(sku):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    s = df[df["sku"]==sku].groupby("date")["quantity"].sum().asfreq("D").fillna(0)
    if s.empty:
        raise ValueError("SKU not found")
    return s

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        s = load_series(req.sku)
    except ValueError:
        raise HTTPException(status_code=404, detail="SKU not found")
    # try load XGB model first
    xgb_path = XGB_MODELS_DIR / f"{req.sku}_xgb.pkl"
    if xgb_path.exists():
        model = joblib.load(xgb_path)
        # simple feature construction for horizon: naive approach using last known values
        last = s[-28:]
        future_idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=req.horizon_days, freq="D")
        # construct features similarly to training script
        df_future = pd.DataFrame(index=future_idx)
        df_future["day_of_week"] = future_idx.dayofweek
        df_future["day"] = future_idx.day
        df_future["month"] = future_idx.month
        df_future["is_weekend"] = future_idx.dayofweek.isin([5,6]).astype(int)
        for lag in [1,7,14,28]:
            df_future[f"lag_{lag}"] = last.shift(lag-1).fillna(method="ffill").iloc[-1]
        for w in [7,14,28]:
            df_future[f"rolling_mean_{w}"] = s.shift(1).rolling(window=w, min_periods=1).mean().iloc[-1]
        preds = model.predict(df_future)
        preds = np.maximum(0, preds).tolist()
        method = "xgboost"
    else:
        # fallback to simple exponential smoothing weekly -> daily distribute
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        weekly = s.resample("W").sum()
        model = ExponentialSmoothing(weekly, seasonal="add", seasonal_periods=52).fit()
        # forecast weeks -> spread to daily by equal allocation
        weeks = int(np.ceil(req.horizon_days / 7))
        wf = model.forecast(weeks)
        # expand to daily
        daily_preds = []
        for wv in wf:
            per_day = float(wv) / 7.0
            daily_preds.extend([per_day]*7)
        preds = daily_preds[:req.horizon_days]
        method = "hw"

    # compute safety stock using forecast errors if weekly model available
    # compute a simple forecast error series using last training residuals if exist
    # Here we do a conservative estimate: use historical daily std
    avg_daily = float(s.mean())
    std_daily = float(s.std(ddof=1))
    ss = safety_stock_from_demand_std(avg_daily, std_daily, req.lead_time_days, req.service_level)
    rop = avg_daily * req.lead_time_days + ss

    return {
        "sku": req.sku,
        "method": method,
        "horizon_days": req.horizon_days,
        "predictions": [float(round(float(x),3)) for x in preds],
        "avg_daily": avg_daily,
        "std_daily": std_daily,
        "safety_stock": float(round(ss,3)),
        "reorder_point": float(round(rop,3))
    }
