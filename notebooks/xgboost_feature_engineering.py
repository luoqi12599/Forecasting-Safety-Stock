# 基于 XGBoost 的特征工程与按 SKU 训练示例（使用日级数据）
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

ROOT = Path('.')
DATA_PATH = ROOT / 'data' / 'sample_sales.csv'
MODELS_DIR = ROOT / 'models_xgb'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'data'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values(['sku', 'date'])

def make_features(s: pd.Series, lags=[1,7,14,28]):
    df_feat = pd.DataFrame({'y': s})
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5,6]).astype(int)
    for l in lags:
        df_feat[f'lag_{l}'] = df_feat['y'].shift(l)
    for w in [7,14,28]:
        df_feat[f'rolling_mean_{w}'] = df_feat['y'].shift(1).rolling(window=w, min_periods=1).mean()
    df_feat = df_feat.dropna()
    return df_feat

results = []
for sku in sorted(df['sku'].unique()):
    print('XGB training:', sku)
    s = df[df['sku']==sku].groupby('date')['quantity'].sum().asfreq('D').fillna(0)
    if len(s) < 180:
        print('  too short, skip')
        continue
    feats = make_features(s)
    X = feats.drop(columns=['y'])
    y = feats['y']
    # time-series CV
    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    # simple full-train model (for demo)
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    # save model
    joblib.dump(model, MODELS_DIR / f"{sku}_xgb.pkl")
    # one-step-ahead test using last 28 days as test
    train_X, test_X = X[:-28], X[-28:]
    train_y, test_y = y[:-28], y[-28:]
    preds = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds)
    results.append({"sku": sku, "mae_28day": float(mae)})
    # save sample forecast (append next 7-day naive forecast example)
    future_idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    # generate features for future by shifting last known values (simple approach)
    last_window = s[-max(28,28):]
    future_df = pd.DataFrame(index=future_idx)
    # naive: use last day's value for lag_1, and rolling means from history
    for lag in [1,7,14,28]:
        future_df[f'lag_{lag}'] = last_window.shift(lag - 1).fillna(method='ffill').iloc[-1]
    future_df['day_of_week'] = future_idx.dayofweek
    future_df['day'] = future_idx.day
    future_df['month'] = future_idx.month
    future_df['is_weekend'] = future_idx.dayofweek.isin([5,6]).astype(int)
    # rolling means
    for w in [7,14,28]:
        future_df[f'rolling_mean_{w}'] = s.shift(1).rolling(window=w, min_periods=1).mean().iloc[-1]
    future_preds = model.predict(future_df)
    pd.DataFrame({'ds': future_idx, 'y_pred': future_preds}).to_csv(OUT / f'xgb_forecast_{sku}.csv', index=False)

pd.DataFrame(results).to_csv(OUT / 'xgb_model_summary.csv', index=False)
print('Wrote XGB outputs to', OUT)
