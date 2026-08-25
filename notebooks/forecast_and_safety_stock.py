# Jupyter-style 脚本（把下面内容分成单元运行）
# 1) 环境准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Prophet 可选（如果已安装）
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from src.safety_stock import safety_stock_from_demand_std, safety_stock_from_forecast_errors

DATA_PATH = Path("data/sample_sales.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.head()

# 2) 简单 EDA：按 SKU 聚合
# 选择一个 SKU 示例
sku = "SKU_001"
df_sku = df[df["sku"] == sku].sort_values("date")
daily = df_sku.groupby("date")["quantity"].sum().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date").asfreq("D").fillna(0)
daily["rolling7"] = daily["quantity"].rolling(7, min_periods=1).mean()

# 可视化
plt.figure(figsize=(12,4))
plt.plot(daily.index, daily["quantity"], label="quantity")
plt.plot(daily.index, daily["rolling7"], label="rolling7")
plt.legend(); plt.title(f"Daily sales - {sku}")

# 3) 汇总到周（用于稳定建模）
weekly = daily["quantity"].resample("W").sum()
weekly.plot(title=f"Weekly sales - {sku}")

# 4) 基线模型：Exponential Smoothing
train = weekly[:-12]  # 留 12 周作为测试
test = weekly[-12:]
model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=52).fit()
pred = model.forecast(len(test))
print("MAE (HW):", mean_absolute_error(test, pred))

# 5) 可选：Prophet 建模（若可用）
if PROPHET_AVAILABLE:
    df_prophet = weekly.reset_index().rename(columns={"date":"ds", "quantity":"y"})
    m = Prophet()
    m.fit(df_prophet[:-12])
    future = m.make_future_dataframe(periods=12, freq="W")
    fcst = m.predict(future)
    pred_p = fcst.set_index("ds")["yhat"][-12:]
    print("MAE (Prophet):", mean_absolute_error(test, pred_p))

# 6) 误差分析，计算安全库存（示例以日为单位）
# 方法 A: 基于日需求标准差
# 把 weekly 预测拆回日均（粗估）
avg_daily = daily["quantity"].mean()
std_daily = daily["quantity"].std(ddof=1)
lead_time_days = 14  # 示例提前期 14 天
ss = safety_stock_from_demand_std(avg_daily, std_daily, lead_time_days, service_level=0.95)
print(f"Safety stock (demand std method): {ss:.1f} units")

# 方法 B: 基于模型残差（更稳健，需按同粒度）
# 这里示例使用 weekly 模型残差并换算为日（粗估）
residuals = test - pred
# 将周级残差转为日级 std（除以 sqrt(7) 仅为示例换算）
resid_std_daily = residuals.std(ddof=1) / np.sqrt(7)
ss2 = safety_stock_from_forecast_errors(np.array(residuals), lead_time_days, service_level=0.95)
print(f"Safety stock (forecast errors method): {ss2:.1f} units (weekly-based)")

# 7) 输出补货建议表（示例）
reorder_point = avg_daily * lead_time_days + ss  # ROP = avg_demand * LT + SS
print(f"Reorder point (example): {reorder_point:.1f} units")
# 将结果写入 CSV 方便业务下发
out = pd.DataFrame([{
    "sku": sku,
    "avg_daily": avg_daily,
    "std_daily": std_daily,
    "lead_time_days": lead_time_days,
    "safety_stock": ss,
    "reorder_point": reorder_point
}])
out.to_csv("data/replenishment_example.csv", index=False)
print("Wrote data/replenishment_example.csv")
