#!/usr/bin/env python3
"""
生成样例销售数据：日级销量，多 SKU，多仓库，带季节性与随机性。
输出： data/sample_sales.csv
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)
np.random.seed(42)

start = datetime(2024, 1, 1)
end = datetime(2025, 6, 30)
days = (end - start).days + 1
dates = [start + timedelta(days=i) for i in range(days)]

# 定义若干 SKU（示例 10 个）
sku_list = [f"SKU_{i:03d}" for i in range(1, 11)]
warehouses = ["WH_A", "WH_B"]

rows = []
for sku in sku_list:
    base = np.random.randint(5, 200)  # 基础日均销量（不同 SKU 差异）
    trend = (np.linspace(0, 1, days) * np.random.uniform(-0.2, 0.6))  # 轻微趋势
    weekly = 1 + 0.2 * np.sin(2 * np.pi * (np.arange(days) % 7) / 7)  # 周效应
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * (np.arange(days) / 365.0))  # 年季节性
    noise = np.random.normal(0, 1, days)
    for i, d in enumerate(dates):
        qty = max(0, int(base * (1 + trend[i]) * weekly[i] * seasonal[i] + noise[i]*np.sqrt(base)))
        # 偶尔促销日（注入 spike）
        if np.random.rand() < 0.01:
            qty = qty + int(base * np.random.uniform(1.5, 4.0))
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "sku": sku,
            "warehouse": np.random.choice(warehouses, p=[0.7, 0.3]),
            "quantity": qty,
            "unit_price": round(np.random.uniform(10, 200), 2)
        })

df = pd.DataFrame(rows)
df.to_csv("data/sample_sales.csv", index=False)
print("Wrote data/sample_sales.csv  (rows = {})".format(len(df)))
