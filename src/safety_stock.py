"""
安全库存计算工具函数
方法说明：
1) 基于需求方差（常用）:
   SS = z * sigma_demand * sqrt(lead_time_days)
   其中 sigma_demand 为每日需求标准差（与 lead_time 同单位）
2) 基于预测误差（更稳健）:
   使用历史预测残差的标准差代替需求波动（考虑模型的误差）
"""
import math
from scipy.stats import norm
import numpy as np

def z_from_service_level(service_level: float) -> float:
    """
    service_level: 0.95 (95% 服务水平) -> 返回 z 值（单侧）
    """
    return norm.ppf(service_level)

def safety_stock_from_demand_std(avg_daily_demand: float, std_daily_demand: float, lead_time_days: float, service_level: float=0.95) -> float:
    z = z_from_service_level(service_level)
    ss = z * std_daily_demand * math.sqrt(lead_time_days)
    return max(0.0, ss)

def safety_stock_from_forecast_errors(forecast_errors: np.ndarray, lead_time_days: float, service_level: float=0.95) -> float:
    """
    forecast_errors: 历史 (actual - predicted) 的日级数组（或按预测粒度）
    """
    sigma = np.std(forecast_errors, ddof=1)
    z = z_from_service_level(service_level)
    ss = z * sigma * math.sqrt(lead_time_days)
    return max(0.0, ss)
