# Short-term Demand Forecasting & Safety Stock (仓库名: demand-safety-stock)

项目简介
- 目标：为 SKU 构建短期需求预测并据此计算安全库存（Safety Stock）与补货建议，支持减少缺货与降低持有成本。
- 产出：样例数据、Jupyter-style 分析脚本、安全库存计算模块、Streamlit 演示应用。

结构
- data/                # 数据位置（生成脚本会写入 sample_sales.csv）
- notebooks/           # 分析脚本与示例（forecast_and_safety_stock.py）
- src/                 # 工具函数（safety_stock.py）
- app/                 # Streamlit demo（streamlit_demo.py）
- requirements.txt
- README.md

快速开始（本地）
1. 克隆仓库并创建虚拟环境：
   python -m venv venv && source venv/bin/activate  # Windows: venv\\Scripts\\activate
2. 安装依赖：
   pip install -r requirements.txt
3. 生成样例数据：
   python data/sample_data_generator.py
   会生成 data/sample_sales.csv
4. 在 notebooks/ 打开并运行分析脚本（或直接运行脚本）：
   jupyter notebook     # 或直接用 VSCode 打开 notebooks/forecast_and_safety_stock.py
5. 启动演示应用：
   streamlit run app/streamlit_demo.py

文件说明（核心步骤）
- data/sample_data_generator.py：生成带季节性、趋势与随机噪声的日级销量样例数据（多 SKU）。
- notebooks/forecast_and_safety_stock.py：EDA → 汇总到周/日粒度 → 建模（Prophet 或 ExponentialSmoothing / XGBoost）→ 评估 → 计算安全库存并输出建议表。
- src/safety_stock.py：安全库存计算函数（支持基于需求方差或基于预测残差两种方法）。
- app/streamlit_demo.py：交互式 demo，选择 SKU、预测周期、服务水平，展示预测与建议安全库存。

如何把结果写进简历（示例）
- 中文：开发短期需求预测与安全库存计算流程（Prophet + XGBoost），在样例项目中将预计缺货天数减少示例值 X%，并生成按 SKU 的补货建议与 CSV 报表用于业务下发。
- English: Built a short-term demand forecasting and safety-stock pipeline (Prophet + XGBoost), producing SKU-level replenishment recommendations and ROP tables to reduce expected stockout days by X% (example).

扩展建议（可选加分项）
- 把预测/安全库存封装为 REST API（FastAPI），并在 CI 中自动跑回测；
- 将 LLM（如 OpenAI）接入，自动把关键 KPI 转为一页 executive summary（示例：每周 top-5 风险 SKU 与建议）。

---

## 打开 / 转换 Notebook (.ipynb)
我同时提供了 .py 与 .ipynb 版本。你可以：
- 直接打开 notebooks/*.ipynb（在 JupyterLab / Jupyter Notebook 中）；
- 或使用 jupytext 在 .py 和 .ipynb 之间互转：
  - pip install jupytext
  - 将 .py 转为 .ipynb： `jupytext --to ipynb notebooks/forecast_and_safety_stock_full.py`
  - 将 .py 转为 .ipynb： `jupytext --to ipynb notebooks/xgboost_feature_engineering.py`

## FastAPI 演示（启动与请求示例）
API 路径： `POST /predict` （定义见 src/api/app.py）

启动（本地）：
1. 确保已训练并生成 models 或 models_xgb（参见 notebooks）或使用 sample 数据；
2. 启动 uvicorn：
   - 本地： `uvicorn src.api.app:app --reload`
   - Docker：
     - 构建镜像： `docker build -t demand-forecast-api .`
     - 运行容器： `docker run -p 8000:8000 demand-forecast-api`
3. 打开 Swagger UI（如果本地运行）: http://127.0.0.1:8000/docs

curl 请求示例：
- 基本请求（JSON body）：
  curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"sku":"SKU_001","horizon_days":14,"lead_time_days":14,"service_level":0.95}'

- 如果用 Docker 且容器映射到主机 8000 端口，使用相同的 curl（替换 host 与端口为部署地址）。

Python requests 示例：
```python
import requests
url = "http://127.0.0.1:8000/predict"
payload = {"sku":"SKU_001","horizon_days":14,"lead_time_days":14,"service_level":0.95}
resp = requests.post(url, json=payload)
print(resp.json())
```

返回字段说明（示例）：
- sku: 请求的 SKU；
- method: 使用的模型（xgboost 或 hw）；
- predictions: 未来 N 天的日级预测（float list）；
- avg_daily / std_daily: 历史日均与日级标准差；
- safety_stock / reorder_point: 基于当前参数计算的安全库存与 ROP。

## LLM 洞察（快速示例）
- 设置环境变量： `export OPENAI_API_KEY="sk-..."`
- 在 Python 中示例调用（参见 src/llm/insights.py）：
```python
from src.llm.insights import generate_insight
metrics = {"sku":"SKU_001","avg_daily":12.3,"safety_stock":45.6,"reorder_point":215}
out = generate_insight(metrics)
print(out["text"]) 
print(out["check"])
```

---

完整的运行步骤与调试提示请参见仓库根目录下的 notebooks 与 app 目录。
