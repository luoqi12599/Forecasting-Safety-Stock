"""
LLM 自动洞察模块（示例使用 OpenAI API）
功能：
- 根据传入的 SKU 指标/预测表格构建 prompt，生成 executive summary（要点 + 建议）
- 简单的校验逻辑：确保 LLM 输出中包含关键数值或提及 Top-N SKU（若适用）
注意：真实使用需配置 OPENAI_API_KEY 环境变量
"""
import os
import json
from typing import Dict, Any, List
import openai
import re

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def build_prompt(metrics: Dict[str, Any], top_reasons: List[str]=None) -> str:
    """
    metrics: 包含 sku, avg_daily, safety_stock, reorder_point, upcoming_pred (list) 等
    """
    lines = []
    lines.append(f"你是经验丰富的供应链分析师，下面是 SKU 的关键指标，请基于这些信息写一份 5-7 条的 executive summary（中文），包含：一段 1-2 句的总体结论，接着 3-4 条可执行建议（优先级排序），最后列出关键数字（avg_daily, safety_stock, reorder_point）。")
    lines.append("数据:")
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    if top_reasons:
        lines.append("已知的高风险驱动因素：")
        lines.append("\n".join(f"- {r}" for r in top_reasons))
    lines.append("输出要求：")
    lines.append("- 使用中文；每条建议尽量简短并可执行；")
    lines.append("- 在结尾用 JSON 格式列出 summary_keys 包含 'sku','avg_daily','safety_stock','reorder_point'，以便机器读取。")
    prompt = "\n\n".join(lines)
    return prompt

def call_openai(prompt: str, model="gpt-3.5-turbo", max_tokens=300):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 OpenAI")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"]

def validate_summary(summary_text: str, metrics: Dict[str,Any]) -> Dict[str,Any]:
    """
    简单校验：尝试从 summary_text 中提取关键数值，并与 metrics 做容差比较
    返回字典 {valid:bool, issues:list, extracted:dict}
    """
    issues = []
    extracted = {}
    # 提取数字（非常简单的策略，提取第一个出现的数字作为 avg_daily 等）
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", summary_text.replace(",", ""))
    # naive mapping: look for occurrences of keywords
    for key in ["avg_daily","safety_stock","reorder_point"]:
        if key in summary_text:
            # try find number after key
            m = re.search(rf"{key}[^0-9\-+]*([0-9]+(?:\.[0-9]+)?)", summary_text)
            if m:
                extracted[key] = float(m.group(1))
    # compare extracted numeric values if available
    for k,v in extracted.items():
        if k in metrics:
            expected = float(metrics[k])
            if abs(expected - v) / max(1.0, expected) > 0.2:  # 20% tolerance
                issues.append(f"value for {k} deviates by >20% (expected {expected}, got {v})")
    valid = len(issues)==0
    return {"valid": valid, "issues": issues, "extracted": extracted}

# 示例封装：generate_insight(metrics)
def generate_insight(metrics: Dict[str,Any], top_reasons: List[str]=None):
    prompt = build_prompt(metrics, top_reasons)
    text = call_openai(prompt)
    check = validate_summary(text, metrics)
    return {"text": text, "check": check}
