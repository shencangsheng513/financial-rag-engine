# -*- coding: utf-8 -*-
"""验证 RAG 提示词的金额单位约束：跑真实检索 + 真实 LLM，检查是否再犯 10 倍错误。

背景：2026-09-10 线上 `rag_ask` 回答「新宇能源市值 414,872,927.75 万元
（即 4148.73 亿元）」——换算错了一个数量级（应为 41487.29 亿元）。
原因是语料里金额单位是**万元**，而提示词只说「精确引用数字」、不给单位规则。
本脚本在改提示词后复跑同一问题，人眼确认量级正确。

用法（在 D:/ML_Projects 下）：
    .venv/Scripts/python.exe rag-finance-qa/tools/check_unit_prompt.py
"""
import os
import re
import sys

ROOT = r"D:/ML_Projects"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rag-finance-qa", "src"))

# key 由运行环境提供（本地 .env / 环境变量），源码中不存放凭证

from rag_engine import FinanceRAGEngine  # noqa: E402

QUERIES = [
    "新宇能源的市值是多少？",
    "鹏华电子的财务状况如何？",
]

YI = re.compile(r"([\d,]+(?:\.\d+)?)\s*亿元")


def main() -> int:
    engine = FinanceRAGEngine()
    bad = 0
    for q in QUERIES:
        res = engine.ask(q, use_llm=True)
        ans = res["answer"]
        print("=" * 70)
        print("Q:", q)
        print(ans[:420])
        found = YI.findall(ans)
        print(f"  -> 文中出现的『亿元』数值: {found}")
        if not found:
            print("  ⚠ 回答里没有出现以『亿元』标注的金额（可能未按单位要求作答）")
        print()
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
