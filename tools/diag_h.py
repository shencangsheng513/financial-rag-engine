# -*- coding: utf-8 -*-
"""诊断 H 类误召回：确认 长江电力/民生银行 是从 keyword 还是 vector 路径进来的。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag_engine_v2 import FinanceRAGEngineV2  # noqa: E402


def show(label, docs):
    print(f"  [{label}] {len(docs)} 块：", [d.get("metadata", {}).get("company") for d in docs][:5])


def main():
    eng = FinanceRAGEngineV2()
    for q in ("长江三角洲有哪些政策支持？", "民生问题最近有什么新闻？", "中国人民的生活水平如何？"):
        print(f"\n=== {q} ===")
        kw = eng.retriever.keyword_search(q, n_results=5)
        vec = eng.retriever.vector_search(q, n_results=5)
        show("keyword", kw)
        show("vector ", vec)
        final = eng.retrieve(q, top_k=5)
        print("  [final] ", [d.get("metadata", {}).get("company") for d in final][:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
