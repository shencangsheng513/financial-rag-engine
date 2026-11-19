# -*- coding: utf-8 -*-
"""用真实引擎 retrieve 跑 A06，打印最终返回的 company/type/rerank_score，
定位 overview 为何掉出 top-5。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rag_engine_v2 import FinanceRAGEngineV2  # noqa: E402


def main():
    eng = FinanceRAGEngineV2()
    q = "东方财富是哪个行业的公司？"
    docs = eng.retrieve(q, top_k=5)
    print(f"\n=== retrieve('{q}') 返回 {len(docs)} 块 ===")
    for i, d in enumerate(docs, 1):
        m = d.get("metadata", {})
        rs = d.get("rerank_score")
        vs = d.get("vector_score")
        rs_s = f"{rs:.3f}" if isinstance(rs, float) else str(rs)
        vs_s = f"{vs:.3f}" if isinstance(vs, float) else str(vs)
        print(f"  #{i} company={m.get('company')} type={m.get('type')} rerank={rs_s} vec={vs_s}")
    # 顺带看候选池里 overview 的排名（不经过 [:5] 截断）
    cands = eng.retriever.hybrid_search(q, n_results=15)
    print(f"\n=== hybrid_search 候选池 {len(cands)} 块 ===")
    for i, d in enumerate(cands, 1):
        m = d.get("metadata", {})
        print(f"  #{i} company={m.get('company')} type={m.get('type')}")
    # 直接看 rerank 对这 7 块的打分，定位 overview 为何掉队
    reranked = eng.reranker.rerank(q, cands, top_k=7)
    print(f"\n=== rerank 对候选池打分（top 7）===")
    for i, d in enumerate(reranked, 1):
        m = d.get("metadata", {})
        rs = d.get("rerank_score")
        vs = d.get("vector_score")
        rs_s = f"{rs:.3f}" if isinstance(rs, float) else str(rs)
        vs_s = f"{vs:.3f}" if isinstance(vs, float) else str(vs)
        print(f"  #{i} company={m.get('company')} type={m.get('type')} rerank={rs_s} vec={vs_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
