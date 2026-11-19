# -*- coding: utf-8 -*-
"""诊断 A06：东方财富 overview 为何召回不到。
只查两件事：
  1) 库里到底有没有 company=东方财富 & type=overview 的块（语料缺口？）
  2) A06 查询下，东方财富相关块排第几（重排压下去了？）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CHROMA_DB_PATH, COLLECTION_NAME  # noqa: E402

import chromadb  # noqa: E402


def main():
    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION_NAME)
    n = col.count()
    print(f"[db] {CHROMA_DB_PATH}\n[db] 共 {n} 块")

    # 1) 东方财富 的所有块
    got = col.get(where={"company": "东方财富"}, include=["metadatas", "documents"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    print(f"\n[1] company=东方财富 命中 {len(ids)} 块：")
    for i, (cid, m, d) in enumerate(zip(ids, metas, docs)):
        t = m.get("type") if m else None
        head = (d or "")[:40].replace("\n", " ")
        print(f"   {i+1:2d}. type={t!s:10s} id={cid[:24]}  {head}")

    # 2) A06 查询，看东方财富相关块排第几
    q = "东方财富是哪个行业的公司？"
    res = col.query(query_texts=[q], n_results=min(20, n), include=["metadatas", "distances"])
    rids = res["ids"][0]
    rmetas = res["metadatas"][0]
    rdists = res["distances"][0]
    print(f"\n[2] 查询「{q}」top{n if n<20 else 20}：")
    dfc_rank = None
    for rank, (cid, m, dist) in enumerate(zip(rids, rmetas, rdists), 1):
        comp = m.get("company") if m else None
        typ = m.get("type") if m else None
        mark = "  <== 东方财富" if comp == "东方财富" else ""
        if comp == "东方财富" and dfc_rank is None:
            dfc_rank = rank
        print(f"   #{rank:2d} dist={dist:.4f} {comp}/{typ}{mark}")
    if dfc_rank is None:
        print("\n[结论] 东方财富 的块根本没进 top20 —— 看 [1] 判断是语料缺口还是向量太远")
    else:
        print(f"\n[结论] 东方财富 最靠前排在 #{dfc_rank}（top_k=5 截断，所以召回不到）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
