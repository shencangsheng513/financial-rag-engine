"""
向量库「结构化部分」定点重建：把 overview / financial / stock 三类块按**当前库**
重做，保留 research_report（研报）块不动。

## 为什么需要这个脚本（2026-09-10）

线上向量库有 269 条，其中 199 条结构化块是**改名迁移之前**的旧快照产物，
`company` 元数据写着「华兴科技 / 中通达 / 盛远集团…」——这些公司在当前库
（finance_db_v2，92 家真实 A 股）里**根本不存在**。结果：任何走向量检索的提问
都可能命中一个虚构公司，模型照单转述。

同时按「宁可缺失，不可错误」剔除三张合成表：
credit_ratings / transactions / employees 只覆盖 company_id 1–20 且内容伪造
（浦发银行=设计部3人、BBB+ 评级、对手方阿里巴巴），不再入库。

## 范围

  - 删除：metadata.type ∈ {overview, financial, stock, rating, transaction, employee}
          以及任何 source 非 research_report 的历史遗留块
  - 保留：source == research_report 的研报块
  - 写入：按当前库重新生成的 overview / financial / stock

## 用法

    python tools/rebuild_structured_chunks.py --dry-run   # 只统计与取样，不写库
    python tools/rebuild_structured_chunks.py             # 执行
"""

import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import pymysql
from config import MYSQL_CONFIG, CHROMA_DB_PATH, COLLECTION_NAME

from src.data_extractor import extract_company_info, company_to_text_chunks

KEEP_SOURCE = "research_report"
STRUCTURED_TYPES = {"overview", "financial", "stock", "rating", "transaction", "employee"}


def build_new_chunks():
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM companies ORDER BY id")
    company_ids = [r[0] for r in cursor.fetchall()]

    chunks, n_missing_profile = [], 0
    for cid in company_ids:
        info = extract_company_info(cursor, cid)
        if not info:
            continue
        if info["website"] is None and info["hq"] is None:
            n_missing_profile += 1
        chunks.extend(company_to_text_chunks(info))
    cursor.close()
    conn.close()
    return chunks, len(company_ids), n_missing_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_chunks, n_companies, n_missing_profile = build_new_chunks()
    by_type = Counter(c["metadata"]["type"] for c in new_chunks)
    print(f"[1/5] 当前库公司数={n_companies}；"
          f"其中 {n_missing_profile} 家档案字段为「未收录」（合成档案已作废）")
    print(f"      待写入新结构化块 {len(new_chunks)} 条，类型分布：{dict(by_type)}")
    for c in new_chunks[:2]:
        print(f"      - {c['id']}: {c['text'][:110]}".replace("\n", " "))

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    before = col.count()

    existing = col.get(include=["metadatas"])
    keep, drop = [], []
    for i, m in zip(existing["ids"], existing["metadatas"]):
        (keep if m.get("source") == KEEP_SOURCE else drop).append(i)
    print(f"[2/5] 现状 {before} 条：保留研报块 {len(keep)} 条，"
          f"待删除结构化块 {len(drop)} 条")

    stale_companies = sorted({
        m.get("company") for i, m in zip(existing["ids"], existing["metadatas"])
        if i in set(drop) and m.get("company")
    })
    print(f"      将删除的块涉及 {len(stale_companies)} 个公司名：{stale_companies[:8]} ...")

    if args.dry_run:
        print("[dry-run] 未写库。")
        return

    # 删：只删结构化块，研报块原样保留
    for i in range(0, len(drop), 100):
        col.delete(ids=drop[i:i + 100])
    print(f"[3/5] 已删除 {len(drop)} 条结构化块，剩余 {col.count()} 条")

    # 写：新结构化块
    for i in range(0, len(new_chunks), 50):
        b = new_chunks[i:i + 50]
        col.upsert(
            ids=[c["id"] for c in b],
            documents=[c["text"] for c in b],
            metadatas=[c["metadata"] for c in b],
        )
    print(f"[4/5] 已写入 {len(new_chunks)} 条，当前总数 {col.count()}")

    # 自检
    r = col.get(include=["metadatas"])
    types = Counter(m.get("type") for m in r["metadatas"])
    comps = {m.get("company") for m in r["metadatas"] if m.get("company")}
    ghost = [c for c in comps if c in {"华兴科技", "中通达", "盛远集团", "创维电子", "博瑞医药"}]
    print(f"[5/5] 类型分布：{dict(types)}")
    print(f"      公司名去重 {len(comps)} 个；残留旧合成公司名：{ghost or '无'}")
    print(f"      最终总数 {col.count()}（= 研报 {len(keep)} + 结构化 {len(new_chunks)}）")


if __name__ == "__main__":
    main()
