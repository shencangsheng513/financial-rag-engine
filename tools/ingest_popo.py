# -*- coding: utf-8 -*-
"""把 MinerU-Popo 的 RAG 切块（JSONL）增量入库到 finance_documents。

上游产物：D:/MinerU-Popo/popo_to_rag_chunks.py 生成的 <doc>.jsonl，
每行一个块：heading_path / level / pages / block_ids / text / type。

用法（在 D:/ML_Projects 下）:
    .venv/Scripts/python.exe rag-finance-qa/tools/ingest_popo.py --jsonl <path.jsonl> --doc <名称> --dry-run
    .venv/Scripts/python.exe rag-finance-qa/tools/ingest_popo.py --jsonl <path.jsonl> --doc <名称>
    .venv/Scripts/python.exe rag-finance-qa/tools/ingest_popo.py --purge        # 只删 popo 块

约定（对齐 ingest_reports.py 的踩坑记录）:
- 只做增量 add，幂等：先按 source=popo_pdf 清旧块再写。
- 非公司文档不写 company/stock_code（行业研报先例，防 by_company 污染）。
- type=research_report + title + chunk → 复用 engine 的父子分块回填，
  title=论文/文档标题（heading_path 首段），chunk=该标题下的块序号。
- engine 会丢弃 <50 字符块（rag_engine_v2.py 的 text_len 过滤），
  本脚本在入库前显式拒绝并打印，不静默丢。
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("ML_PROJECTS_ROOT", "D:/ML_Projects"))
DB_PATH = Path(os.environ.get("RAG_CHROMA_DB_PATH", ROOT / "rag-finance-qa" / "data" / "chroma_db_zh"))
COLLECTION = os.environ.get("RAG_COLLECTION_NAME", "finance_documents")

SOURCE_TAG = "popo_pdf"
MIN_CHARS = 50  # 与 rag_engine_v2._score 的 text_len<50 过滤同源，别改小


def load_jsonl(path: Path):
    chunks = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        for key in ("heading_path", "text", "pages"):
            if key not in c:
                raise ValueError(f"line {ln}: 缺少字段 {key}，确认来自 popo_to_rag_chunks.py")
        chunks.append(c)
    return chunks


def build_docs(chunks, doc_name):
    ids, docs, metas = [], [], []
    seq_by_title = defaultdict(int)
    dropped = []
    for c in chunks:
        text = c["text"].strip()
        if len(text) < MIN_CHARS:
            dropped.append((c["heading_path"], len(text)))
            continue
        parts = [p for p in c["heading_path"].split(" > ") if p.strip()]
        doc_title = parts[0] if parts else doc_name
        leaf = parts[-1] if parts else doc_name
        pages = c.get("pages") or []
        pages_s = ",".join(str(p) for p in pages)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        m = {
            "source": SOURCE_TAG,
            "type": "research_report",
            "scope": "document",
            "doc": doc_name,
            "title": doc_title,
            "heading_path": c["heading_path"],
            "level": int(c.get("level", -1)),
            "pages": pages_s,
            "block_ids": ",".join(str(b) for b in c.get("block_ids", [])),
            "chunk": seq_by_title[doc_title],
        }
        seq_by_title[doc_title] += 1
        ids.append(f"popo_{doc_name}_{h}")
        docs.append(f"【{doc_title}·{leaf}】(p.{pages_s or '?'})\n{text}")
        metas.append(m)
    return ids, docs, metas, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="popo_to_rag_chunks.py 的 JSONL 产物路径")
    ap.add_argument("--doc", help="文档标识，用于 id 前缀与 where 过滤")
    ap.add_argument("--dry-run", action="store_true", help="只解析打印，不碰库")
    ap.add_argument("--purge", action="store_true", help="删除全部 source=popo_pdf 块后退出")
    args = ap.parse_args()

    if args.jsonl:
        chunks = load_jsonl(Path(args.jsonl))
    else:
        chunks = None

    ids = docs = metas = None
    dropped = []
    if chunks is not None:
        if not args.doc:
            ap.error("--jsonl 与 --doc 必须同时提供")
        ids, docs, metas, dropped = build_docs(chunks, args.doc)
        print(f"[1/3] 读入 {len(chunks)} 块，采纳 {len(docs)}，拒收(<{MIN_CHARS}字) {len(dropped)}")
        for hp, n in dropped:
            print(f"      拒收: {hp} ({n}字)")

    if args.dry_run:
        if metas is None:
            ap.error("--dry-run 需要 --jsonl 与 --doc")
        for i, m in enumerate(metas[:3]):
            print(f"      样本{i}: title={m['title']} chunk={m['chunk']} "
                  f"heading={m['heading_path']} pages={m['pages']}")
        print("[dry-run] 未碰数据库")
        return 0

    if args.jsonl and dropped:
        print(f"[abort] 有 {len(dropped)} 块低于 {MIN_CHARS} 字被拒，先修正切块再入库（不静默丢）")
        return 2

    if not args.purge and (ids is None or not ids):
        ap.error("需要 --jsonl/--doc（或 --purge）")

    import chromadb

    client = chromadb.PersistentClient(path=str(DB_PATH))
    col = client.get_collection(COLLECTION)

    if args.purge:
        old = col.get(where={"source": SOURCE_TAG}, include=[])["ids"]
        if old:
            col.delete(ids=old)
        print(f"[purge] 已删除 {len(old)} 块，集合现 {col.count()}")
        return 0

    # chroma 1.5 的 get(where=) 不吃多键隐式与 → 单键过滤 + Python 侧二次筛
    got = col.get(where={"source": SOURCE_TAG}, include=["metadatas"])
    old = [i for i, m in zip(got["ids"], got["metadatas"]) if m.get("doc") == args.doc]
    if old:
        col.delete(ids=old)
    before = col.count()
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"[2/3] 清理该 doc 旧块 {len(old)}，写入 {len(ids)}")
    print(f"[3/3] 集合总数 {before} -> {col.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
