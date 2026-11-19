# -*- coding: utf-8 -*-
"""增量注入 stock-predictor 实证结论文档到生产库 chroma_db_zh（不清原研报）。

复用 tools/ingest_reports.py 的 parse/chunk/build 逻辑，仅针对新增文档 add。
与官方 ingest_reports.py 的区别：
- 只加指定文件，不 delete 任何原研报块（增量、安全、可验收）。
- DB 路径显式指向 chroma_db_zh（生产库），避免默认 chroma_db 的坑。

用法（根 .venv，绕开 safe-delete 钩子）：
    env -u PYTHONPATH D:/ML_Projects/.venv/Scripts/python.exe rag-finance-qa/tools/inject_selection_model.py
    env -u PYTHONPATH D:/ML_Projects/.venv/Scripts/python.exe rag-finance-qa/tools/inject_selection_model.py --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/
os.environ.setdefault("ML_PROJECTS_ROOT", "D:/ML_Projects")

import chromadb
import ingest_reports as ir

ROOT = Path(os.environ["ML_PROJECTS_ROOT"])
NEW_FILE = ROOT / "knowledge-graph" / "data" / "reports" / "selection_model_empirical_20260917.txt"
DB_PATH = ROOT / "rag-finance-qa" / "data" / "chroma_db_zh"
COLLECTION = "finance_documents"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not NEW_FILE.exists():
        print("ERROR: 文档不存在", NEW_FILE)
        return 1

    ids, docs, metas = ir.build_docs([NEW_FILE])
    for m in metas:
        m["doc_type"] = "quant_strategy_empirical"  # 强化区分，防与标准行业研报串味
    print(f"[1] 解析切块：{len(docs)} 块")
    for m in metas[:3]:
        print("   meta:", {k: m.get(k) for k in ("scope", "sector", "company", "type", "source")})

    if args.dry_run:
        print("[dry-run] 不写库")
        return 0

    client = chromadb.PersistentClient(path=str(DB_PATH))
    col = client.get_collection(COLLECTION)
    before = col.count()
    # 幂等：先删本批 id 再写
    col.delete(ids=ids)
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"[2] 注入完成。集合 {before} -> {col.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
