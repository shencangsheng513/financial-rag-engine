# -*- coding: utf-8 -*-
"""用中文 embedding 重建向量库（**只读原库、写新库**，原库一个字节都不动）。

为什么这样建（换 embedding 的 A/B 纪律）：
- 从**现有库**读出 664 块的 id/document/metadata，只用新模型重新编码，写进新库。
  这样前后两个库的内容完全一致，唯一变量就是 embedding —— 召回率的差异
  可以干净地归因到模型，而不是"顺手也改了分块/元数据"。
- 用 chroma **内置** SentenceTransformerEmbeddingFunction：它的模型名会写进
  collection 元数据，之后引擎里那句 `client.get_collection(name)`（不传 EF）
  会自动复用同一个模型。**因此不需要改任何生产代码**，换库 = 换路径。

用法（D:/ML_Projects/.venv 解释器）：
    python rag-finance-qa/tools/build_zh_vector_db.py                    # 默认 bge-small-zh-v1.5
    python rag-finance-qa/tools/build_zh_vector_db.py --dry-run          # 只报数量不建库
    python rag-finance-qa/tools/build_zh_vector_db.py --model BAAI/bge-base-zh-v1.5
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CHROMA_DB_PATH, COLLECTION_NAME  # noqa: E402（repo 根 config.py 已负责 shared 路径）

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-db", default=CHROMA_DB_PATH)
    ap.add_argument("--dst-db", default=str(ROOT / "data" / "chroma_db_zh"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import chromadb
    from chromadb.utils import embedding_functions

    print(f"[src] {args.src_db}")
    src = chromadb.PersistentClient(path=args.src_db).get_collection(COLLECTION_NAME)
    got = src.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    print(f"[src] {len(ids)} 块 / {len(docs)} 文档 / {len(metas or [])} 元数据")
    if args.dry_run:
        print("[dry-run] 未建库，退出")
        return 0
    if not docs:
        print("[ERROR] 源库没有文档，退出")
        return 1

    print(f"[model] 加载 {args.model} …（首次会从 HF 下载，可设 HF_ENDPOINT=https://hf-mirror.com 加速）")
    t0 = time.time()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model)
    print(f"[model] 就绪 {time.time() - t0:.1f}s")

    texts = [d if d else " " for d in docs]
    t1 = time.time()
    vectors = ef(texts)
    print(f"[embed] {len(vectors)} 条，维度 {len(vectors[0]) if vectors else '?'}，"
          f"耗时 {time.time() - t1:.1f}s")

    dst_path = Path(args.dst_db)
    if dst_path.exists():
        print(f"[WARN] 目标库已存在：{dst_path} —— 先删掉再建，避免混入旧向量")
        return 1

    print(f"[dst] 写入 {dst_path}")
    dst_client = chromadb.PersistentClient(path=str(dst_path))
    dst = dst_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    B = args.batch
    for i in range(0, len(ids), B):
        dst.add(
            ids=ids[i:i + B],
            documents=docs[i:i + B],
            metadatas=metas[i:i + B] if metas else None,
            embeddings=vectors[i:i + B],
        )
        print(f"  写入 {min(i + B, len(ids))}/{len(ids)}")

    n = dst.count()
    src_n = src.count()
    print(f"[check] 新库 {n} 块 / 原库 {src_n} 块 -> {'一致 ✅' if n == src_n else '不一致 ❌'}")
    print(f"[done] 新库路径（喂给 run_eval.py --db）：{dst_path}")
    return 0 if n == src_n else 1


if __name__ == "__main__":
    sys.exit(main())
