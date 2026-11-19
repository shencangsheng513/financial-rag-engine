# -*- coding: utf-8 -*-
"""把 knowledge-graph/data/reports/ 下的研报文本批量入库到 ChromaDB。

用法（在 D:/ML_Projects 下）:
    .venv/Scripts/python.exe rag-finance-qa/tools/ingest_reports.py            # 入库
    .venv/Scripts/python.exe rag-finance-qa/tools/ingest_reports.py --dry-run  # 只看解析结果

特性:
- 复用现有集合 finance_documents（180 块公司数据），只做增量 add，不重建
- metadata **对齐既有约定**：`company` + `stock_code`（与 data_extractor.py / web_app.py 一致）
- 段落聚合分块：~600 字符/块
- 幂等：id 含内容 hash；每次先清掉旧的 report_* 块再写，改元数据后重跑即生效

## metadata 约定（踩过的坑，别改回去）

旧版写的是 `stock`（不是 `stock_code`）、且**完全没有 `company`**。而全库读取方
（web_app.py 引用展示、rag_engine.py stats 的 by_company、rag_engine_v2.py 的
`where={"company": {"$contains": ...}}` 过滤）一律认 `company` / `stock_code`。
后果：89 块研报在 by_company 里全部落进 "unknown"，引用里公司名显示成 "?"。

本脚本按报告形态分两类：
  公司研报 report_001–080：文件名带 SH 代码，标题「XX深度研究报告」
      → company=公司名, stock_code=SHxxxxxx, sector=行业, scope="company"
  行业研报 report_081–089：标题「XX行业深度研究报告」，无公司
      → 不写 company/stock_code（不冒充公司），只写 sector + scope="industry"

`industry` 两类都要写（2026-09-24 REQ-049）：检索层的行业通道是
`where={"industry": 真值}`，只有 sector（口语名）等于这个行业问题结构性捞不到研报。
公司研报按 stock_code 回查库里档案，行业研报只在 sector 恰好已是库内真值时才写。
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb

# 路径改为 env 可覆盖（原先写死 D:/ML_Projects，容器内不存在 → 定时 ingest 必失败）。
# 部署时通过 ML_PROJECTS_ROOT / RAG_REPORT_DIR / RAG_CHROMA_DB_PATH 指向挂载卷；
# 本地未设环境变量时回退到 D:/ML_Projects（仅本地开发用，容器必须设环境变量）。
ROOT = Path(os.environ.get("ML_PROJECTS_ROOT", "D:/ML_Projects"))
REPORT_DIR = Path(os.environ.get("RAG_REPORT_DIR", ROOT / "knowledge-graph" / "data" / "reports"))
DB_PATH = Path(os.environ.get("RAG_CHROMA_DB_PATH", ROOT / "rag-finance-qa" / "data" / "chroma_db"))
COLLECTION = "finance_documents"

CHUNK_CHARS = 600
BATCH = 50

SUFFIX_INDUSTRY = "行业深度研究报告"
SUFFIX_COMPANY = "深度研究报告"
SOURCE_TAG = "research_report"


def _extract_sector(title: str, body: str, fallback: str) -> str:
    """尽力从标题/正文里取行业名；取不到就用 fallback。"""
    for pat in (r"（S[HZ]\d{6}）\s*(\S{1,6}?)行业",      # 鹏华电子（SH600012）保险行业深度研究报告
                r"是一家(\S{1,6}?)行业的公司",            # 鹏华电子是一家保险行业的公司
                rf"^(\S{{1,6}}?){SUFFIX_INDUSTRY}"):      # 保险行业深度研究报告
        m = re.search(pat, title + "\n" + body, re.M)
        if m:
            return m.group(1).strip()
    return fallback


def _industry_index(col):
    """从**已入库的公司档案**里取 (code→industry, 全部行业真值)。

    为什么要回查而不是直接用研报自己的 `sector`：`_extract_sector` 抠的是标题里的口语名
    （「保险」「电子」混着来，还有「白酒」这种），而库里存的是 20 个真值标签，
    `where={"industry": ...}` 只认真值 —— 填错等于没填（REQ-049 实测）。
    公司研报按股票代码回查档案最可靠；行业研报没有代码，只在 `sector` 恰好已是真值时才写。
    """
    code2ind, labels = {}, set()
    for m in (col.get(include=["metadatas"])["metadatas"] or []):
        m = m or {}
        ind = m.get("industry")
        if not ind:
            continue
        labels.add(ind)
        if m.get("type") == "overview" and m.get("stock_code"):
            code2ind[m["stock_code"]] = ind
    return code2ind, labels


def parse_report(path: Path):
    """→ (title, body, meta)。metadata 键名对齐全库约定。"""
    text = path.read_text(encoding="utf-8", errors="ignore")

    meta = {"source": SOURCE_TAG, "type": "research_report"}

    # 文件名里的股票代码（公司研报才有）
    m_code = re.search(r"(S[HZ]\d{6})", path.name)
    stock_code = m_code.group(1) if m_code else ""

    # 头部的键值行
    for key, field in [("标题", "title"), ("机构", "org"), ("评级", "rating"), ("日期", "date")]:
        m = re.search(rf"^{key}[:：]\s*(.+)$", text, re.M)
        if m:
            meta[field] = m.group(1).strip()[:80]

    title = meta.get("title", path.stem)
    body = re.sub(r"^(标题|机构|评级|日期)[:：].*$", "", text, flags=re.M).strip()

    # 公司研报 vs 行业研报
    if title.endswith(SUFFIX_INDUSTRY):
        entity = title[: -len(SUFFIX_INDUSTRY)].strip()
        meta["scope"] = "industry"
        meta["sector"] = entity
        # 行业研报不属于任何公司：不写 company/stock_code，避免污染 by_company 与引用
    else:
        entity = title[: -len(SUFFIX_COMPANY)].strip() if title.endswith(SUFFIX_COMPANY) else title.strip()
        meta["scope"] = "company"
        meta["company"] = entity
        if stock_code:
            meta["stock_code"] = stock_code
    meta["sector"] = meta.get("sector") or _extract_sector(title, body, entity)

    return title, body, meta


def chunk_text(body: str):
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units = []
    for p in paras:
        if len(p) <= CHUNK_CHARS:
            units.append(p)
        else:
            for i in range(0, len(p), CHUNK_CHARS):
                units.append(p[i:i + CHUNK_CHARS])
    chunks, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) > CHUNK_CHARS:
            chunks.append(buf)
            buf = u
        else:
            buf = (buf + "\n" + u).strip()
    if buf:
        chunks.append(buf)
    return chunks


def build_docs(files, code2ind=None, labels=None):
    ids, docs, metas = [], [], []
    seen_local = set()
    for f in files:
        title, body, meta = parse_report(f)
        if not body:
            continue
        for i, ch in enumerate(chunk_text(body)):
            h = hashlib.md5(ch.encode("utf-8")).hexdigest()[:12]
            key = meta.get("company") or f"industry_{meta.get('sector', 'x')}"
            cid = f"report_{key}_{h}"
            if cid in seen_local:
                continue
            seen_local.add(cid)
            m = dict(meta)
            # industry 必须写：检索层的行业通道是 `where={"industry":真值}`，研报块过去
            # 只有 sector（口语名，与真值不是一套词表）→ 行业问题结构性捞不到研报。
            ind = (code2ind or {}).get(meta.get("stock_code") or "")
            if not ind and meta.get("sector") in (labels or set()):
                ind = meta["sector"]
            if ind:
                m["industry"] = ind
            m["chunk"] = i
            m["title"] = title
            # 评级可能为空（合成评级已作废，重生成的研报不带评级）；
            # 这种情况直接省略，不要写出「评级 -」这种残句。
            rating = (meta.get("rating") or "").strip()
            rating_bit = f"，评级 {rating}" if rating and rating not in {"-", "N/A", "无"} else ""
            if meta.get("scope") == "company":
                head = f"【研报】{title}（{meta.get('stock_code', '?')}，{meta.get('date', '')}{rating_bit}）"
            else:
                head = f"【行业研报】{title}（{meta.get('date', '')}{rating_bit}）"
            ids.append(cid)
            docs.append(f"{head}\n{ch}")
            metas.append(m)
    return ids, docs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只解析并打印，不写库")
    args = ap.parse_args()

    files = sorted(REPORT_DIR.glob("*.txt"))
    print(f"[1/4] 发现 {len(files)} 份研报")

    col = None
    if args.dry_run:
        # 不碰库：打开 PersistentClient 会推后 sqlite 的 mtime，甚至在路径不存在时把库建出来，
        # 而 --dry-run 的承诺是"只看解析结果"。代价是这里看不到 industry，下面如实标注。
        ids, docs, metas = build_docs(files)
    else:
        col = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION)
        ids, docs, metas = build_docs(files, *_industry_index(col))
    n_co = sum(1 for m in metas if m.get("scope") == "company")
    n_in = sum(1 for m in metas if m.get("scope") == "industry")
    n_ind = sum(1 for m in metas if m.get("industry"))
    print(f"[2/4] 切块完成：{len(docs)} 块（公司研报 {n_co} / 行业研报 {n_in}）｜带 industry {n_ind}")

    if args.dry_run:
        print("\n--dry-run 解析抽样（industry 要读库里档案才能定，此处不解析）--")
        shown = set()
        for m in metas:
            key = (m.get("scope"), m.get("company") or m.get("sector"))
            if key in shown:
                continue
            shown.add(key)
            print(f"  {key}: company={m.get('company', '(无)')} stock_code={m.get('stock_code', '(无)')} "
                  f"sector={m.get('sector')} rating={m.get('rating')}")
            if len(shown) >= 8:
                break
        print(f"  ... 共 {len(shown)} 类")
        return 0

    before = col.count()

    # 先清旧块：旧版元数据错误（stock 而非 stock_code、无 company），改键名后
    # 旧 id 也变了，不清会留下 89 块「影子文档」——同一份研报出现两遍。
    old = col.get(where={"source": SOURCE_TAG}, include=[])["ids"]
    if old:
        col.delete(ids=old)
    print(f"[3/4] 已清理旧研报块 {len(old)} 块（原总数 {before}）")

    added = 0
    for s in range(0, len(ids), BATCH):
        b = range(s, min(s + BATCH, len(ids)))
        col.add(ids=[ids[i] for i in b], documents=[docs[i] for i in b],
                metadatas=[metas[i] for i in b])
        added += len(b)
        print(f"        已写入 {added}/{len(ids)}")

    print(f"[4/4] 完成。集合总数：{col.count()}（清理前 {before}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
