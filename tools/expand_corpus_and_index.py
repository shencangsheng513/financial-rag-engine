# -*- coding: utf-8 -*-
"""向量库与语料库扩充及向量化训练更新脚本。

功能：
1. 扩充公司结构化语料：
   - 从 MySQL (finance_db_v2) 补全缺失的 A 股重点上市公司档案、最新财报与近 30 日行情摘要；
   - 深度回溯多期财报（默认支持到近 8 期财报，即近 2 年完整财务周期），支持多季度同比/环比分析；
2. 扩充金融资讯语料：
   从 MySQL financial_news 抽取精选高价值资讯：
   - 个股研判/新闻（全量覆盖 112 家核心公司及其俗名别名）
   - 宏观经济/产业政策快讯（scope="macro"）
3. 向量编码与增量更新：
   使用 BAAI/bge-small-zh-v1.5 中文嵌入模型计算 512 维向量，
   以批处理方式 upsert 写入生产库 chroma_db_zh。
4. 保证幂等与基线安全：
   - 包含 --dry-run 支持只解析验证不写库
   - 写库前自动创建快照备份
   - 严格遵循 AGENTS.md 规范（无硬编码绝对路径、SQL 参数化、无 eval/shell）

用法：
    python tools/expand_corpus_and_index.py --dry-run
    python tools/expand_corpus_and_index.py --quarters 8 --max-macro-news 600
"""

import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from chromadb.utils import embedding_functions
import pymysql

from config import CHROMA_DB_PATH, COLLECTION_NAME, MYSQL_CONFIG
from src.data_extractor import extract_company_info, company_to_text_chunks, _fmt, _pct
from src.rag_engine_v2 import _CURATED_ALIAS

# 本地模型优先，回退到 HuggingFace / 镜像站模型名称
LOCAL_MODEL_DIR = ROOT.parent / "models" / "bge-small-zh-v1.5"
MODEL_NAME = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5"

MACRO_KEYWORDS = (
    "央行", "货币政策", "降准", "降息", "财政部", "发改委", "美联储", "通胀",
    "GDP", "CPI", "PPI", "PMI", "A股", "北向资金", "资本市场", "房地产",
    "促消费", "新能源汽车", "半导体", "芯片", "人工智能", "大模型",
    "宏观经济", "稳增长", "流动性", "国债", "科创板", "专精特新",
    "低空经济", "商业航天", "工业母机", "高端制造", "机器人", "算力",
    "固态电池", "氢能", "光伏", "风电", "储能", "电网", "外贸", "出口",
    "消费电子", "信创", "自主可控", "新质生产力", "扩大内需",
)


def get_existing_metadata(col):
    """读取库内已有元数据与公司清单"""
    got = col.get(include=["metadatas"])
    metas = got.get("metadatas") or []
    ids = got.get("ids") or []
    existing_companies = set()
    existing_stocks = set()
    existing_ids = set(ids)
    for m in metas:
        m = m or {}
        if m.get("company"):
            existing_companies.add(m["company"])
        if m.get("stock_code"):
            existing_stocks.add(m["stock_code"])
    return {
        "count": len(ids),
        "ids": existing_ids,
        "companies": existing_companies,
        "stocks": existing_stocks,
        "types": Counter(m.get("type", "unknown") for m in metas if m),
    }


def extract_missing_company_chunks(cursor, existing_companies):
    """从 MySQL 提取未入向量库的公司，并切块"""
    cursor.execute("SELECT id, company_name, stock_code, industry FROM companies ORDER BY id")
    all_companies = cursor.fetchall()
    
    missing_companies = [
        row for row in all_companies 
        if row[1] not in existing_companies and row[1].replace(" ", "") not in existing_companies
    ]
    
    print(f"[公司概览] MySQL 中共有 {len(all_companies)} 家公司，"
          f"库内已有 {len(existing_companies)} 家，待补全 {len(missing_companies)} 家公司")
    
    chunks = []
    for cid, cname, scode, ind in missing_companies:
        info = extract_company_info(cursor, cid)
        if not info:
            continue
        c_chunks = company_to_text_chunks(info)
        chunks.extend(c_chunks)
        print(f"  + 抽取 {cname}（{scode}，{ind}）：生成 {len(c_chunks)} 块")
        
    return chunks, missing_companies


def extract_historical_financial_chunks(cursor, all_companies, existing_ids, max_quarters=8):
    """从 MySQL financial_statements 抽取回溯多期财报（扩展到近 max_quarters 期）"""
    print(f"[财务深度] 抽取各公司近 {max_quarters} 期历史财务数据...")
    chunks = []
    n_added_stmts = 0
    for cid, cname, scode, ind in all_companies:
        cursor.execute("""
            SELECT report_date, quarter, year, revenue, net_profit, total_assets,
                   total_liabilities, operating_cash_flow, roe, debt_ratio, 
                   gross_margin, net_margin
            FROM financial_statements 
            WHERE company_id = %s 
            ORDER BY report_date DESC LIMIT %s
        """, (cid, max_quarters))
        statements = cursor.fetchall()
        for stmt in statements:
            report_date, quarter, year, revenue, net_profit, total_assets, \
                total_liabilities, cash_flow, roe, debt_ratio, gross_margin, net_margin = stmt
            chunk_id = f"{scode}_finance_{year}Q{quarter}"
            if chunk_id in existing_ids:
                continue
            
            chunk_text = f"""{cname}（{scode}）{year}年第{quarter}季度财务数据：
营业收入：{_fmt(revenue)}万元，净利润：{_fmt(net_profit)}万元。
总资产：{_fmt(total_assets)}万元，总负债：{_fmt(total_liabilities)}万元。
经营性现金流：{_fmt(cash_flow)}万元。
ROE（净资产收益率）：{_pct(roe)}，资产负债率：{_pct(debt_ratio)}。
毛利率：{_pct(gross_margin)}，净利率：{_pct(net_margin)}。"""
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "type": "financial",
                    "company": cname,
                    "stock_code": scode,
                    "year": year,
                    "quarter": quarter,
                }
            })
            n_added_stmts += 1

    print(f"  + 扩充历史财报块：{n_added_stmts} 条")
    return chunks


def extract_curated_news_chunks(cursor, all_companies, existing_ids, max_macro=600):
    """从 MySQL financial_news 抽取个股新闻与宏观快讯"""
    cursor.execute("""
        SELECT id, title, summary, publish_time, source
        FROM financial_news
        WHERE summary IS NOT NULL AND CHAR_LENGTH(summary) >= 30 AND CHAR_LENGTH(title) >= 5
        ORDER BY publish_time DESC
    """)
    raw_news = cursor.fetchall()
    print(f"[资讯语料] 扫描到 {len(raw_news)} 条有效新闻记录")
    
    # 建立公司匹配字典：归一化名/股票代码尾号/人工俗名 -> (真值名称, 股票代码)
    co_lookup = {}
    for cid, cname, scode, ind in all_companies:
        c_clean = cname.replace(" ", "")
        co_lookup[c_clean] = (cname, scode)
        if len(scode) >= 6:
            co_lookup[scode[2:]] = (cname, scode)
            
    for alias, full in _CURATED_ALIAS.items():
        fn = full.replace(" ", "")
        if fn in co_lookup:
            co_lookup[alias] = co_lookup[fn]
            
    seen_titles = set()
    company_chunks = []
    macro_chunks = []
    
    for nid, title, summary, pub_time, source in raw_news:
        t_clean = (title or "").strip()
        if t_clean in seen_titles:
            continue
        seen_titles.add(t_clean)
        
        s_clean = summary or ""
        full_text = f"{t_clean} {s_clean}"
        pub_date = str(pub_time)[:10] if pub_time else ""
        pub_src = source or "金融资讯"
        
        # 1. 尝试匹配个股
        matched = None
        for key, (co_name, scode) in co_lookup.items():
            if len(key) >= 2 and (key in t_clean or (len(key) >= 4 and key in full_text)):
                matched = (co_name, scode)
                break
                
        if matched:
            co_name, scode = matched
            chunk_id = f"news_{scode}_{nid}"
            if chunk_id not in existing_ids:
                doc_text = f"【新闻】{t_clean}（{pub_src}，{pub_time}）\n{summary}"
                meta = {
                    "type": "news",
                    "source": pub_src,
                    "company": co_name,
                    "stock_code": scode,
                    "date": pub_date,
                }
                company_chunks.append({
                    "id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                })
        else:
            # 2. 尝试匹配宏观经济/政策资讯
            chunk_id = f"news_macro_{nid}"
            if chunk_id not in existing_ids and len(macro_chunks) < max_macro and any(k in t_clean for k in MACRO_KEYWORDS):
                doc_text = f"【宏观快讯】{t_clean}（{pub_src}，{pub_time}）\n{summary}"
                meta = {
                    "type": "news",
                    "source": pub_src,
                    "scope": "macro",
                    "date": pub_date,
                }
                macro_chunks.append({
                    "id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                })
                
    print(f"  + 提炼新个股新闻块：{len(company_chunks)} 条")
    print(f"  + 提炼新宏观资讯块：{len(macro_chunks)} 条")
    return company_chunks + macro_chunks


def backup_database(src_path: Path):
    """备份向量数据库目录"""
    if not src_path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = src_path.parent / f"{src_path.name}.bak_{ts}"
    print(f"[备份] 复制 {src_path} -> {backup_path} ...")
    shutil.copytree(src_path, backup_path)
    print(f"[备份] 完成！")
    return backup_path


def main():
    parser = argparse.ArgumentParser(description="扩充金融向量库与语料库")
    parser.add_argument("--dry-run", action="store_true", help="只做数据抽取与验证，不执行写库与编码")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份现有数据库")
    parser.add_argument("--batch", type=int, default=64, help="Embedding 编码批处理大小")
    parser.add_argument("--quarters", type=int, default=8, help="财务报表回溯季度数（默认8期）")
    parser.add_argument("--max-macro-news", type=int, default=600, help="最多收纳的宏观资讯条数")
    args = parser.parse_args()

    print("=" * 65)
    print("  RAG 金融问答系统 — 语料库扩充与向量数据库增量训练更新")
    print("=" * 65)
    print(f"ChromaDB 路径: {CHROMA_DB_PATH}")
    print(f"Collection 名: {COLLECTION_NAME}")
    print(f"Embedding 模型: {MODEL_NAME}")
    
    # 1. 检查 ChromaDB 现状
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    meta_info = get_existing_metadata(col)
    print(f"\n[1/5] 当前库内状态:")
    print(f"  总文档块: {meta_info['count']} 条")
    print(f"  覆盖公司: {len(meta_info['companies'])} 家")
    print(f"  类型分布: {dict(meta_info['types'])}")
    
    # 2. 连接 MySQL 抽取语料
    print("\n[2/5] 连接 MySQL 抽取扩充语料...")
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, company_name, stock_code, industry FROM companies ORDER BY id")
    all_companies = cursor.fetchall()
    
    missing_company_chunks, missing_cos = extract_missing_company_chunks(cursor, meta_info["companies"])
    historical_fin_chunks = extract_historical_financial_chunks(cursor, all_companies, meta_info["ids"], max_quarters=args.quarters)
    news_chunks = extract_curated_news_chunks(cursor, all_companies, meta_info["ids"], max_macro=args.max_macro_news)
    
    cursor.close()
    conn.close()
    
    candidate_chunks = missing_company_chunks + historical_fin_chunks + news_chunks
    all_new_chunks = [c for c in candidate_chunks if c["id"] not in meta_info["ids"]]
    new_types = Counter(c["metadata"]["type"] for c in all_new_chunks)
    print(f"\n[3/5] 新增语料块聚合完成: 共 {len(all_new_chunks)} 条")
    print(f"  新增类型分布: {dict(new_types)}")
    if all_new_chunks:
        print("  抽样示例:")
        for c in all_new_chunks[:3]:
            txt_preview = c['text'].replace('\n', ' ')[:90]
            print(f"    - [{c['id']}] {txt_preview}...")

    if args.dry_run:
        print("\n[dry-run] 运行模式为只读检查，未对向量数据库做任何更改。退出。")
        return 0

    if not all_new_chunks:
        print("\n[OK] 没有需要新增的文本块，数据库已是最新。")
        return 0

    # 3. 备份现有数据库
    if not args.no_backup:
        backup_database(Path(CHROMA_DB_PATH))

    # 4. 加载 Embedding 模型进行向量化编码
    print(f"\n[4/5] 加载 Embedding 模型 ({MODEL_NAME}) 进行向量编码...")
    t0 = time.time()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    print(f"  模型加载就绪，耗时 {time.time() - t0:.1f}s")
    
    texts = [c["text"] for c in all_new_chunks]
    t1 = time.time()
    embeddings = ef(texts)
    print(f"  编码完成：共 {len(embeddings)} 条向量（维度 {len(embeddings[0]) if embeddings else '?'}），"
          f"耗时 {time.time() - t1:.1f}s")
    
    # 5. 批量写入 ChromaDB
    print(f"\n[5/5] 批量 upsert 写入 ChromaDB...")
    B = args.batch
    total = len(all_new_chunks)
    for i in range(0, total, B):
        batch = all_new_chunks[i:i + B]
        b_ids = [c["id"] for c in batch]
        b_docs = [c["text"] for c in batch]
        b_metas = [c["metadata"] for c in batch]
        b_embs = embeddings[i:i + B]
        col.upsert(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_metas,
            embeddings=b_embs,
        )
        print(f"  已写入 {min(i + B, total)} / {total}")
        
    final_info = get_existing_metadata(col)
    print("\n" + "=" * 65)
    print("[OK] 语料扩充与向量数据库更新完成！")
    print(f"  原文档数: {meta_info['count']} -> 现文档数: {final_info['count']}（净增 {final_info['count'] - meta_info['count']}）")
    print(f"  原公司数: {len(meta_info['companies'])} -> 现公司数: {len(final_info['companies'])}")
    print(f"  更新后类型分布: {dict(final_info['types'])}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
