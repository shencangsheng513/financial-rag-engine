"""
数据抽取模块：从 MySQL 金融数据库抽取数据，转换为文本块存入 ChromaDB
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pymysql
import chromadb
from config import MYSQL_CONFIG, CHROMA_DB_PATH, COLLECTION_NAME


def get_mysql_connection():
    """获取 MySQL 连接"""
    return pymysql.connect(**MYSQL_CONFIG)


def _fmt(v, nd=2, dash="暂无数据"):
    """格式化可能为 None 的数值。

    财务表里 NULL 很常见（实测 roe 17 条、gross_margin 1327 条、
    operating_cash_flow 240 条为 NULL），旧的 f"{v:.2f}" 会直接抛
    TypeError 让整次重建中断。这里对 None 返回占位符，不臆造 0。
    """
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def _pct(v, nd=2, dash="暂无数据"):
    """百分比字段：库内**已经是百分数**（roe=16.9849 即 16.98%），
    直接补 % 号即可，绝不能再 ×100（那会得到 1698.5%）。"""
    try:
        return f"{float(v):.{nd}f}%"
    except (TypeError, ValueError):
        return dash


def _has_cjk(text):
    """真实域名不会含汉字；出现汉字即是合成档案的铁证（判据同 clean_synthetic_fields.py）。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def extract_company_info(cursor, company_id):
    """抽取单个公司的完整信息（只取可信来源）。

    ⚠️ 2026-09-10 数据质量审计：`credit_ratings` / `transactions` / `employees`
    三张表**只存在于 company_id 1–20**，且内容是合成的。而 id 1–20 在公司改名
    迁移里已被赋成真实公司名（浦发银行/贵州茅台…），于是出现「真实公司名 +
    合成内容」——浦发银行有"设计部3人"、被评 BBB+、对手方"阿里巴巴"。
    这与本项目「宁可缺失，不可错误」的原则冲突，故**不再入库**。
    真正可信的是 financial_statements / stock_prices（已抽样对照真实年报：
    茅台 2024 营收 1741.4 亿、宁德 3620.1 亿，均吻合）。
    """
    # 公司基础信息
    cursor.execute("""
        SELECT company_name, stock_code, industry, founded_year, 
               headquarters, employees_count, market_cap, website
        FROM companies WHERE id = %s
    """, (company_id,))
    company = cursor.fetchone()
    if not company:
        return None
    
    name, stock, industry, founded, hq, emp_count, mcap, website = company

    # 档案字段若来自合成批次（官网域名含汉字），整体作废为 None。
    # 容器内 entrypoint 会调 clean_synthetic_fields.py 改库；但本地重建时库里仍是脏值，
    # 这里再兜一层，保证「谁生成向量块谁负责不含假档案」。
    if _has_cjk(website):
        website = hq = emp_count = founded = None
    
    # 财务报表（最近4期）
    cursor.execute("""
        SELECT report_date, quarter, year, revenue, net_profit, total_assets,
               total_liabilities, operating_cash_flow, roe, debt_ratio, 
               gross_margin, net_margin
        FROM financial_statements 
        WHERE company_id = %s 
        ORDER BY report_date DESC LIMIT 4
    """, (company_id,))
    statements = cursor.fetchall()
    
    # 股票行情（最近30天摘要）
    cursor.execute("""
        SELECT trade_date, open_price, close_price, high_price, low_price, 
               volume, change_pct
        FROM stock_prices 
        WHERE company_id = %s 
        ORDER BY trade_date DESC LIMIT 30
    """, (company_id,))
    stocks = cursor.fetchall()
    
    return {
        "name": name, "stock": stock, "industry": industry,
        "founded": founded, "hq": hq, "emp_count": emp_count,
        "mcap": float(mcap) if mcap else 0, "website": website,
        "statements": statements, "stocks": stocks,
    }


def company_to_text_chunks(info):
    """将公司信息转换为多个文本块（每个块对应一个问题域）"""
    chunks = []
    name = info["name"]
    
    # 1. 公司概览
    # 档案字段可能整体为 None（合成档案被作废）；此时如实写「未收录」，
    # 不要吐出「成立于 None 年」这种残句，更不要臆造。
    profile_bits = []
    if info["founded"] is not None:
        profile_bits.append(f"成立于{info['founded']}年")
    if info["hq"] is not None:
        profile_bits.append(f"总部位于{info['hq']}")
    if info["emp_count"] is not None:
        profile_bits.append(f"员工约{info['emp_count']}人")
    if info["website"] is not None:
        profile_bits.append(f"官网{info['website']}")
    profile_text = ("，".join(profile_bits) + "。") if profile_bits else "公司档案信息暂未收录。"

    chunks.append({
        "id": f"{info['stock']}_overview",
        "text": f"""{name}（{info['stock']}）是一家{info['industry']}行业的公司。{profile_text}
市值约{info['mcap']:.2f}万元。""",
        "metadata": {"type": "overview", "company": name, "stock_code": info["stock"]}
    })
    
    # 2. 财务数据（每期一个块）
    for stmt in info["statements"]:
        report_date, quarter, year, revenue, net_profit, total_assets, \
            total_liabilities, cash_flow, roe, debt_ratio, gross_margin, net_margin = stmt
        
        chunks.append({
            "id": f"{info['stock']}_finance_{year}Q{quarter}",
            "text": f"""{name}（{info['stock']}）{year}年第{quarter}季度财务数据：
营业收入：{_fmt(revenue)}万元，净利润：{_fmt(net_profit)}万元。
总资产：{_fmt(total_assets)}万元，总负债：{_fmt(total_liabilities)}万元。
经营性现金流：{_fmt(cash_flow)}万元。
ROE（净资产收益率）：{_pct(roe)}，资产负债率：{_pct(debt_ratio)}。
毛利率：{_pct(gross_margin)}，净利率：{_pct(net_margin)}。""",
            "metadata": {"type": "financial", "company": name, "stock_code": info["stock"],
                         "year": year, "quarter": quarter}
        })
    
    # 3. 股票行情摘要
    if info["stocks"]:
        prices = [float(s[2]) for s in info["stocks"] if s[2] is not None]  # close_price
        volumes = [int(s[5]) for s in info["stocks"] if s[5] is not None]
        latest = info["stocks"][0]
        
        chunks.append({
            "id": f"{info['stock']}_stock_summary",
            "text": f"""{name}（{info['stock']}）近30个交易日股票行情摘要：
最新收盘价：{_fmt(latest[2])}元（{latest[0]}）。
30日最高价：{_fmt(max(prices))}元，最低价：{_fmt(min(prices))}元。
平均成交量：{sum(volumes)//len(volumes):,}股。
最新涨跌幅：{_fmt(latest[6], 4)}%。""",
            "metadata": {"type": "stock", "company": name, "stock_code": info["stock"]}
        })
    
    # 4/5/6. 信用评级 / 交易记录 / 员工统计：**不再生成**
    # 依据 2026-09-10 数据质量审计：credit_ratings / transactions / employees
    # 三张表只覆盖 company_id 1–20，内容为合成伪造（浦发银行=设计部3人、BBB+ 评级、
    # 对手方阿里巴巴），且这 20 行已被改名成真实公司。入库即等于把假数据挂到真公司头上，
    # 违反「宁可缺失，不可错误」——故整类剔除，相关提问将由模型如实答「未收录」。
    # 详见 clean_synthetic_fields.py 与 diagnose_data_quality.py。

    return chunks


def build_vectordb():
    """从 MySQL 抽取数据，构建 ChromaDB 向量数据库"""
    print("[1/4] 连接 MySQL...")
    conn = get_mysql_connection()
    cursor = conn.cursor()
    
    print("[2/4] 初始化 ChromaDB（内置 Embedding）...")
    
    print("[3/4] 抽取公司数据并生成文本块...")
    cursor.execute("SELECT id FROM companies ORDER BY id")
    company_ids = [row[0] for row in cursor.fetchall()]
    
    all_chunks = []
    for cid in company_ids:
        info = extract_company_info(cursor, cid)
        if info:
            chunks = company_to_text_chunks(info)
            all_chunks.extend(chunks)
            print(f"  {info['name']}（{info['stock']}）: {len(chunks)} 个文本块")
    
    cursor.close()
    conn.close()
    
    print(f"\n[4/4] 写入 ChromaDB（共 {len(all_chunks)} 个文本块）...")
    
    # 初始化 ChromaDB（使用内置 embedding，自动处理向量化）
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 删除旧集合（如果存在）——集合不存在是正常路径；删除失败会在下方 create 时大声失败
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "金融数据库文档向量集合"}
    )
    
    # 批量写入（ChromaDB 自动处理 embedding）
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  已写入 {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
    
    print(f"\n✅ 完成！共写入 {len(all_chunks)} 个文本块到 ChromaDB")
    print(f"   数据库路径：{CHROMA_DB_PATH}")
    
    return collection


if __name__ == "__main__":
    build_vectordb()
