"""
丰富金融数据库：从 Yahoo Finance 获取真实股票数据，写入 MySQL
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pymysql
import requests
import json
from datetime import datetime, timedelta
from config import MYSQL_CONFIG


def get_mysql_connection():
    return pymysql.connect(**MYSQL_CONFIG)


def fetch_yahoo_finance(stock_code, period="3mo"):
    """从 Yahoo Finance API 获取股票数据"""
    # 将 A 股代码转换为 Yahoo Finance 格式
    # SH600001 -> 600001.SS
    code = stock_code.replace("SH", "").replace("SZ", "")
    suffix = ".SS" if stock_code.startswith("SH") else ".SZ"
    ticker = f"{code}{suffix}"
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period": period, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Yahoo Finance 是境外端点：先试直连，失败自动回退 Clash 代理（7897）
    def _get(proxies):
        return requests.get(url, params=params, headers=headers, proxies=proxies, timeout=20)
    try:
        resp = _get({"http": None, "https": None})
    except Exception:
        resp = _get({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
        resp.raise_for_status()
        data = resp.json()
        
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        
        records = []
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts)
            records.append({
                "trade_date": dt.strftime("%Y-%m-%d"),
                "open_price": round(quotes["open"][i], 2) if quotes["open"][i] else 0,
                "close_price": round(quotes["close"][i], 2) if quotes["close"][i] else 0,
                "high_price": round(quotes["high"][i], 2) if quotes["high"][i] else 0,
                "low_price": round(quotes["low"][i], 2) if quotes["low"][i] else 0,
                "volume": int(quotes["volume"][i]) if quotes["volume"][i] else 0,
            })
        
        return records
    except Exception as e:
        print(f"  [WARN] 获取 {ticker} 数据失败: {e}")
        return []


def enrich_stock_prices():
    """丰富股票价格数据"""
    conn = get_mysql_connection()
    cursor = conn.cursor()
    
    # 获取所有公司
    cursor.execute("SELECT id, company_name, stock_code FROM companies")
    companies = cursor.fetchall()
    
    print(f"共 {len(companies)} 家公司，开始获取真实股票数据...\n")
    
    total_new = 0
    for company_id, name, stock_code in companies:
        print(f"  {name}（{stock_code}）...", end=" ")
        
        records = fetch_yahoo_finance(stock_code, period="6mo")
        
        if not records:
            print("跳过（无数据）")
            continue
        
        new_count = 0
        for rec in records:
            try:
                cursor.execute("""
                    INSERT IGNORE INTO stock_prices 
                    (company_id, trade_date, open_price, close_price, high_price, low_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (company_id, rec["trade_date"], rec["open_price"], 
                      rec["close_price"], rec["high_price"], rec["low_price"], rec["volume"]))
                if cursor.rowcount > 0:
                    new_count += 1
            except Exception as e:
                pass
        
        conn.commit()
        total_new += new_count
        print(f"+{new_count} 条")
    
    cursor.close()
    conn.close()
    
    print(f"\n[OK] 新增 {total_new} 条股票数据")
    return total_new


def generate_sample_financial_data():
    """【已废弃】随机示例数据生成器。

    警告：本函数用 random.uniform 基于市值编造财务数字，非真实财报。
    下游 RAG/KG/webapp 引用这些数字会得到不可信结论。
    真实数据请用 shared/collect_financials.py（akshare 东方财富三张表）。
    为防误用污染，本函数写入的行的 is_sample 固定为 1，
    check_artifacts.py 会据此 BLOCK。
    """
    import random
    print("[WARN] generate_sample_financial_data() 已废弃，真实财报请用 shared/collect_financials.py")

    conn = get_mysql_connection()
    cursor = conn.cursor()
    
    # 检查哪些公司缺少财务数据
    cursor.execute("""
        SELECT c.id, c.company_name, c.stock_code, COUNT(fs.id) as stmt_count
        FROM companies c
        LEFT JOIN financial_statements fs ON c.id = fs.company_id
        GROUP BY c.id
        HAVING stmt_count < 4
    """)
    companies = cursor.fetchall()
    
    if not companies:
        print("所有公司已有足够的财务数据")
        cursor.close()
        conn.close()
        return
    
    print(f"\n为 {len(companies)} 家公司补充财务数据...\n")
    
    for company_id, name, stock_code, existing_count in companies:
        # 获取公司市值作为基准
        cursor.execute("SELECT market_cap FROM companies WHERE id = %s", (company_id,))
        row = cursor.fetchone()
        market_cap = float(row[0]) if row and row[0] else 1000000
        
        # 生成4个季度的数据
        for year in [2024, 2025]:
            for quarter in [1, 2, 3, 4]:
                if year == 2025 and quarter > 2:
                    continue
                
                # 基于市值生成合理的财务数据
                base_revenue = market_cap * random.uniform(0.1, 0.3)
                revenue = round(base_revenue * random.uniform(0.8, 1.2), 2)
                net_profit = round(revenue * random.uniform(-0.1, 0.25), 2)
                total_assets = round(market_cap * random.uniform(0.8, 1.5), 2)
                total_liabilities = round(total_assets * random.uniform(0.3, 0.7), 2)
                cash_flow = round(net_profit * random.uniform(0.5, 1.5), 2)
                roe = round(random.uniform(0.03, 0.25), 4)
                debt_ratio = round(total_liabilities / total_assets, 4)
                gross_margin = round(random.uniform(0.15, 0.65), 4)
                net_margin = round(net_profit / revenue if revenue > 0 else 0, 4)
                
                report_date = f"{year}-{quarter*3:02d}-28"
                
                try:
                    cursor.execute("""
                        INSERT IGNORE INTO financial_statements
                        (company_id, report_date, quarter, year, revenue, net_profit,
                         total_assets, total_liabilities, operating_cash_flow,
                         roe, debt_ratio, gross_margin, net_margin, is_sample)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
                    """, (company_id, report_date, quarter, year, revenue, net_profit,
                          total_assets, total_liabilities, cash_flow,
                          roe, debt_ratio, gross_margin, net_margin))
                except:
                    pass
        
        conn.commit()
        print(f"  {name}（{stock_code}）: +{4} 条财务数据")
    
    cursor.close()
    conn.close()
    print("\n[OK] 财务数据补充完成")


if __name__ == "__main__":
    print("=" * 50)
    print("  金融数据库丰富工具")
    print("=" * 50)
    
    # 1. 获取真实股票数据
    print("\n[1/2] 获取 Yahoo Finance 实时股票数据...")
    enrich_stock_prices()
    
    # 2. 补充财务数据
    print("\n[2/2] 补充财务报表数据...")
    generate_sample_financial_data()
    
    # 验证
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stock_prices")
    stock_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM financial_statements")
    stmt_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    print(f"\n[完成] 数据库现状：")
    print(f"  股票价格：{stock_count} 条")
    print(f"  财务报表：{stmt_count} 条")
