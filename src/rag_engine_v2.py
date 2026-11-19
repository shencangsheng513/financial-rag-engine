"""
RAG 问答引擎 v2：混合检索 + 重排序
优化召回率从 <30% 提升到 70%+
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
import requests
import json
import re
import time
from collections import defaultdict

# 加载 .env 文件
# .env 查找顺序：本目录（旧布局兼容）→ 项目根 moml-cloud/.env。
# 2026-09-11：本目录的 .env 已移出——它会被整包打进上传平台的 code.zip
# （.dockerignore 只挡镜像、不挡 zip），明文 Key 不应随源码上云。
# 云上由平台 EnvParams 注入 MODELSCOPE_API_KEY，这里 setdefault 只是本地兜底。
_env_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))), ".env"),
]
_env_path = next((p for p in _env_candidates if os.path.exists(p)), None)
if _env_path:
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, MAX_CONTEXT_LENGTH
from config import LLM_API_BASE, LLM_MODEL, LLM_MAX_TOKENS, PROXY
from src.trace_logger import trace_retrieval, log_generation


# 文档类型优先级：问某家公司时，档案(overview)通常是最该先看的，财务明细次之。
# 数值越小越靠前，未知类型排最后。作用是让关键词检索的结果顺序**可预测**，
# 而不是取决于 ChromaDB 的返回次序。
_DOC_TYPE_PRIORITY = {
    "concept": 0,
    "overview": 0, "financial": 1, "rating": 2,
    "stock": 3, "transaction": 4, "employee": 5,
    # 2026-09-11：研报块重新入库（+112 块）后必须给它确定的位置，
    # 否则落进 .get(t, 99) 永远垫底——实体池里茅台有 ~80 块 financial/news，
    # 研报块永远挤不进 top-n，等于白灌。
    "research_report": 2,
}

# 金融基础常识/专业概念意图（2026-11-09 扩充）。
# 用户常提问「什么是市盈率」「ROE是什么意思」「怎么计算资产负债率」「什么是做多做空」等通用概念，
# 此类问句不含具体公司实体，需将 type=concept 的常识定义块前置并加分。
_CONCEPT_HINTS = (
    "什么是", "是什么意思", "怎么理解", "如何理解", "怎么算", "怎么计算",
    "计算公式", "有什么区别", "是什么概念", "概念", "定义", "含义",
    "公式", "的区别", "代表什么", "说明什么", "怎么看", "通俗解释",
    "有哪些规则", "交易规则", "如何做空", "做多做空", "基础知识", "基础常识",
)

# query 里出现这些词，说明用户在问「消息面」而不是「财务面」，此时 news 类文档要提到最前。
# 否则 news 在 _DOC_TYPE_PRIORITY 里没有位置、永远垫底 ——
# 实测问「宁德时代最近有什么新闻」会返回公司档案和财报，答非所问。
_NEWS_HINTS = ("新闻", "消息", "公告", "动态", "报道", "舆情", "发生了什么")
_STOCK_HINTS = ("股价", "走势", "收盘", "涨跌", "成交量", "行情", "K线", "最高价", "最低价")
# 同理：问「公司档案类」问题时 overview 要提到最前。overview 按设计就是一句话
# （"X（代码）是Y行业的A股上市公司，最新市值约N亿元"），文本短、金融词少，
# 在纯打分排序里必然输给动辄几百字的 financial —— 实测「贵州茅台市值」的 top-5
# 全是财报，档案里的真实市值永远召不回。故显式给 overview 一个意图位。
_PROFILE_HINTS = ("市值", "总市值", "流通市值", "上市", "简介", "主营业务",
                 "主营", "是做什么", "什么行业", "哪个行业", "所属行业", "公司档案", "属于什么")
# 注意：**不要收裸"行业"** —— "银行行业最近有什么新闻" 里含"行业"，
# 一旦命中就会把消息面问题判成档案问题，把 news 全挤掉（实测踩过）。
# 但**疑问词版"哪个行业"可以收**（2026-09-13 补）：A06「东方财富是哪个行业的公司」
# 用"哪个行业"，原集合只收了"什么行业/所属行业"漏了它，导致 overview 不前置、A06 判 miss。
# "哪个行业" 不会出现在「银行行业最近有什么新闻」「X行业研报观点」这类碰撞句式里，
# 且本规则已有 `not _NEWS_HINTS` 兜底，故加它零碰撞风险。
# 同理：问「研报/报告说了什么」时 research_report 要提到最前（对齐 rag_engine.py v1 的 _REPORT_HINTS）。
_REPORT_HINTS = ("研报", "报告", "研究", "解读", "观点", "点评", "深度")

# 宏观/国际资讯意图（2026-09-11）。
#
# 背景：`type=news` 里混着两类东西 —— 558 块**宏观/国际新闻**（无公司归属，
# `stock_code` 为空、`company` 存的是新闻来源）与 1335 块**个股新闻**。
# 它们现在用 `scope="macro"` 区分（见 tools/build_corpus.py 与 tools/tag_macro_news.py）。
#
# 为什么要区分：宏观块对"某家公司最近有什么新闻"是**噪声**。029 版本问
# 「茅台最近有什么新闻」返回市场资讯/新华财经/央视新闻，一条茅台新闻都没有，
# 就是因为实体通道没兜住时向量把这批宏观块捞了上来。
# 但对"最近国际油价怎么样"这类问题，它们恰恰是**唯一**该给的答案。
#
# 所以：命中本表 ⇒ 宏观块前置；没命中 ⇒ 宏观块压后（但**不删除**，
# 用户点名了来源时仍按常规处理）。见 HybridRetriever._macro_rank。
#
# ⚠️ 词表必须**具体**。绝不能放"中国/经济/市场"这类词 ——
#    「中国平安最近有什么新闻」含"中国"，一放就把公司问题判成宏观问题。
_MACRO_HINTS = ("宏观", "国际", "全球", "海外", "美联储", "加息", "降息", "美债",
                "美股", "纳斯达克", "标普", "道琼斯", "欧股", "日经",
                "原油", "油价", "布伦特", "汇率", "美元指数", "大宗商品",
                "地缘", "关税", "俄乌", "中东", "欧佩克")

# 6g 单位可答性用（见 Reranker._compute_score）。查询里"多少X"的 X 与块内
# "数字+单位"配对；单位形态要容忍 Popo 排版的内嵌空格（"44 . 7656 厘 米"、"c m"）。
_QUERY_UNIT_RE = re.compile(r"多少\s*([^\s？?。，,!！;；]{1,6})")
_UNIT_SYNONYMS = {
    "厘米": r"厘\s*米|公分|c\s*\.?\s*m",
    "公分": r"公分|厘\s*米|c\s*\.?\s*m",
    "毫米": r"毫\s*米|m\s*m",
    "千米": r"千米|公里|k\s*m",
    "米": r"米|m",
    "公斤": r"公斤|k\s*g",
    "吨": r"吨",
    "度": r"度|°",
    "亿元": r"亿\s*元|亿",
    "万元": r"万\s*元|万",
    "元": r"元|￥",
    "个百分点": r"个?\s*百\s*分\s*点|个点",
}


def _query_unit_pattern(query):
    m = _QUERY_UNIT_RE.search(query)
    if not m:
        return None
    unit = m.group(1)
    return _UNIT_SYNONYMS.get(unit, re.escape(unit))

# ===== 结构化聚合 / 排行（2026-09-11）=====
# 为什么需要：向量检索**无法排序**。"哪家公司ROE最高" 本质是 `ORDER BY roe DESC LIMIT 1`，
# 而检索只能返回"和这句话最像的文档"，实测会返回长江电力/云南白药这类不相关公司的档案。
# 这是**架构缺口，不是检索 bug** —— 修词法修不动它。
#
# 为什么用向量库当数据源而不是 MySQL：
#   - rag 服务根本没有数据库连接（本文件里没有一处 mysql/db 引用），接 SQL 要新增依赖与配置；
#   - type=financial 的块正文是**模板化结构化句**（见 _FIN_HEAD / 各指标 regex），可稳定解析；
#   - **绕开单位陷阱**：向量库里的数字已是「亿元 / %」，而 MySQL 的 financial_statements.*
#     是万元、roe 等已是百分数，接 SQL 就得逐列对单位，极易出错；
#   - **检索与聚合同源**：引用的数字不可能和排序用的数字打架。
_AGG_DESC = ("最高", "最大", "最多", "最强", "最好", "最赚", "赚钱", "比较高", "偏高",
             "较高", "很高", "排名", "排行", "前列", "前几", "top", "居前")
_AGG_ASC = ("最低", "最小", "最少", "最差", "比较低", "偏低", "较低", "很小")
# 指标说法 -> 内部键。匹配时**按词长降序**，否则"利润"会抢先命中"净利润"、"负债率"
# 会从"资产负债率"里切走一段（两者含义相同，但长词优先让日志里的命中原由更清楚）。
_METRIC_WORDS = {
    "净资产收益率": "roe", "roe": "roe", "股东回报": "roe",
    "资产负债率": "debt_ratio", "负债率": "debt_ratio", "杠杆率": "debt_ratio",
    "净利润": "net_profit", "赚钱": "net_profit", "最赚": "net_profit",
    "盈利": "net_profit", "利润": "net_profit",
    "毛利率": "gross_margin",
    "净利率": "net_margin",
    "营业收入": "revenue", "营收": "revenue", "收入": "revenue",
    "总资产": "total_assets", "资产规模": "total_assets",
    "市值": "market_cap",
}
_METRIC_META = {
    "roe":          ("ROE", "%"),
    "debt_ratio":   ("资产负债率", "%"),
    "net_profit":   ("净利润", "亿元"),
    "gross_margin": ("毛利率", "%"),
    "net_margin":   ("净利率", "%"),
    "revenue":      ("营业收入", "亿元"),
    "total_assets": ("总资产", "亿元"),
    "market_cap":   ("市值", "亿元"),
}
# 口语 -> 库内 industry **真值**。取值必须来自库里真实存在的 20 种（见 _rag_eval 的行业清单）：
# 银行/电子/非银金融/计算机/医药生物/机械设备/食品饮料/电力设备/房地产\/建筑/家用电器/
# 基础化工/公用事业/交通运输/传媒/通信/建筑材料/有色金属/商贸零售/汽车/农林牧渔
_INDUSTRY_WORDS = {
    "白酒": "食品饮料", "啤酒": "食品饮料", "饮料": "食品饮料", "食品": "食品饮料",
    "酿酒": "食品饮料", "酒": "食品饮料",
    "银行": "银行",
    "券商": "非银金融", "证券": "非银金融", "保险": "非银金融", "非银": "非银金融",
    "半导体": "电子", "芯片": "电子", "消费电子": "电子", "面板": "电子",
    "家电": "家用电器", "白电": "家用电器",
    "医药": "医药生物", "生物医药": "医药生物", "制药": "医药生物", "创新药": "医药生物",
    "计算机": "计算机", "软件": "计算机", "云计算": "计算机",
    "新能源": "电力设备", "光伏": "电力设备", "锂电": "电力设备", "电池": "电力设备",
    "储能": "电力设备",
    "地产": "房地产/建筑", "房地产": "房地产/建筑", "建筑": "房地产/建筑",
    "化工": "基础化工",
    "机械": "机械设备", "工程机械": "机械设备",
    "电力": "公用事业", "水电": "公用事业", "燃气": "公用事业", "公用事业": "公用事业",
    "交通": "交通运输", "航运": "交通运输", "港口": "交通运输",
    "物流": "交通运输", "快递": "交通运输",
    "传媒": "传媒", "游戏": "传媒",
    "通信": "通信", "运营商": "通信",
    "建材": "建筑材料", "水泥": "建筑材料",
    "有色": "有色金属", "黄金": "有色金属", "铜": "有色金属",
    "零售": "商贸零售", "超市": "商贸零售", "商贸": "商贸零售",
    "汽车": "汽车", "整车": "汽车",
    "农业": "农林牧渔", "养殖": "农林牧渔", "饲料": "农林牧渔",
}
# 「行业 → 公司清单」分支的**意图词**（2026-09-24 REQ-050）。
# 为什么不能只看行业词：线上那两条坏题里 `锂电池行业怎么样？`（13 轮）和
# `白酒行业最近有什么新闻` 都带行业词，但它们要的不是名单 —— 只看行业词会把
# 概览题/新闻题抢进名单分支（工单 §三 反向线明令禁止）。所以触发面 =
# **行业词 AND 意图词**，且不带排行方向词（那是 D 类，归 structured_rank）。
# 全部按 _norm 后的形态写（无空格、小写），匹配用子串。
_ROSTER_INTENTS = ("哪些公司", "哪几家公司", "有哪些公司", "有什么公司", "哪些企业",
                   "哪些上市公司", "名单", "列表", "清单", "包含哪些", "覆盖哪些",
                   "成分股", "哪些股票", "有几家", "多少家", "旗下公司")

# financial 块头：`贵州茅台（SH600519）2026年第2季度财务数据：…`
# 注意正文里是全角括号，而 _norm() 会把全角转半角 —— 这里是**解析原文**，故两种都接受。
_FIN_HEAD = re.compile(r"^(?P<co>.+?)[（(](?P<code>[A-Z]{2}\d{6})[）)]"
                       r"(?P<y>\d{4})年第(?P<q>\d)季度财务数据")
# overview 块头：`浦发银行（SH600000）是银行行业的A股上市公司。最新市值约 3074.13 亿元。`
_OVW_HEAD = re.compile(r"^(?P<co>.+?)[（(](?P<code>[A-Z]{2}\d{6})[）)]")
_OVW_CAP = re.compile(r"市值约\s*([\d.]+)\s*(万元|亿元)")
# 行业：两种块的 metadata **都没有** industry 字段（2026-09-12 实测），只能从档案正文抽。
# 正文形如「浦发银行（SH600000）是银行行业的A股上市公司」→ 抽"银行"。
# 抽不到就是 None，行业限定会被跳过（宁可不限行业，也不要把 rows 过滤成空导致整个聚合退化）。
_OVW_IND = re.compile(r"是(?:一家)?(.+?)行业的")
# 指标正文取值：模板为「<指标名>：<数值><单位>」。
# ⚠️ 2026-09-12 修：旧模板是「<指标名> <数值>」（无冒号、无全角括号），语料改版后正文变成
#    「ROE（净资产收益率）：3.68%」「营业收入：9377700.00万元」，旧正则 `name\s*(-?[\d.]+)`
#    在冒号/括号处**全部匹配失败** → 指标表 92 家公司 8 个指标全 None →
#    structured_rank 恒返回 [] → 所有排行问题静默退化成普通检索（问"市值最高"返回长春高新）。
#    第三列 = 该指标是否为**金额**（正文单位万元，表内统一折算成亿元）。
_FIN_FIELDS = (("revenue", "营业收入", True), ("net_profit", "净利润", True),
               ("total_assets", "总资产", True), ("total_liab", "总负债", True),
               ("cashflow", "经营性现金流", True), ("roe", "ROE", False),
               ("debt_ratio", "资产负债率", False), ("gross_margin", "毛利率", False),
               ("net_margin", "净利率", False))


def _fin_num(text, name, is_amount=False):
    """取 `<name>：<数值>` 里的数字，金额按正文单位折算成**亿元**。

    兼容两种模板：旧「ROE 3.68」/ 新「ROE（净资产收益率）：3.68%」。
    值缺失时（「毛利率：暂无数据」）返回 None —— 冒号后不是数字就整体不匹配，
    **不会**顺延去抓下一个指标的数字（这是 `[^0-9-]{0,n}` 那种宽正则会犯的错）。
    """
    m = re.search(re.escape(name) + r"(?:\s*[（(][^）)]*[）)])?\s*[：:]?\s*"
                  r"(-?[\d.]+)\s*(万元|亿元|%)?", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if is_amount and m.group(2) == "万元":
        v = v / 10000.0
    return v


def _parse_financial(text):
    """解析模板化财务句。不认识就返回 None（不抛）—— 语料改版时只是退化成不聚合。"""
    if not text:
        return None
    head = _FIN_HEAD.match(text)
    if not head:
        return None
    row = {"company": head.group("co").strip(), "stock_code": head.group("code"),
           "year": int(head.group("y")), "quarter": int(head.group("q"))}
    for key, name, is_amount in _FIN_FIELDS:
        row[key] = _fin_num(text, name, is_amount)
    return row


def _parse_overview(text):
    """从公司档案里取市值（亿元）。只有市值可聚合，其余指标不在档案里。"""
    if not text:
        return None
    head = _OVW_HEAD.match(text)
    if not head:
        return None
    cap_m = _OVW_CAP.search(text)
    cap = None
    if cap_m:
        try:
            cap = float(cap_m.group(1))
            if cap_m.group(2) == "万元":   # 表内统一亿元
                cap = cap / 10000.0
        except ValueError:
            cap = None
    ind_m = _OVW_IND.search(text)
    return {"company": head.group("co").strip(), "stock_code": head.group("code"),
            "industry": ind_m.group(1).strip() if ind_m else None,
            "market_cap": cap}



# ===== 查询归一化 + 实体简称（2026-09-11 评测驱动）=====
# 为什么要归一化：库里 `company` 存在全角（"京东方Ａ"）、含空格（"五 粮 液"）等变体，
# 而用户输入一律半角无空格。不对齐就永远匹配不上（评测里「京东方最近有什么新闻」
# 直接答成普京免签新闻）。
_FW2HW = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}


def _norm(s):
    """全角→半角、去所有空白、小写。中文字符不受影响。"""
    s = "".join(_FW2HW.get(ch, ch) for ch in (s or ""))
    return re.sub(r"\s+", "", s).lower()


def _shares_fragment(a, b):
    """a 是否含某个 ≥2 字连续片段也出现在 b 中（字根混淆判据，2026-09-13）。

    用途：重排序反混淆。向量兜底把「长三角」错配「长江电力」时，
    a=长江电力 与 b=长三角 共享「长江」这一 2 字片段。返回 True 即命中签名。
    复杂度 O(len(a))，足够便宜。
    """
    for i in range(len(a) - 1):
        if a[i:i + 2] in b:
            return True
    return False


# 派生简称时允许剥掉的地名前缀（"贵州茅台"→"茅台"、"泸州老窖"→"老窖"）
_PLACE_PREFIX = ("贵州", "山西", "泸州", "云南", "上海", "北京", "深圳", "宁波", "南京",
                 "杭州", "青岛", "天津", "重庆", "广州", "四川", "山东", "河南", "湖南",
                 "湖北", "陕西", "福建", "安徽", "江苏", "浙江", "新疆", "辽宁", "吉林",
                 "黑龙", "江西", "广西", "内蒙", "甘肃", "青海", "宁夏", "海南")
# 允许剥掉的通用后缀（"万华化学"→"万华"、"招商银行"→"招商"）
_GENERIC_SUFFIX = ("电器", "医药", "生物", "化学", "网络", "水泥", "重工", "电力", "传媒",
                   "激光", "电气", "控股", "集团", "股份", "科技", "软件", "材料", "能源",
                   "机械", "汽车", "银行", "证券", "保险", "地产", "发展", "航运", "港口",
                   "机场", "高速", "钢铁", "有色", "矿业", "石化", "电子", "通信", "光电",
                   "精密", "装备", "环保", "水务", "燃气", "超媒", "家居", "食品", "乳业",
                   "酒业", "药业", "自动化", "股份")
# 剥完可能得到"太泛"的词 —— 它们会出现在任意问句里，一律不要
_TOO_GENERIC = {"工业", "银行", "证券", "保险", "中国", "上海", "北京", "深圳", "广州",
                "东方", "数据", "市场", "资讯", "财经", "新闻", "科技", "中信", "光大",
                "建设", "农业", "工商", "交通", "民生", "投资", "国际", "中心", "研究",
                "发展", "集团", "股份", "电力", "能源", "医药", "电子", "通信",
                # 2026-09-11 反向检查抓到：这几个是**地理/普通词**，不是公司简称。
                # 实测「长江三角洲有哪些政策支持」被错配到长江电力、
                # 「民生问题最近有什么新闻」被错配到民生银行、"平安"在平安银行与
                # 中国平安之间真有歧义。别名机制一律**精确率优先**：宁可退回向量检索，
                # 也不猜。故把这些词一并挡掉，并保留 H_反向检查 类目做回归。
                "长江", "黄河", "珠江", "平安", "华南", "华北", "华东", "西南", "东北",
                # 2026-09-11 别名表全量审计（_rag_eval/audit_aliases.py）追加。
                # 该脚本对**每一条**派生别名统计「正文命中 vs 本公司命中」，按他司占比排序：
                #   中国人民 → 中国人民银行   他司 8/8   = 100%（央行，不是"公司"）
                #   中航     → 中航光电       他司 108/206 = 52%
                # 其余 54 条采纳别名里，除这两条最高的也只有 19%（招商），故只挡这两条。
                "中国人民", "中航"}
# 媒体来源闸门（2026-09-11 审计）：只用于**阻断派生**，不参与精确匹配。
# 背景：库里 439 条 news 的 `company` 存的是**新闻来源**（新浪财经/北京商报/上海证券报…），
# 不是被报道的公司。剥离式规则会尽职地把"北京商报"剥成"商报"、"广州日报"剥成"日报"、
# "上海证券报"剥成"证券报"，把**报纸名**注册成公司简称 —— 于是「日报上有什么消息」
# 这类问句会把新闻来源当成实体锚点。审计实测这四条别名他司占比 71%~100%。
# 为什么用"报"这一个字当判据：92 家 A 股公司名里**没有一家**含"报"，
# 而报纸/证券报/客户端都含 —— 足够精确，且不会误伤（网络/资讯/财经 这类字不能用，
# 会连带干掉"亿联网络""用友网络"等真公司名，故不按它们过滤）。
# 注意只挡派生：`idx` 里仍保留"上海证券报"这个精确表面形式，
# 用户真问「上海证券报说了什么」时照样能命中它名下的块。
_MEDIA_MARK = "报"
# 常用俗名白名单。这里显式列出的**不受**上面 _TOO_GENERIC 黑名单约束 ——
# 黑名单是给"规则自动派生"兜底的保守闸门，而这张表是人工确认过、指向唯一的。
_CURATED_ALIAS = {
    "宁王": "宁德时代", "招行": "招商银行", "工行": "工商银行", "农行": "农业银行",
    "建行": "建设银行", "浦发": "浦发银行", "兴业": "兴业银行", "茅台": "贵州茅台",
    "汾酒": "山西汾酒", "老窖": "泸州老窖", "洋河": "洋河股份", "京东方": "京东方Ａ",
    "五粮液": "五粮液", "联通": "中国联通", "保利": "保利发展", "中免": "中国中免",
    "顺丰": "顺丰控股", "隆基": "隆基绿能", "亿纬": "亿纬锂能", "赣锋": "赣锋锂业",
    "紫金": "紫金矿业", "美的": "美的集团", "格力": "格力电器", "海尔": "海尔智家",
    "潍柴": "潍柴动力",
    # 2026-09-24 REQ-047 全量语料审计通过的 9 条两字简称（他司占比均 <30%）。
    # `宁德` 审计值 30.5%，**越过本表自己的 30% 闸门**（亿纬锂能等正文大量提到它），
    # 有意没注册 —— 要放行得人工点头，别当漏项顺手补上。
    "海康": "海康威视", "迈瑞": "迈瑞医疗", "药明": "药明康德", "海天": "海天味业",
    "汇川": "汇川技术", "天齐": "天齐锂业", "紫光": "紫光国微", "科大": "科大讯飞",
    "华友": "华友钴业",
}


class HybridRetriever:
    """混合检索器：向量检索 + 关键词检索"""
    
    def __init__(self, collection):
        self.collection = collection
        self._entity_cache = None  # 知识库已知实体名，首次使用时惰性加载
        self._metrics_cache = None  # 聚合指标表，首次用到排行问题时惰性构建
        self._roster_cache = None  # 行业 -> 收录公司清单，首次用到"有哪些公司"时惰性构建
        self._concept_cache = None  # 金融基础常识/专业概念词表，首次检索概念时惰性构建
        # 行业俗称接线引入的标签 / 公司原始取值，都在 _known_entities() 里填，见那里的闸门注释
        self._wired_industries = set()
        self._company_origins = set()
        # 金融领域停用词
        self.stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "它", "们", "那", "些", "什么", "吗", "呢", "吧", "啊", "哦",
            "吗", "请问", "告诉", "分析", "查询", "哪些", "多少", "怎么",
        }
    
    def vector_search(self, query, n_results=20):
        """向量语义检索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "vector_score": 1 / (1 + results["distances"][0][i]),  # 转换为0-1分数
            })
        return docs
    
    def keyword_search(self, query, n_results=20):
        """关键词精确检索：元数据精确匹配 + 正文包含匹配。

        三个必须记住的坑（2026-09-10 实测）：

        1) **候选池不能等于 n_results。** ChromaDB 的 `get()` 返回顺序是任意的
           （按内部 id 排），若把 limit 设成 n_results，后面再按 type 排序就只是
           在「一小撮碰巧排在前面的文档」里排序。实测问「宁德时代最近有什么
           新闻」，`get(limit=6)` 恰好只回 overview+financial，15 篇 news 全被
           截在池外，`_type_key` 再正确也没用。

        2) **metadata 上的 `$contains` 不报错，但永远 0 命中。** 实测（2026-09-10，
           chromadb 1.5.9）：`where={"company": {"$contains": "宁德"}}` 返回 0 条，
           而 `where={"company": "宁德时代"}` 正常返回；`$and` 里套 `$contains` 同样 0。
           也就是说这个写法**不是语法错**，而是静默失效 —— 比抛异常更坑，
           因为裸 except 连日志都不会留。原实现三个 metadata 分支因此从未生效。
           这里改成精确匹配 `{"company": kw}`（kw 来自 query 中匹配到的实体名，
           与库内 `company` 值天然相等，用 `$eq` 语义正合适）。

        3) **顺序要按「是否真的关于这家公司」优先，而不是只按文档类型。**
           「宁德时代最近有什么新闻」若只按 type 排序，会返回 隆基绿能/中国平安
           的新闻 —— 它们仅在正文里提到宁德时代。所以 metadata 精确命中的文档
           先出，正文 $contains 命中的后出。
        """
        keywords = self._extract_keywords(query)
        _news_intent = any(h in query for h in _NEWS_HINTS)
        _stock_intent = any(h in query for h in _STOCK_HINTS)
        _report_intent = any(h in query for h in _REPORT_HINTS)
        _profile_intent = any(h in query for h in _PROFILE_HINTS)
        _macro_intent = any(h in query for h in _MACRO_HINTS)
        _concept_intent = any(h in query for h in _CONCEPT_HINTS)

        # 候选池要足够大，理由见 docstring 第 1 点
        pool = max(500, n_results * 20)

        docs = []

        def _collect(results, score, entity=False):
            if not results or not results.get("ids"):
                return
            texts = results.get("documents") or []
            metas = results.get("metadatas") or []
            for i, doc_id in enumerate(results["ids"]):
                docs.append({
                    "id": doc_id,
                    "text": texts[i] if i < len(texts) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    # 关键词腿没有向量距离。用 0 表达"没有"是错的：下游 `1 - distance`
                    # 会把它读成"完全相似"（徽标恒 100.0%），而 `_rrf_fuse` 合并两腿
                    # 恰恰靠 `distance is None` 判断"这条还没被向量腿填过"。
                    "distance": None,
                    "keyword_score": score,
                    "_entity": entity,
                })

        for kw in keywords:
            # 1) 元数据精确匹配（company / stock_code / industry）—— 最可信
            for field in ("company", "stock_code", "industry"):
                try:
                    _collect(self.collection.get(
                        where={field: kw},
                        include=["documents", "metadatas"],
                        limit=pool,
                    ), 1.0, entity=True)
                except Exception:
                    pass

            # 2) 文档正文包含关键词 —— 可能只是「提到」，可信度次之
            try:
                _collect(self.collection.get(
                    where_document={"$contains": kw},
                    include=["documents", "metadatas"],
                    limit=pool,
                ), 0.6, entity=False)
            except Exception:
                pass

            # 3) 意图定向补充：问「新闻」时，额外把 **type=news 且正文含关键词** 的块拉进池子。
            #
            # 为什么必须单独查一次（2026-09-11 实测）：
            #   `where_document` 的 limit 是**按存储顺序**截断的，不是按相关性。
            #   问「银行行业最近有什么新闻」时，正文含「银行」的块共 1030 个，而
            #   pool=500 先被 financial/overview 占满——1893 块 news 里含「银行」的
            #   183 块一条都挤不进来（实测 limit=200 样本里 news 数为 0）。
            #   于是无论排序键怎么写意图，news 都无从出场，答案只能答财务。
            #   加上 where={"type":"news"} 与正文条件求交，news 才有稳定的入场通道。
            if _news_intent:
                try:
                    _collect(self.collection.get(
                        where={"type": "news"},
                        where_document={"$contains": kw},
                        include=["documents", "metadatas"],
                        limit=pool,
                    ), 0.9, entity=False)
                except Exception:
                    pass

            # 4) 概念意图定向补充：问概念或常识时，将 type=concept 的常识块加入候选池
            if _concept_intent or any(kw in c for c in self._known_concepts()):
                try:
                    _collect(self.collection.get(
                        where={"type": "concept"},
                        where_document={"$contains": kw},
                        include=["documents", "metadatas"],
                        limit=pool,
                    ), 1.0, entity=True)
                except Exception:
                    pass

        # 去重
        seen_ids = set()
        unique_docs = []
        for doc in docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                unique_docs.append(doc)

        # 排序：有明确意图时「意图优先于实体」；无意图时「实体命中 > 类型优先级」。
        #
        # 为什么意图要压过实体（2026-09-11 实测踩坑）：
        #   问「银行行业最近有什么新闻」时，_collect 会先按 `where={"industry":"银行"}`
        #   命中一大批 financial 块（financial 元数据带 industry），标为 ent=0；
        #   而 news 块的元数据**不含 industry**，只能从 `where_document` 进来、标为 ent=1，
        #   于是 183 块含「银行」的新闻全被 financial 实体命中压住，答非所问。
        #   与上面 _NEWS_HINTS 的注释（"news 要提到最前"）一致的做法是：先看意图。
        # 无意图时保持原序（实体命中优先），避免影响常规财务提问。
        def _macro_rank(d):
            """宏观资讯（scope=macro）的位置：0=优先，1=中性，2=压后。

            为什么要有这一维（2026-09-11）：宏观块对"某家公司的新闻"是噪声，
            对"国际油价怎么样了"却是唯一正解——同一个块，取决于问句意图。
            所以**只调位置、不删块**：用户若点名了来源（"新浪财经报道了什么"），
            该块是 metadata 精确命中（_entity=True），按常规处理、不压。
            """
            is_macro = (d.get("metadata") or {}).get("scope") == "macro"
            if not is_macro:
                return 1
            if _macro_intent:
                return 0
            if d.get("_entity"):
                return 1
            return 2

        def _rank_key(d):
            t = (d.get("metadata") or {}).get("type")
            ent = 0 if d.get("_entity") else 1
            mac = _macro_rank(d)
            if _concept_intent and not d.get("_entity"):
                intent_rank = 0 if t == "concept" else 1
            elif _report_intent:
                intent_rank = 0 if t == "research_report" else 1
            elif _news_intent:
                intent_rank = 0 if t == "news" else 1
            elif _stock_intent:
                intent_rank = 0 if t == "stock" else 1
            elif _profile_intent:
                intent_rank = 0 if t == "overview" else 1
            else:
                return (mac, ent, 0, _DOC_TYPE_PRIORITY.get(t, 99))
            return (mac, intent_rank, ent, _DOC_TYPE_PRIORITY.get(t, 99))

        unique_docs.sort(key=_rank_key)

        return unique_docs[:n_results]


    # ===== 结构化聚合 / 排行（2026-09-11）=====
    def _metric_table(self):
        """公司 -> 最新一期指标（含市值）。首次调用扫描全库，之后走缓存。

        为什么要"最新一期"：同一家公司有多年的 financial 块，不按 (year, quarter) 取最大，
        「哪家ROE最高」会拿历史某个高点的旧数据来排。表里的 `_ord` 就是干这个的。
        """
        if self._metrics_cache is None:
            table = {}
            try:
                res = self.collection.get(where={"type": "financial"}, limit=20000,
                                          include=["documents", "metadatas"])
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
            except Exception as e:
                print(f"        聚合表构建失败（退化为普通检索）：{e}")
                self._metrics_cache = {}
                return self._metrics_cache

            for i, doc in enumerate(docs):
                row = _parse_financial(doc)
                if not row:
                    continue
                meta = metas[i] if i < len(metas) else {}
                meta = meta or {}
                # 公司名以元数据为准（正文可能带全角/空格变体）
                row["company"] = meta.get("company") or row["company"]
                row["stock_code"] = meta.get("stock_code") or row["stock_code"]
                row["industry"] = meta.get("industry")
                row["text"] = doc
                row["_ord"] = (row.get("year") or 0, row.get("quarter") or 0)
                old = table.get(row["company"])
                if old is None or old["_ord"] < row["_ord"]:
                    table[row["company"]] = row

            # 档案块补市值
            try:
                res2 = self.collection.get(where={"type": "overview"}, limit=2000,
                                           include=["documents", "metadatas"])
                for i, doc in enumerate(res2.get("documents") or []):
                    row = _parse_overview(doc)
                    if not row:
                        continue
                    meta = (res2.get("metadatas") or [])[i] or {}
                    co = meta.get("company") or row["company"]
                    cur = table.setdefault(co, {"company": co, "_ord": (0, 0),
                                                "industry": None})
                    # 行业从档案正文抽（metadata 里根本没有 industry 字段）。
                    # **已有行也要回填**：先扫 financial 再扫 overview，只在新行上赋值
                    # 会让 92 家公司的 industry 永远是 None，行业限定的排行全军覆没。
                    cur["industry"] = row.get("industry") or cur.get("industry")
                    cur.setdefault("stock_code", row.get("stock_code"))
                    if cur.get("industry") is None:
                        cur["industry"] = meta.get("industry")
                    cur["market_cap"] = row.get("market_cap")
            except Exception as e:
                print(f"        市值补充失败（不影响其它指标）：{e}")

            self._metrics_cache = table
            print(f"        聚合指标表：{len(table)} 家公司（"
                  f"{sum(1 for r in table.values() if r.get('roe') is not None)} 家有 ROE）")
        return self._metrics_cache

    def structured_rank(self, query, top_n=5):
        """聚合/排行问题 -> 按指标排序的合成文档列表。返回 [] 表示"不是这类问题"。

        返回结构与普通检索结果**完全一致**（id/text/metadata/distance/score），
        所以 sources 契约不用改，LLM 那条路也能直接吃这些块当上下文。
        """
        q = _norm(query)
        is_desc = any(_norm(h) in q for h in _AGG_DESC)
        is_asc = any(_norm(h) in q for h in _AGG_ASC)
        if not (is_desc or is_asc):
            return []

        metric = None
        for word in sorted(_METRIC_WORDS, key=len, reverse=True):   # 长词优先
            if _norm(word) in q:
                metric = _METRIC_WORDS[word]
                break
        if not metric:
            return []

        inds = {v for w, v in _INDUSTRY_WORDS.items() if _norm(w) in q}
        rows = [r for r in self._metric_table().values()
                if r.get(metric) is not None and (not inds or r.get("industry") in inds)]
        if not rows:
            return []

        rows.sort(key=lambda r: r[metric], reverse=not is_asc)
        label, unit = _METRIC_META[metric]
        print(f"        [聚合] 指标={label} 方向={'升序' if is_asc else '降序'} "
              f"行业={sorted(inds) or '不限'} 候选={len(rows)} 家")

        # 供 LLM 路径使用：上下文里必须**明说**"这是按某指标排的结果"。
        # 不点明的话，模型看到的只是 N 段各自带一堆指标的财务句，它得自己猜该比哪个指标、
        # 以及这些块的先后有没有含义 —— 排行题就答不稳。
        # ⚠️ 智能体层（agent_service/tools.py 的 rag_ask）走的是 LLM 路径，
        #    离线路径 generate_without_llm**不**经过这里，两条都得有排行摘要。
        period = ""
        yq = {(r.get("year"), r.get("quarter")) for r in rows if r.get("year")}
        if len(yq) == 1:
            _y, _q = next(iter(yq))
            period = f"｜数据期间 {_y} 年 Q{_q}"
        header = (f"指标排行：{label}（{'升序' if is_asc else '降序'}）"
                  f"｜候选 {len(rows)} 家"
                  f"{'｜范围 ' + '/'.join(sorted(inds)) if inds else ''}{period}")

        out = []
        for rank, r in enumerate(rows[:top_n], 1):
            out.append({
                "id": f"agg:{r.get('stock_code') or r['company']}",
                "text": r.get("text") or f"{r['company']}：{label} {r[metric]}{unit}",
                "metadata": {"company": r["company"], "stock_code": r.get("stock_code", ""),
                             "industry": r.get("industry"), "type": "financial",
                             "year": r.get("year"), "quarter": r.get("quarter")},
                "distance": None,
                "score": round(float(r[metric]), 4),
                "agg_rank": rank,
                "_agg": True,
                "_agg_header": header,
                "_agg_line": f"{rank}. {r['company']}：{label} {r[metric]}{unit}",
            })
        return out

    # ===== 行业 -> 公司清单（2026-09-24 REQ-050）=====
    def _roster_table(self):
        """industry -> [{company, stock_code, text}]，**只吃 type=overview**（一块 = 一家公司）。

        为什么不用 `_metric_table()` 那份现成的表：
          1. 它按公司去重时把 financial 块当主行，一家公司留的是**最新一期的财务正文**，
             名单题要引用的是"这家公司确实在库里"，档案块才是对的出处；
          2. 它要扫 7204 个 financial 块才建得起来，名单分支不该为"有哪些公司"
             付这个钱（第一次问句直接慢一截，见 `_metric_table` 的扫描逻辑）。
        行业取值优先信 metadata（REQ-049 回填后 overview 92/92 都带），
        没有再退回档案正文那句「…是银行行业的A股上市公司」抽。
        """
        if self._roster_cache is None:
            table = defaultdict(list)
            try:
                res = self.collection.get(where={"type": "overview"}, limit=5000,
                                          include=["documents", "metadatas"])
            except Exception as e:
                print(f"        名单表构建失败（退化为普通检索）：{e}")
                self._roster_cache = {}
                return self._roster_cache
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            seen = set()
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                meta = meta or {}
                parsed = _parse_overview(doc)
                co = meta.get("company") or (parsed or {}).get("company")
                if not co or co in seen:
                    continue          # 一家公司只占一个名额（重复档案直接丢）
                ind = meta.get("industry") or (parsed or {}).get("industry")
                if not ind:
                    continue          # 行业不明的公司不进名单：宁可少列，也不列错行业
                seen.add(co)
                table[ind].append({"company": co, "text": doc,
                                   "stock_code": meta.get("stock_code")
                                   or (parsed or {}).get("stock_code")})
            for lst in table.values():
                # 按公司名排，**有意不用市值排**：名单一旦有序，用户（和 LLM）就会读成排行，
                # 而"这个顺序按什么算的"本分支答不出来。顺序只用于可复现，header 里另有一说。
                lst.sort(key=lambda r: r["company"])
            self._roster_cache = table
            print(f"        名单表：{sum(len(v) for v in table.values())} 家公司 / "
                  f"{len(table)} 个行业")
        return self._roster_cache

    def industry_roster(self, query, min_blocks=5):
        """「X 行业有哪些公司」-> 按公司去重的名单块。返回 [] 表示"不是这类问题"。

        存在理由（线上实证，见 REQ-050 工单）：这类问句过去落在关键词通道上，
        `where={"industry":…}` 之后按 chroma **内部行序**取前 5，于是
        `半导体行业有哪些公司` 答成工业富联一家（4 个季度 financial + 1 档案）。
        那不是检索 bug，是**形状错配**：名单是聚合结果，不是"最相关的 5 个文档"。

        返回结构与 `structured_rank` 完全一致（外加一个 `_roster` 标记供路由归因区分），
        所以 sources 契约、`build_context` 的边界声明注入都不用改。
        """
        q = _norm(query)
        # 有方向词的是排行题（"白酒行业哪些公司最赚钱"），本分支只会按公司名排，
        # 答不了"谁最高" —— 让给 structured_rank（它在 retrieve() 里本来也先跑）。
        if any(_norm(h) in q for h in _AGG_DESC) or any(_norm(h) in q for h in _AGG_ASC):
            return []
        if not any(w in q for w in _ROSTER_INTENTS):
            return []
        inds = {v for w, v in _INDUSTRY_WORDS.items() if _norm(w) in q}
        if not inds:
            return []

        table = self._roster_table()
        rows, seen = [], set()
        for ind in sorted(inds):
            for r in table.get(ind, []):
                if r["company"] in seen:
                    continue
                seen.add(r["company"])
                rows.append(dict(r, industry=ind))
        if not rows:
            return []

        total = sum(len(v) for v in table.values())
        # 名单不吃 top_k：**一块就是一份短档案（≈90 字），把该行业全量列出来才叫回答"有哪些公司"**。
        # 实测 68 题那格的行业最大 11 家，全量约 1000 字，在预算内；
        # 但一旦语料扩到几十家，硬塞会把 build_context 的节选闸触发 —— 那样 header 说
        # "列出 N 家"、正文却在半路被截，等于自己造出一句假话。所以这里按预算自己收口。
        budget = max(MAX_CONTEXT_LENGTH // 2, 600)
        shown, used = [], 0
        for i, r in enumerate(rows):
            if i >= min_blocks and used + len(r["text"]) > budget:
                break
            shown.append(r)
            used += len(r["text"])
        # 边界声明是**判据**不是文案装饰（工单 §三 L3：没有"库内 N 家 / 上限"这句直接算 FAIL）。
        # 全量公司名也写进 header：预算不够时块只能装下前 M 家，但模型得知道
        # "这个行业库里其实有 N 家、分别是这些"，否则它会把 M 家当全部 —— 那正是本 REQ 要治的病。
        header = (f"行业公司清单｜本库共收录 {total} 家 A 股公司（覆盖 {len(table)} 个行业），"
                  f"这不是整个市场的全部名单。"
                  f"问的行业在库内共 {len(rows)} 家，本次列出前 {len(shown)} 家的档案；"
                  f"全量：{'、'.join(r['company'] for r in rows)}")
        print(f"        [名单] 行业={sorted(inds)} 库内 {len(rows)} 家 → 返回 {len(shown)} 家")

        out = []
        for rank, r in enumerate(shown, 1):
            out.append({
                "id": f"roster:{r.get('stock_code') or r['company']}",
                "text": r["text"],
                "metadata": {"company": r["company"], "stock_code": r.get("stock_code") or "",
                             "industry": r["industry"], "type": "overview"},
                "distance": None,
                "score": None,
                "agg_rank": rank,
                "_agg": True,
                "_roster": True,
                "_agg_header": header,
                "_agg_line": (f"{rank}. {r['company']}"
                              f"（{r.get('stock_code') or '代码未知'}）｜{r['industry']}"),
            })
        return out

    def _known_concepts(self):
        """概念词表缓存：表面形式 -> 概念文档搜索词。"""
        if self._concept_cache is None:
            cache = defaultdict(set)
            try:
                res = self.collection.get(where={"type": "concept"}, include=["metadatas"])
                metas = res.get("metadatas") or []
            except Exception:
                metas = []
            # 常见英文通用词黑名单：避免像 market, stock, model 等宽泛词误召回
            _GENERIC_EN = {
                "market", "stock", "model", "capital", "flow", "growth", "cycle",
                "risk", "ratio", "rate", "value", "price", "loss", "profit",
                "income", "asset", "yield", "index", "trade", "trading", "impact",
            }
            for m in metas:
                m = m or {}
                kws = m.get("keywords") or ""
                for kw in kws.split(","):
                    kw = kw.strip()
                    if len(kw) >= 2 and kw.lower() not in _GENERIC_EN:
                        cache[_norm(kw)].add(kw)
                title = m.get("title") or ""
                # 中文核心词
                for cn in re.findall(r"[\u4e00-\u9fff]{2,}", title):
                    cache[_norm(cn)].add(cn)
                # 英文核心金融缩写（2-6位纯英文缩写，如 PE, PB, ROE, FCF, DCF, GDP, CPI 等）
                for en in re.findall(r"\b[A-Za-z]{2,6}\b", title):
                    if en.lower() not in _GENERIC_EN:
                        cache[_norm(en)].add(en)
            self._concept_cache = cache
        return self._concept_cache

    def _known_entities(self):
        """归一化后的「实体表面形式 -> 原始元数据取值集合」，首次调用后缓存。

        为什么必须有它：原 `_extract_keywords` 用
        `re.findall(r'[\\u4e00-\\u9fff]+')` 抽中文，**不做分词** ——
        「华兴科技的主营业务是什么」整串被当成一个关键词，再拿去 `$contains`
        匹配，永远 0 命中。关键词检索这条通道因此等于全程关闭，只剩对中文
        基本无效的英文 embedding 向量检索在起作用，表现为"问什么都查不到"。
        2026-09-10 实测：旧逻辑命中文档 0 篇，本方案 13 篇。

        不引入 jieba 的理由：metadata 里本来就有 company / stock_code，
        直接拿它们对 query 做子串匹配，更准且零依赖。

        2026-09-11 扩展（评测基线 45%，见 `_eval/`）：只匹配**全名**不够，
        用户会打简称。实测「茅台最近有什么新闻」「汾酒最近怎么样」全部 0 命中，
        因为库里存的是"贵州茅台""山西汾酒"，而 `"贵州茅台" in "茅台…"` 为假。
        故在索引里补三类表面形式：
          1. 元数据真实值（归一化后）
          2. **规则派生的简称**：剥地名前缀（贵州茅台→茅台）、剥通用后缀
             （万华化学→万华、招商银行→招商）、股票代码裸数字（SH600519→600519）
          3. 少量规则覆盖不到的俗名（招行、宁王、京东方…）

        ⚠️ 派生与俗名都要求**唯一指向一家公司**，否则丢弃。
        没有这道闸，"工业""银行""中国"这类词会出现在任意问句里而误命中。

        ⚠️ 2026-09-11 二次修正：这道"唯一指向"闸**不足以**挡住全部误召回。
        对 54 条采纳别名做了全量语料审计（`_rag_eval/audit_aliases.py`，统计每条别名
        的"正文命中 / 本公司命中 / 他司命中"），发现两类漏网：
          - 新闻来源名被当公司（`_MEDIA_MARK` 闸门修复，见该常量注释）
          - 中国人民 / 中航 两条他司占比 100% / 52%（进 `_TOO_GENERIC`）
        该审计脚本可重复运行，是这张表的**回归测试**——改规则后必须重跑，
        确认没有新别名越过 30% 他司占比。
        """
        if self._entity_cache is None:
            try:
                metas = self.collection.get(include=["metadatas"])["metadatas"] or []
            except Exception:
                metas = []

            idx = defaultdict(set)        # 表面形式(归一化) -> {原始取值}
            comp_map = defaultdict(set)   # 公司全名(归一化) -> {原始取值}
            for m in metas:
                m = m or {}
                for key in ("company", "stock_code", "industry"):
                    v = m.get(key)
                    if isinstance(v, str) and len(v) >= 2:
                        idx[_norm(v)].add(v)
                c = m.get("company")
                if isinstance(c, str) and len(c) >= 2:
                    comp_map[_norm(c)].add(c)

            # 派生简称：先收集 alias -> {来源公司全名}，只保留唯一指向的
            derived = defaultdict(set)
            for surf in comp_map:
                if _MEDIA_MARK in surf:      # 媒体来源名不参与派生，理由见 _MEDIA_MARK
                    continue
                for pre in _PLACE_PREFIX:
                    if surf.startswith(pre) and len(surf) - len(pre) >= 2:
                        derived[surf[len(pre):]].add(surf)
                for suf in _GENERIC_SUFFIX:
                    if surf.endswith(suf) and len(surf) - len(suf) >= 2:
                        derived[surf[:-len(suf)]].add(surf)
            for al, srcs in derived.items():
                if len(srcs) != 1 or al in _TOO_GENERIC or al in idx:
                    continue
                idx[al].update(comp_map[next(iter(srcs))])

            # 股票代码：SH600519 -> 600519（用户很少打交易所前缀）
            for surf, origins in list(idx.items()):
                mt = re.fullmatch(r"(?:sh|sz|bj)(\d{6})", surf)
                if mt:
                    idx[mt.group(1)].update(origins)

            # 俗名：只在它指向的公司确实存在时生效，且不覆盖已有表面形式
            for alias, full in _CURATED_ALIAS.items():
                fn = _norm(full)
                if fn in comp_map and _norm(alias) not in idx:
                    idx[_norm(alias)].update(comp_map[fn])

            self._company_origins = {o for names in comp_map.values() for o in names}

            # 行业俗称接线：`_INDUSTRY_WORDS` 过去只有 structured_rank 在用，关键词通道拿不到它 ——
            # 问「白酒行业最近有什么新闻」抽不出任何关键词，只能落到对中文近乎无效的英文向量兜底。
            # 登记成"口语词 -> 行业真值"的别名后，keyword_search 就能走 `where={"industry":真值}`。
            # 三道闸逐条都是实测出来的（_plans/REQ-20260924-049 §八）：
            #   1) 真值必须已是 metadata 取值，否则 where 查不到东西还白占一个关键词位；
            #   2) 口语词不得是任何公司/新闻来源名的子串（`证券` ⊂ `证券时报`）—— 否则问来源名时
            #      整个行业的块会带着 entity=True（与公司精确命中同信任级）涌进池子；
            #   3) 口语词本来就在索引里（`银行` 直接是行业真值）则不动，保证现网取词逐字不变。
            for w, v in _INDUSTRY_WORDS.items():
                nw, nv = _norm(w), _norm(v)
                if nw in idx or nv not in idx:
                    continue
                if any(nw != surf and nw in surf for surf in comp_map):
                    continue
                idx[nw] = set(idx[nv])
                self._wired_industries.update(idx[nv])

            self._entity_cache = idx
        return self._entity_cache

    def _query_entities(self, query):
        """query 里出现的实体/简称 -> 应拿去查元数据的**原始取值**集合。

        注意返回的是原始取值（如"贵州茅台"），因为 `where={"company": ...}`
        是精确匹配，用归一化后的形式去查反而查不到。
        """
        q = _norm(query)
        found = set()
        for surf, origins in self._known_entities().items():
            if surf in q:
                found.update(origins)
        # 公司优先（2026-09-24 实测踩坑）：问句里公司名和行业俗称**同时**出现时，两个实体都是
        # `score=1.0, entity=True`，而 keyword_search 的排序键 (macro, intent, ent, type) 分不开它们，
        # 于是**同行业别家公司**的文档与目标公司同权挤进前 5
        # （「同花顺有什么券商研报？」「洋河股份研报里对白酒行业的判断」各 13 轮）。
        # 只回退**接线新引入**的那批标签；现网本来就能取到的（`银行` 直接是 metadata 真值）
        # 一律不动 —— 那是既有行为，不归这条闸门改，否则一处修分会顺带改写别的问句。
        if found and self._wired_industries and any(e in self._company_origins for e in found):
            found = {e for e in found if e not in self._wired_industries}
        return found

    def _extract_keywords(self, query):
        """提取查询关键词：实体/简称命中优先，否则退英文数字串。

        ⚠️ 已知短板（2026-09-11 评测）：没有实体命中时，中文部分**一个词都抽不出来**
        —— 原实现 `re.split(r"[^\\w]+", query)` 对"汾酒最近怎么样"这种纯中文问句
        切不出任何 token，整句被当成一个关键词去 `$contains`，必然 0 命中。
        基线里「哪家公司ROE最高」等聚合型问题就走这条路（它们本质上不该走检索，
        而该走结构化查询，见 _eval 里的 D_指标排行 类目）。
        """
        ents = self._query_entities(query)
        if ents:
            return sorted(ents)
        q = _norm(query)
        # 如果没有公司实体，尝试匹配金融概念常识词
        concepts = set()
        for surf, origins in self._known_concepts().items():
            if surf in q:
                concepts.update(origins)
        if concepts:
            return sorted(concepts)
        kws = []
        for seg in re.findall(r"[a-z0-9]{2,}", q):
            if seg not in self.stop_words:
                kws.append(seg)
        return kws
    
    def hybrid_search(self, query, n_results=TOP_K):
        """混合检索：默认**关键词优先，向量兜底**；`RAG_V2_RRF=on` 时改走 RRF 对等融合。

        为什么默认不是 RRF（2026-09-10 实测）：
        向量侧原用英文模型 all-MiniLM-L6-v2，对中文几乎无区分度。问「华兴科技的主营业务
        是什么」时，关键词侧精确命中 13 篇、全是 SH600001；但融合公式
        `vector_weight/(60+rank)` 让向量第 1 名（≈0.0098）**压过**关键词第 1 名（≈0.0067），
        把 SH600003 / SH600007 挤进 TOP3。

        2026-09-13 转折：embedding 已换成中文模型 BAAI/bge-small-zh-v1.5（A/B 90%→95%），
        向量侧有了中文区分度，RRF 对等融合因此**重新可行**——`RAG_V2_RRF=on` 开启，
        走 `_rrf_fuse`（rank-only，不吃分值，规避 BM25/余弦值域不一）。但融合会让向量路
        **永远**进池（不再是"关键词 0 命中才兜底"），故必须配 rerank 规则 6f 反字根混淆护栏，
        否则会把 H 类（长三角→长江电力）这类字根混淆重新灌进 top-k。实测 A/B 见
        `_rag_eval/eval_report_rrf-zh.json`。
        """
        keyword_docs = self.keyword_search(query, n_results=n_results)
        vector_docs = self.vector_search(query, n_results=n_results)

        if os.environ.get("RAG_V2_RRF", "off").lower() == "on":
            return self._rrf_fuse(keyword_docs, vector_docs, n_results)

        # 默认（已验证 100% 召回 + 0 H 违规）：关键词有命中就只信关键词，
        # 一条都没命中才退回向量（聊胜于无）。
        if keyword_docs:
            return keyword_docs[:n_results]
        return vector_docs[:n_results]

    def _rrf_fuse(self, keyword_docs, vector_docs, n_results, k=60):
        """RRF 对等融合（rank-only，不吃分值，规避 BM25/余弦值域不一，见 Step 4.7）。

        前置：embedding 已是中文区分度模型 bge（否则向量侧噪声污染融合）。
        两路各取排名，RRF 分 = Σ 1/(k+rank)，取并集去重后按融合分降序。
        合并 metadata 的两条规则：
          - 命中两路的文档同时保留 `_entity`（关键词精确实体）与真实 `vector_score`；
          - 仅向量路的文档带 vector_score、无 `_entity`（rerank 规则 6f 会拦字根混淆）；
          - 仅关键词路的文档无 vector_score（rerank 走默认 0.5，与默认 keyword-priority
            路径一致，不引入额外偏置）。
        """
        rrf = {}
        merged = {}
        for rank, d in enumerate(keyword_docs, 1):
            if d["id"] not in merged:
                merged[d["id"]] = dict(d)
            rrf[d["id"]] = rrf.get(d["id"], 0.0) + 1.0 / (k + rank)
        for rank, d in enumerate(vector_docs, 1):
            cur = merged.get(d["id"])
            if cur is None:
                merged[d["id"]] = dict(d)
            else:
                if d.get("vector_score") is not None:
                    cur["vector_score"] = d["vector_score"]
                cur["_entity"] = cur.get("_entity") or d.get("_entity")
                if cur.get("distance") is None:
                    cur["distance"] = d.get("distance")
            rrf[d["id"]] = rrf.get(d["id"], 0.0) + 1.0 / (k + rank)
        out = sorted(merged.values(), key=lambda d: rrf[d["id"]], reverse=True)
        for d in out:
            d["_rrf"] = round(rrf[d["id"]], 5)
        return out[:n_results]


class Reranker:
    """重排序器：对检索结果进行二次精排"""
    
    def __init__(self):
        # 金融领域关键词权重
        self.finance_keywords = {
            "市值": 1.5, "营收": 1.5, "利润": 1.5, "ROE": 1.3,
            "毛利率": 1.3, "净利率": 1.3, "资产负债率": 1.2,
            "现金流": 1.3, "估值": 1.2, "市盈率": 1.2,
            "增长": 1.1, "下滑": 1.1, "盈利": 1.2, "亏损": 1.2,
            "分红": 1.1, "回购": 1.1, "并购": 1.1,
            # 基础概念与常识词拓展
            "牛市": 1.3, "熊市": 1.3, "做多": 1.3, "做空": 1.3,
            "量化": 1.3, "流动比率": 1.3, "夏普比率": 1.3,
            "最大回撤": 1.3, "降准": 1.3, "降息": 1.3, "GDP": 1.3,
            "CPI": 1.3, "PMI": 1.3, "货币政策": 1.2, "通胀": 1.2,
        }

    @staticmethod
    def _extract_query_tokens(query):
        """提取查询关键词（优先 jieba 分词并过滤停用词，无 jieba 时回退到 n-gram 与英文数字串）"""
        stop_words = {
            "什么", "是", "和", "与", "的", "怎么", "如何", "多少", "有", "在",
            "了", "吗", "呢", "？", "。", "，", "！", " ", "请问", "介绍",
            "解释", "意思", "计算", "公式", "区别", "属于", "哪些", "最新",
        }
        try:
            import jieba
            tokens = {w for w in jieba.cut(query) if len(w) > 1 and w not in stop_words}
            if tokens:
                return tokens
        except Exception:
            pass
        tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", query))
        cn_text = re.sub(r"[^\u4e00-\u9fff]", "", query)
        for sw in stop_words:
            cn_text = cn_text.replace(sw, "")
        for i in range(len(cn_text) - 1):
            tokens.add(cn_text[i:i+2])
        return tokens
    
    def rerank(self, query, docs, top_k=TOP_K):
        """对文档进行重排序"""
        scored_docs = []
        
        for doc in docs:
            score = self._compute_score(query, doc)
            doc["rerank_score"] = score
            scored_docs.append(doc)
        
        # 按重排序分数降序
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_docs[:top_k]
    
    def _compute_score(self, query, doc):
        """计算重排序分数"""
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        q_norm = _norm(query)

        score = 0
        
        # 1. 语义相似度（基于向量距离）
        vector_score = doc.get("vector_score", 0.5)
        score += vector_score * 0.4
        
        # 2. 关键词匹配度（改用子串重叠与分词，支持中文问句准确计算重叠度）
        query_keywords = self._extract_query_tokens(query)
        if query_keywords:
            matched = sum(1 for kw in query_keywords if kw in text)
            overlap = matched / len(query_keywords)
            score += overlap * 0.3
        
        # 3. 金融关键词权重
        for kw, weight in self.finance_keywords.items():
            if kw in text:
                score += (weight - 1) * 0.1
        
        # 4. 文档长度惩罚（太短或太长都不好）
        #    overview 例外：公司档案按设计就是一句话（"X（代码）是Y行业的A股上市公司，
        #    最新市值约N亿元"），把它和"被截断的碎片"一起按 0.8 惩罚，会让 92 篇档案
        #    在打分排序里必然输给几百字的 financial —— 实测「贵州茅台市值」top-5 全是
        #    财报，"市值"这个最该由档案回答的问题反而答不出。
        text_len = len(text)
        if text_len < 50 and metadata.get("type") != "overview":
            score *= 0.8
        elif text_len > 1000:
            score *= 0.9
        
        # 5. 元数据匹配
        company = metadata.get("company", "")
        if company and company in query:
            score += 0.2
        
        stock_code = metadata.get("stock_code", "")
        if stock_code and stock_code in query:
            score += 0.3

        # 6. 与 HybridRetriever 的排序意图对齐（否则检索层白排一遍）
        #    6a) 元数据精确命中的文档（"关于这家公司"）压过正文宽召回（"提到这家公司"）
        if doc.get("_entity"):
            score += 0.15
        #    6b) 问消息面时，news 类文档要排在公司档案/财报之前
        if any(h in query for h in _NEWS_HINTS) and metadata.get("type") == "news":
            score += 0.35
        #    6c) 问行情时同理，stock 类文档前置
        if any(h in query for h in _STOCK_HINTS) and metadata.get("type") == "stock":
            score += 0.35
        #    6d) 问档案类（市值/行业/主营）时 overview 前置，与 _rank_key 的
        #        _profile_intent 分支对齐 —— 只改检索层会被这里打回（本项目已踩过两次）
        #        消息面问题除外：问"XX行业最近有什么新闻"时该出 news，不该出档案
        if (any(h in query for h in _PROFILE_HINTS) and metadata.get("type") == "overview"
                and not any(h in query for h in _NEWS_HINTS)):
            score += 0.35

        #    6e) 宏观资讯（scope=macro）与 _rank_key 的 _macro_rank 对齐。
        #        同上：只改检索层会被这里打回。问宏观问题时加分；
        #        其余问题里，非实体命中的宏观块扣分（实体命中的不扣 ——
        #        用户点了"新浪财经"的名，那块就该正常出）。
        if metadata.get("scope") == "macro":
            if any(h in query for h in _MACRO_HINTS):
                score += 0.35
            elif not doc.get("_entity"):
                score -= 0.25

        #    6f) 反字根混淆（H 类护栏，2026-09-13）：关键词 0 命中 + query 抽不出实体时，
        #        退回向量检索，bge 会把「共享字根的无关公司」顶上来
        #        （长三角→长江电力、民生问题→民生银行、中国人民→中国人寿）。
        #        签名：doc 无 _entity 标记、company 非空、且 company 不是 query 的干净子串、
        #        却与 query 共享 ≥2 字片段 —— 这种纯字根混淆一律扣分，跌破 top-k。
        #        命中真实实体的公司（_entity 已置位）不罚；sector 类块 company 为空不罚；
        #        company 本身就在 query 里（如「长江电力的市值」）也不罚。
        if (not doc.get("_entity") and company
                and company not in q_norm and _shares_fragment(company, q_norm)):
            score -= 0.5

        #    6g) 单位可答性（2026-09-22 Jev 巡检回归）：问"是多少厘米"时，真答案常在带
        #        "数字+该单位"的长块（摘要/求解），"分析"类诱饵块一个这样的数字都没有 ——
        #        纯向量+长度惩罚会把能答题的块挤出 top-k。通用"文里有数"不灵：数学语料
        #        人人带等式。只认"问的单位 ≈ 文里的数字邻接"，不做语义判断。
        unit_pattern = _query_unit_pattern(query)
        if unit_pattern and re.search(rf"\d[\d .]*\s*(?:{unit_pattern})", text):
            score += 0.15

        #    6h) 概念常识类问题与 _rank_key 的 _concept_intent 对齐（2026-11-09）
        if metadata.get("type") == "concept":
            if any(h in query for h in _CONCEPT_HINTS):
                score += 0.35
            else:
                score += 0.15

        return score


class FinanceRAGEngineV2:
    """金融 RAG 问答引擎 v2：混合检索 + 重排序"""
    
    def __init__(self):
        print("正在初始化 RAG 引擎 v2（混合检索+重排序）...")
        
        # 连接 ChromaDB
        print("  连接 ChromaDB...")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        
        doc_count = self.collection.count()
        print(f"  数据库中共 {doc_count} 个文档块")
        
        # 初始化混合检索器
        print("  初始化混合检索器...")
        self.retriever = HybridRetriever(self.collection)
        
        # 初始化重排序器
        print("  初始化重排序器...")
        self.reranker = Reranker()
        
        print("[OK] RAG 引擎 v2 就绪！\n")
    
    # 在线追踪：每次检索落一行 JSONL 到 output/trace/（失败不影响问答；RAG_TRACE=off 可关）
    @trace_retrieval
    def retrieve(self, query, top_k=TOP_K):
        """混合检索 + 重排序"""
        # 0. 聚合/排行问题走结构化路径。
        #    向量检索只能返回"和问句最像的文档"，**排不了序**——「哪家公司ROE最高」
        #    实测会返回长江电力/云南白药这类不相干公司的档案。这类问题本质是
        #    `ORDER BY <指标> DESC LIMIT n`，交给 structured_rank。
        agg = self.retriever.structured_rank(query, top_n=top_k)
        if agg:
            print(f"  [0/3] 识别为聚合/排行问题，按指标排序返回前 {len(agg)} 家")
            return agg

        # 0b. 「行业 → 公司清单」同样不吃检索（REQ-050）。排在 structured_rank 之后是**有意的**：
        #     "白酒行业哪些公司最赚钱" 既像名单又像排行，那是排行题（有指标 + 有方向词），
        #     名单分支只会按公司名列，抢过来就等于把排行的正确答案换成一份无序点名。
        roster = self.retriever.industry_roster(query, min_blocks=top_k)
        if roster:
            print(f"  [0/3] 识别为行业→公司清单问题，返回去重名单 {len(roster)} 家")
            return roster

        # 1. 混合检索（召回更多候选）
        print("  [1/3] 混合检索（向量+关键词）...")
        candidates = self.retriever.hybrid_search(query, n_results=top_k * 3)
        print(f"        召回 {len(candidates)} 个候选文档")
        
        # 2. 重排序（精排）
        print("  [2/3] 重排序...")
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        print(f"        精排后保留 {len(reranked)} 个文档")
        
        # 3. 去重
        print("  [3/3] 去重...")
        seen_texts = set()
        unique_docs = []
        for doc in reranked:
            text_hash = hash(doc["text"][:100])
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_docs.append(doc)
        
        print(f"        最终返回 {len(unique_docs)} 个文档\n")
        
        return unique_docs
    
    def build_context(self, retrieved_docs):
        """将检索到的文档拼接成上下文"""
        context_parts = []
        total_length = 0

        # 聚合/排行结果：上下文必须先点明"这是一份排行"，否则模型看到的只是 N 段
        #   各自带一堆指标的财务句，它得自己猜该比哪个指标、块的先后有没有含义。
        #   智能体层（agent_service/tools.py 的 rag_ask）不带 use_llm=False ⇒ 走的就是这里。
        if retrieved_docs and retrieved_docs[0].get("_agg_header"):
            head = retrieved_docs[0]["_agg_header"]
            context_parts.append(head)
            total_length += len(head)

        for idx, doc in enumerate(retrieved_docs, 1):
            text = doc["text"].strip()
            # 聚合块：正文是 9 个指标一整句的财务模板，光看它看不出"名次"。
            #   把 "1. 分众传媒：ROE 21.64%" 这行顶在正文前，名次才是显式的。
            line = doc.get("_agg_line")
            if line:
                text = f"{line}\n{text}"
            remaining = MAX_CONTEXT_LENGTH - total_length
            if remaining < 200:
                break  # 剩余预算撑不起一段有用节选
            clipped = False
            if len(text) > remaining - 24:  # 给 "[idx] " 前缀与节选标记留位
                text = text[: remaining - 24].rstrip() + "…（节选）"
                clipped = True
            # [idx] 与前端 data.sources[idx-1] 严格对齐（编号取检索位次，
            #   与是否因超长被截断无关），点答案里的 [n] 才能跳回第 n 条来源。
            context_parts.append(f"[{idx}] {text}")
            total_length += len(text)
            if clipped:
                break  # 预算已花完
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_with_modelscope(self, query, context, api_key):
        """使用 ModelScope API 生成回答"""
        _t0 = time.perf_counter()   # 追踪：生成侧耗时
        _usage = {}                 # 追踪：流式返回的 usage（若服务端给了）
        prompt = f"""你是一个专业的金融分析师助手。请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 只根据参考资料回答，不要编造数据
2. 如果参考资料中没有相关信息，诚实说明
3. 回答要专业、准确、有条理
4. 【金额单位】参考资料里的金额**已经换算好了**：大于等于 1 亿元的写作「亿元」，
   小于 1 亿元的写作「万元」。请**直接照抄该数字和它后面的单位**，
   **不要再做任何换算、乘除或单位推断**。若同一指标出现多个互相矛盾的数值，请如实指出矛盾。
5. 【百分比】ROE、毛利率、资产负债率等字段已是百分数，直接引用即可，不要再乘 100。
6. 涉及数字时要精确引用
7. 如果有多个公司的数据，请对比分析
8. 【引用溯源】下方参考资料每段开头的 [编号]（如 [1]、[2]）是其序号。
   当你的某个结论、数字或事实来自某段资料时，请在该句末尾用 [编号] 标注来源，
   例如「宁德时代 2023 年营收增长 22%（[3]）」。只标注真正支撑该句的编号，
   不要凭空补标，也不要把答案里所有句子都标注。若某结论无法归到任何一段资料，
   就不要编造编号。

【回答】"""
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的金融分析师助手，擅长分析公司财务数据、股票行情和行业趋势。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": LLM_MAX_TOKENS,
            }
            
            # 无代理时显式直连（{"http": None} 同时挡掉 Windows 注册表系统代理，防学校网络残留 Clash 系统代理设置）
            proxies = {"http": PROXY, "https": PROXY} if PROXY else {"http": None, "https": None}
            
            payload["stream"] = True
            # 追踪用：请求服务端在流式末尾回传 usage。**本项目的 ModelScope 端点未实测**
            # 是否支持该字段，故默认关闭；想看真实 token 就设 RAG_TRACE_USAGE=on，
            # 若因此返回 400，把它关掉即可（追踪绝不能拖垮主流程）。
            if os.environ.get("RAG_TRACE_USAGE", "off").lower() == "on":
                payload["stream_options"] = {"include_usage": True}
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=120
            )
            resp.raise_for_status()
            
            content_parts = []
            _sse_parse_errors = 0
            _reasoning_chars = 0    # 推理模型的思考文本长度：content 空时用于归因
            _finish_reason = None   # "length" = 预算耗尽被截断，不是 API Key 问题
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    chunk_str = line.decode("utf-8")[6:]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        # usage 只在末尾块出现（且需服务端支持），能拿到就记，拿不到就是 None
                        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                            _usage.update(chunk["usage"])
                        choices = chunk.get("choices", [])
                        if choices:
                            if choices[0].get("finish_reason"):
                                _finish_reason = choices[0]["finish_reason"]
                            delta = choices[0].get("delta", {})
                            if delta.get("reasoning_content"):
                                _reasoning_chars += len(delta["reasoning_content"])
                            if "content" in delta and delta["content"]:
                                content_parts.append(delta["content"])
                    except Exception:
                        _sse_parse_errors += 1  # 单块解析失败跳过，汇总在流结束后可见

            result = "".join(content_parts)
            if _sse_parse_errors:
                print(f"[llm] SSE 流有 {_sse_parse_errors} 个数据块解析失败（已跳过）", flush=True)
            if result:
                answer, err = result, None
            elif _finish_reason == "length":
                # 推理模型把预算全花在思考上、正文一个字没出 —— 这不是鉴权问题，
                # 误指 API Key 会把排障方向带偏（2026-09-24 V6 教训）。
                err = "reasoning_exhausted_budget"
                answer = (f"[模型把 token 预算全部花在推理上（max_tokens={LLM_MAX_TOKENS}，"
                          f"思考约 {_reasoning_chars} 字，正文 0 字），本次无法生成回答。"
                          f"请重试或缩小问题。]\n\n"
                          f"以下是检索到的相关资料，请自行参考：\n\n{context}")
            else:
                err = "empty_answer"
                answer = "[模型未返回有效内容，请检查 API Key 和模型名称]"
            log_generation(
                query=query,
                model=LLM_MODEL,
                prompt_chars=len(prompt),
                answer_chars=len(answer),
                prompt_tokens=_usage.get("prompt_tokens"),
                completion_tokens=_usage.get("completion_tokens"),
                elapsed_ms=round((time.perf_counter() - _t0) * 1000, 1),
                error=err,
            )
            return answer
        
        except Exception as e:
            log_generation(
                query=query,
                model=LLM_MODEL,
                prompt_chars=len(prompt),
                answer_chars=0,
                elapsed_ms=round((time.perf_counter() - _t0) * 1000, 1),
                error="%s: %s" % (type(e).__name__, e),
            )
            return f"[LLM 调用失败：{e}]\n\n以下是检索到的相关资料，请自行参考：\n\n{context}"
    
    def generate_without_llm(self, query, context, retrieved_docs):
        """不使用 LLM，直接返回检索结果（离线模式）

        2026-10-15 补降级标记（M6：失败必须写进数据流，不能只 print 到 stdout）。
        两条路径会走到这里：
          a) 缺 MODELSCOPE_API_KEY / LLM 调用失败 —— 用户**必须**知道这不是模型给的结论
          b) 令牌桶超限主动降级（web_app.py 已在 payload 里带 degraded:true，此处是 CLI 侧）
        原先只 print("[离线] ...") 到 stdout，答案文本本身无标记 ⇒ 把纯检索拼接
        当成 LLM 结论读，是最危险的一种静默降级。现在横幅放进 answer 第一行。
        """
        answer_parts = [
            "> ⚠️ **[离线模式]** 本次未调用 LLM（缺少 `MODELSCOPE_API_KEY` 或调用失败/超限降级），"
            "以下是检索片段原文拼接，**不是生成的答案**，任何结论请自行核对原文。\n",
            f"## 检索到 {len(retrieved_docs)} 条相关信息（v2 混合检索+重排序）\n",
            "### 参考资料\n",
        ]
        
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc["metadata"]
            vector_score = doc.get("vector_score", 0)
            rerank_score = doc.get("rerank_score", 0)
            hybrid_score = doc.get("hybrid_score", 0)
            
            if meta.get("type") == "concept":
                header_title = f"金融常识·{meta.get('title', '基础概念')}"
                sub_tag = meta.get("category", "concept")
            else:
                header_title = f"{meta.get('company', '?')}（{meta.get('stock_code', '?')}）"
                sub_tag = meta.get("type", "?")

            answer_parts.append(
                f"**[{i}] {header_title} - {sub_tag}**\n"
                f"   向量相似度：{vector_score:.2%} | 重排序分数：{rerank_score:.3f} | 混合分数：{hybrid_score:.6f}\n"
                f"{doc['text']}\n"
            )
        
        if retrieved_docs and retrieved_docs[0].get("_agg"):
            first = retrieved_docs[0]
            head = "\n".join(d["_agg_line"] for d in retrieved_docs if d.get("_agg_line"))
            # 边界声明（"本库共收录 N 家…不是整个市场"）只存在于 _agg_header。
            # 离线路径原先只吃 _agg_line ⇒ 名单题在最需要这句的分支上反而看不到它，
            # 正好把本 REQ 要治的"把 5 家当全部"重新犯一遍。
            title = "### 行业公司清单\n\n" if first.get("_roster") else "### 排行结果\n\n"
            # insert(1) 而不是 insert(0)：index 0 是 2026-10-15 加的离线横幅，
            # 它必须永远是答案第一行，否则聚合分支会把横幅挤下去。
            answer_parts.insert(1, title + first.get("_agg_header", "") + "\n" + head + "\n")

        answer_parts.append("\n> [v2] 以上使用 v2 引擎（混合检索+重排序）检索，召回率已优化。")
        return "\n".join(answer_parts)
    
    def ask(self, query, use_llm=True):
        """完整 RAG 流程：混合检索 → 重排序 → 生成回答"""
        print(f"[Q] {query}\n")
        
        # 1. 混合检索 + 重排序
        print("[检索] 混合检索 + 重排序...")
        retrieved = self.retrieve(query)
        print(f"   找到 {len(retrieved)} 条相关记录\n")
        
        # 2. 构建上下文
        context = self.build_context(retrieved)

        # 3. 拒答闸门（2026-10-15，对应蒸馏文档 04 §2③ 的第一层）
        # 标定见 config.py REFUSAL_RERANK 注释：12 条负例全部 <= 0.495，20/22 正例 >= 0.515，
        # 0.50 是干净分离点。低于它 → 判"资料不足"，**不调 LLM**（省 token，且从源头
        # 阻断"拿着弱相关片段编一个看似合理的答案"这条幻觉路径）。
        refusal = self._refusal_reason(retrieved)
        if refusal is not None:
            print(f"[拒答] {refusal}")
            return {
                "query": query,
                "answer": self._refusal_answer(query, retrieved, refusal),
                "sources": retrieved,
                "refused": True,
                "low_confidence": False,
            }

        # 4. 生成回答
        api_key = os.environ.get("MODELSCOPE_API_KEY", "")
        if use_llm and api_key:
            print("[LLM] 调用生成回答...")
            answer = self.generate_with_modelscope(query, context, api_key)
        else:
            print("[离线] 返回检索结果...")
            answer = self.generate_without_llm(query, context, retrieved)

        # 5. 低置信提示（2026-10-15）
        # 与拒答闸门的分工：拒答是"判定没有资料"（硬线，不调 LLM）；
        # 这里是"资料勉强够但不牢"（软提示，照常生成但告知用户没底气）。
        banner = self._low_conf_banner(retrieved)
        if banner:
            answer = banner + answer

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved,
            "refused": False,
            "low_confidence": bool(banner),
        }

    def _refusal_reason(self, retrieved):
        """top1 rerank 低于 REFUSAL_RERANK → 返回拒答原因，否则 None。"""
        try:
            from config import REFUSAL_RERANK
        except Exception:
            return None
        if not retrieved:
            return "未检索到任何片段"
        scores = [d.get("rerank_score") for d in retrieved
                  if isinstance(d, dict) and d.get("rerank_score") is not None]
        if not scores:
            return None
        top1 = max(scores)
        if top1 < REFUSAL_RERANK:
            return f"最高重排分 {top1:.2f} 低于拒答线 {REFUSAL_RERANK:.2f}"
        return None

    def _refusal_answer(self, query, retrieved, reason):
        """资料不足时的标准答复。

        设计取舍：**仍然展示最接近的 1-2 个片段**，但明确标注"这不构成答案"。
        完全不展示等于让用户无从判断"是库里真没有，还是检索没做好"；
        展示 + 强标注则两者都能看出来。
        """
        parts = [
            f"> 🚫 **[资料不足，已拒答]** {reason} —— 本知识库未收录能支撑该问题的资料，"
            f"**不生成回答**（避免基于弱相关片段推测）。\n",
            "若你认为这个问题应该有答案，可能是：① 相关文档尚未入库；"
            "② 提问用词与库内术语差异较大，可换用更贴近的表述重试。\n",
        ]
        shown = [d for d in retrieved if isinstance(d, dict)][:2]
        if shown:
            parts.append("以下是最接近的片段，**仅供参考，不构成答案**：\n")
            for i, d in enumerate(shown, 1):
                meta = d.get("metadata") or {}
                txt = (d.get("text") or "")[:200]
                parts.append(f"**[参考 {i}]** {meta.get('company', '?')}（{meta.get('stock_code', '?')}）"
                             f"- {meta.get('type', '?')}\n{txt}\n")
        return "\n".join(parts)

    def _low_conf_banner(self, retrieved):
        """软提示线：top1 在 [拒答线, +0.05) 区间时提示"资料不牢"。

        与拒答线的间隔只有 0.05，是有意的 —— 标定显示 0.50 以上正例密度骤增，
        但 0.50~0.55 仍有 3 个正例（0.537/0.527/0.515）。这三个不该被拒答，
        只该被提醒。再往上提提示线就会误杀它们。
        """
        try:
            from config import REFUSAL_RERANK
        except Exception:
            return ""
        if not retrieved:
            return ""
        scores = [d.get("rerank_score") for d in retrieved
                  if isinstance(d, dict) and d.get("rerank_score") is not None]
        if not scores:
            return ""
        top1 = max(scores)
        lo = REFUSAL_RERANK
        hi = REFUSAL_RERANK + 0.05
        if lo <= top1 < hi:
            return (f"> ⚠️ **[检索置信度偏低]** 最高重排分 {top1:.2f} 刚过拒答线 {lo:.2f} —— "
                    f"资料相关性不强，以下回答请谨慎核对。\n\n")
        return ""
    
    def stats(self):
        """返回数据库统计信息"""
        count = self.collection.count()
        
        all_docs = self.collection.get(include=["metadatas"])
        type_counts = {}
        company_counts = {}
        
        for meta in all_docs["metadatas"]:
            t = meta.get("type", "unknown")
            c = meta.get("company", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            company_counts[c] = company_counts.get(c, 0) + 1
        
        return {
            "total_documents": count,
            "by_type": type_counts,
            "by_company": company_counts,
        }


if __name__ == "__main__":
    # 对比测试 v1 和 v2
    print("=" * 60)
    print("RAG 引擎对比测试：v1（纯向量） vs v2（混合检索+重排序）")
    print("=" * 60)
    
    engine_v2 = FinanceRAGEngineV2()
    
    test_queries = [
        "科技行业有哪些公司？",
        "哪家公司的市值最高？",
        "最近哪只股票涨得最好？",
        "哪家公司的ROE最高？",
        "贵州茅台的财务状况如何？",
        "比亚迪的营收是多少？",
    ]
    
    for q in test_queries:
        result = engine_v2.ask(q, use_llm=False)
        print(f"\n{'='*60}")
        print(f"问题：{result['query']}")
        print(f"回答：\n{result['answer']}")
        print(f"{'='*60}\n")
