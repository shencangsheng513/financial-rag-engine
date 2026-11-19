# -*- coding: utf-8 -*-
"""build_concepts_corpus.py — 构建金融基础常识与财务概念词典并向量化入库

为解决用户提问「什么是市盈率？」「什么是ROE？」「什么是做多做空？」等
基础概念却被 0.50 拒答闸门判定为「资料不足」的问题。
收录 72+ 个权威、标准、清晰的核心金融概念块，涵盖：
  1. 估值指标（PE/PB/PS/PEG/股息率/DCF/EV-EBITDA...）
  2. 财务报表与指标（ROE/杜邦分析/ROA/ROIC/毛利率/净利率/资产负债率/现金流/流动比率...）
  3. 交易规则与市场制度（T+1/涨跌停/集合竞价/换手率/除权除息/分红转增/两融/北向资金...）
  4. 投资风格与行情状态（牛熊震荡市/做多做空/价值投资/护城河/白马周期股/止盈止损...）
  5. 量化金融与风控（多因子/Alpha-Beta/夏普比率/最大回撤/胜率盈亏比/滑点/对冲...）
  6. 宏观经济与货币政策（GDP/CPI/PPI/PMI/M1-M2/降准降息/LPR-MLF/逆回购/社融...）

写入 ChromaDB：
  id: concept_<name>
  text: 【金融常识·<分类>】...
  metadata: {
      type: "concept",
      category: "<cat>",
      title: "<title>",
      keywords: "<kw1,kw2...>",
      scope: "concept",
      company: "金融常识",
      stock_code: "CONCEPT"
  }
"""

import json
from pathlib import Path
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DB_PATH, COLLECTION_NAME
from shared.config import PORTS

LOCAL_MODEL_DIR = ROOT.parent / "models" / "bge-small-zh-v1.5"
MODEL_NAME = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5"
DATA_FILE = ROOT / "data" / "financial_concepts.json"


CONCEPTS_EXTENDED = [
  {
    "id": "concept_fcf",
    "title": "自由现金流（Free Cash Flow）",
    "category": "financial_statements",
    "keywords": "自由现金流,FCF,资本开支,分红基础,真金白银,现金流折现",
    "text": "【金融常识·财务指标】自由现金流（Free Cash Flow，简称 FCF）\n定义：企业在满足了再投资需求（维持正常经营与扩大再生产所需的资本性支出 CAPEX）后，所剩余的、可以完全自由分配给企业所有投资者（包括股权股东和债权人）的最大现金金额。\n计算公式：自由现金流（FCF） = 经营活动现金流量净额 - 资本性支出（购置固定资产、无形资产和其他长期资产支付的现金）。\n实战意义：巴菲特极度看重企业的「所有者收益（Owner Earnings）」，其实质就是自由现金流。只有拥有充沛自由现金流的企业，才具备不依赖外部借债而持续进行高额现金分红、回购股票注销或逆周期兼并收购的真正实力。"
  },
  {
    "id": "concept_economic_moat",
    "title": "经济护城河（Economic Moat）",
    "category": "market_concepts",
    "keywords": "经济护城河,护城河,巴菲特,晨星,竞争壁垒,超额收益,定价权",
    "text": "【金融常识·投资哲学】经济护城河（Economic Moat）\n定义：沃伦·巴菲特提出的经典投资概念，指企业抵御竞争对手蚕食、长期保持高于行业平均资本回报率（高ROIC）的结构性竞争优势与壁垒。\n晨星（Morningstar）归纳的五大护城河来源：\n1. 无形资产：包括享有溢价定价权的品牌（如贵州茅台）、排他性专利权（如创新药龙头）、政府准入特许牌照；\n2. 转换成本（Switching Costs）：客户改用竞品时面临极高的金钱、时间或业务中断代价（如工业软件、银行核心系统）；\n3. 网络效应（Network Effect）：产品价值随使用用户数量的增长而呈指数级增加（如微信、彭博终端）；\n4. 成本优势：凭借独特的工艺、低成本原材料渠道或规模经济实现同行无法企及的超低成本（如台积电、福耀玻璃）；\n5. 有效规模（Efficient Scale）：在自然垄断或有限容量的小市场中，新进入者投资必亏，现有巨头独享市场。"
  },
  {
    "id": "concept_stock_styles",
    "title": "白马股、蓝筹股与成长股的区别",
    "category": "market_concepts",
    "keywords": "白马股,蓝筹股,成长股,白马蓝筹,投资风格,核心资产",
    "text": "【金融常识·股票类型】白马股、蓝筹股与成长股的核心区别\n定义与特征对比：\n1. 白马股（White Horse Stock）：信息高度透明、业绩长期优良稳健、具备持续高分红能力与行业龙头地位的优质股票（如贵州茅台、格力电器）。特点是明牌、风险低、估值相对合理，但短期暴利爆发力弱；\n2. 蓝筹股（Blue Chip）：市值庞大、业绩稳定、在国家经济体系中处于中流砥柱地位的大型传统企业（如工农中建、中国神华、中国石油）。多属于公用事业或传统重工业，波动小、高股息、分红稳定；\n3. 成长股（Growth Stock）：处于高速成长赛道，营业收入和净利润年均复合增速通常高于20%~30%的股票（如早期宁德时代、半导体芯片龙头）。特点是高研发投入、低现金分红（利润全用于扩张）、高市盈率估值，股价弹性大但波动剧烈。"
  },
  {
    "id": "concept_cyclical_stocks",
    "title": "周期股（Cyclical Stock）",
    "category": "market_concepts",
    "keywords": "周期股,强周期,景气周期,猪周期,煤炭钢铁化工,周期陷阱",
    "text": "【金融常识·股票类型】周期股（Cyclical Stock）及周期反向投资\n定义：公司经营业绩与产品价格与宏观经济景气周期或大宗商品供需周期高度正相关的股票。典型行业包括煤炭、钢铁、有色金属、基础化工、海运航运、生猪养殖等。\n周期投资核心规律与陷阱：\n1. 繁荣顶点买入陷阱：在周期繁荣顶点，产品供不应求价格飙升，上市公司单季暴赚数十亿，市盈率（PE）看起来只有 3~5 倍极具诱惑力，但此时往往是产能过剩供求见顶之时，低PE买入常面临股价腰斩；\n2. 低谷惨烈反转买入：在周期见底阶段，全行业大幅亏损、中小产能破产出清，企业单季巨亏、市盈率显示为负值，此时供需格局重塑，往往是周期股反向布局的黄金时刻。\n投资法则：周期股投资讲究「高PE/亏损时买入，低PE/暴利时卖出」。"
  },
  {
    "id": "concept_stop_loss_profit",
    "title": "止盈与止损（Take Profit & Stop Loss）",
    "category": "market_concepts",
    "keywords": "止盈,止损,风控,回撤控制,截断亏损,让利润奔跑,ATR止损",
    "text": "【金融常识·交易风控】止盈（Take Profit）与止损（Stop Loss）\n交易哲学与纪律：华尔街名言「截断亏损，让利润奔跑」是专业交易员与业余散户的根本分水岭。\n1. 止损（Stop Loss）：当买入逻辑被证伪或股价跌破预设警戒线时，坚决卖出头寸以保全本金。常见止损方法包括：\n   - 固定比例止损：如亏损达到 -5% 或 -8% 强制平仓；\n   - 关键技术位止损：跌破前期平台支撑位或关键均线（如60日均线）；\n   - ATR波动率止损：根据真实波动幅度设置动态距离，防止日内杂波被震出；\n2. 止盈（Take Profit）：当盈利达到预期目标或趋势出现见顶信号时兑现收益。常见方式为移动止盈（Trailing Stop），即随股价上涨将保护线上移，回落一定比例后离场锁定大部分利润。"
  },
  {
    "id": "concept_lockup_expiration",
    "title": "限售股解禁（Lockup Expiration）",
    "category": "trading_rules",
    "keywords": "限售股解禁,解禁,大小非,首发原股东限售,定增限售,流通盘",
    "text": "【金融常识·市场机制】限售股解禁（Lockup Expiration）\n定义：上市公司首次公开发行（IPO）或定向增发（定增）时，大股东、高管及战略投资者持有的具有锁定承诺（通常1年、3年不得卖出）的股份，在锁定期满后转为可在二级市场自由买卖流通的普通股。\n分类与市场冲击：\n1. 大非与小非：持股 5% 以上为大非，5% 以下为小非；\n2. 对股价的影响：限售股解禁本身不等于股东一定会立即卖出，但它使二级市场潜在的可抛售供给量骤增，特别当持股机构成本极低存在巨额浮盈时，股东减持套现的冲动会给股价带来显著的抛压与心理打压；\n3. 减持新规约束：监管对破发、破净、过去三年未进行现金分红公司的控股股东及实控人的二级市场减持有着严格的限制条件。"
  },
  {
    "id": "concept_stock_pledge_risk",
    "title": "股票质押爆仓风险（Stock Pledge Risk）",
    "category": "trading_rules",
    "keywords": "股票质押,股权质押,质押率,预警线,平仓线,爆仓,强平",
    "text": "【金融常识·信用风险】上市公司大股东股票质押与爆仓风险\n定义：上市公司大股东（控股股东）将持有的本公司股票作为质押担保物，向证券公司、银行或信托公司融资借款的行为。\n预警线与平仓线机制：\n1. 质押率：通常为股票市值的 30%~50%（折价借款）；\n2. 预警线（一般 150%~160%）：当股价持续下跌导致质押股票市值接近预警线时，资金方会要求大股东追加现金或补充质押股票；\n3. 平仓线（一般 130%~140%）：若股价跌破平仓线且大股东无钱补仓，券商等质押方有权在二级市场强制卖出被质押股票（强制平仓/爆仓）；\n恶性循环：强制平仓导致的巨额抛盘会进一步砸低股价，触发更多质押盘爆仓，形成「股价下跌 → 强平踩踏 → 股价进一步暴跌」的系统性负反馈。"
  },
  {
    "id": "concept_dragon_tiger_list",
    "title": "龙虎榜（Dragon & Tiger List）",
    "category": "trading_rules",
    "keywords": "龙虎榜,席位,机构专用,知名游资,买卖五档,异动公告",
    "text": "【金融常识·交易盘口】A股 龙虎榜制度\n定义：交易所每日收盘后，公布当日交易中存在显著异常波动股票的买入和卖出金额最大的前 5 名营业部席位清单。\n主要上榜条件（满足其一即可）：\n1. 日收盘价涨跌幅偏离值达到 ±7%；\n2. 日价格振幅达到 15%；\n3. 日换手率达到 20%；\n4. 连续三个交易日累计涨跌幅偏离值达到 20%。\n席位分析实战：\n- 机构专用席位：公募基金、社保基金、保险资管、券商自营等正规机构的交易通道，多个机构专用席位净买入通常代表中长期资金真金白银建仓；\n- 游资营业部席位：活跃游资大本营，多体现短线打板接力、次日获利砸盘的高频博弈风格。"
  },
  {
    "id": "concept_block_trade",
    "title": "大宗交易（Block Trade）",
    "category": "trading_rules",
    "keywords": "大宗交易,大单场外协议,折价率,溢价率,大股东减持",
    "text": "【金融常识·交易机制】A股 大宗交易制度（Block Trade）\n定义：单笔交易规模达到规定限额（如主板单笔买卖申报数量在 30 万股以上，或交易金额在 200 万元人民币以上），由买卖双方在场外自主协议约定价格与数量，通过交易所大宗交易平台撮合成交的交易方式。\n特点与看盘要点：\n1. 不影响二级市场实时集合竞价与分时K线走势：大宗交易独立结算，不计入当日即时成交量与指数计算；\n2. 折价成交：实务中大宗交易绝大部分是以二级市场市价的 90%~95%（折价 5%~10%）成交，通常是大股东、定增机构套现离场的主要途径，过大的折价率往往对次日股价构成心理压制；\n3. 溢价成交：极少数情况下买方愿以高于现价的价格溢价接盘，可能体现特定战略投资者的强烈长期看好。"
  },
  {
    "id": "concept_delisting_rules",
    "title": "股票退市制度（Delisting Rules）",
    "category": "trading_rules",
    "keywords": "退市制度,面值退市,财务退市,重大违法退市,常态化退市",
    "text": "【金融常识·市场制度】A股 常态化退市制度体系\n定义：不符合挂牌上市条件的上市公司依照法定程序退出二级市场交易的制度，是资本市场优胜劣汰、吐故纳新的核心机制。\n四大法定退市标准：\n1. 交易类强制退市：连续 20 个交易日的每日股票收盘价均低于 1 元人民币（著名的「1元面值退市」，无退市整理期，一票否决）；连续 20 个交易日股东人数少于规定人数等；\n2. 财务类强制退市：利润总额、净利润或者扣除非经常性损益后的净利润孰低者为负值且营业收入低于规定金额（如主板 3 亿元）；期末净资产为负值等；\n3. 规范类退市：长期财务造假不改正、信息披露严重违规、丧失独立经营能力；\n4. 重大违法类退市：欺诈发行、重大财务造假触及刑事立案或行政重罚。"
  },
  {
    "id": "concept_st_stocks",
    "title": "ST 与 *ST 风险警示制度",
    "category": "trading_rules",
    "keywords": "ST,*ST,特别处理,退市风险警示,摘帽,炒差炒壳",
    "text": "【金融常识·市场制度】ST 与 *ST 风险警示板块\n定义：交易所对财务状况异常或其他状况异常的上市公司股票交易实行的特殊处理标识。\n两类核心区别：\n1. ST（Special Treatment，其他风险警示）：公司出现生产经营活动严重受阻、主要银行账号被冻结、违规担保或资金被大股东占用等异常，虽暂无即刻退市风险，但日常涨跌幅被限制为 ±5%；\n2. *ST（退市风险警示）：公司触及财务类亏损指标、期末净资产为负或涉及重大违法，已被处以退市风险警示。若在下一年度年报披露时仍未改善，将被直接终止上市强制退市；\n3. 摘帽：当公司完成合规整改或盈利恢复达到标准后，可申请撤销风险警示，俗称「摘帽」。"
  },
  {
    "id": "concept_goodwill_impairment",
    "title": "商誉与商誉减值风险（Goodwill Impairment）",
    "category": "financial_statements",
    "keywords": "商誉,商誉减值,高溢价并购,资产减值,利润洗澡",
    "text": "【金融常识·财务风险】商誉与商誉减值（Goodwill & Impairment）\n形成机制：在非同一控制下的企业合并并购中，购买方支付的收购对价超过被收购方可辨认净资产公允价值的溢价差额，在购买方的资产负债表上确认为「商誉」无形资产。\n减值风险与暴雷路径：\n1. 商誉不逐年摊销，但必须每年末进行减值测试；\n2. 业绩对赌期满暴雷：很多收购标的在 3 年对赌期内踩线达标，对赌期一过业绩断崖式变脸；\n3. 减值对利润的直接打击：商誉一旦计提减值准备，直接冲减当期资产减值损失，全额吞噬当期净利润（常常导致原本盈利数亿的公司单年巨亏数十亿），且在会计准则上**商誉减值损失一经计提、以后期间绝对不得转回**。"
  },
  {
    "id": "concept_turnover_inventory_receivables",
    "title": "存货周转率与应收账款周转率",
    "category": "financial_statements",
    "keywords": "存货周转率,应收账款周转率,营运能力,存货周转天数,赊销风险",
    "text": "【金融常识·财务指标】存货周转率与应收账款周转率（营运资金周转效率）\n定义与计算：\n1. 存货周转率（Inventory Turnover） = 营业成本 / 平均存货。\n   - 存货周转天数 = 360 / 存货周转率。天数越短，说明原材料到产成品的销售变现速度越快，占用流动资金越少；若存货周转天数突然大幅攀升，需高度警惕产品滞销积压与跌价计提风险；\n2. 应收账款周转率（Receivables Turnover） = 营业收入 / 平均应收账款。\n   - 应收账款周转天数（账期） = 360 / 应收账款周转率。账期越长，说明公司对下游客户议价地位弱、大量采用赊销刺激营收，坏账暴雷几率上升。"
  },
  {
    "id": "concept_rd_capitalization",
    "title": "研发费用资本化与费用化",
    "category": "financial_statements",
    "keywords": "研发费用,资本化,费用化,无形资产,利润调节,盈余管理",
    "text": "【金融常识·财务辨析】研发投入资本化 vs 费用化的会计玄机\n会计处理路径：\n1. 费用化：研发支出发生时全额计入当期利润表的「研发费用」，直接抵减当期净利润。这是一种最为谨慎稳健的会计处理（如多数优质科技巨头研发费用化率达 100%）；\n2. 资本化：将符合条件的开发阶段支出计入资产负债表上的「开发支出」，项目完成后转为「无形资产」，在未来多年内分期摊销。\n投资辨析：\n若某公司将研发投入的 50% 甚至 80% 大比例资本化，其实质是将原本应当吞噬当期利润的大笔研发成本推迟到未来摊销，在账面上虚增了当期净利润。一旦研发的新产品商业化失败，巨额无形资产将面临一次性减值大暴雷。"
  },
  {
    "id": "concept_win_rate_profit_ratio",
    "title": "胜率与盈亏比（交易期望值）",
    "category": "quant_finance",
    "keywords": "胜率,盈亏比,赔率,数学期望,交易系统,风控核心",
    "text": "【金融常识·量化交易】胜率（Win Rate）与盈亏比（Profit-Loss Ratio）的数学期望\n核心公式：交易系统的长期数学期望值（Expectancy） = （胜率 × 平均单笔盈利） - （（1 - 胜率） × 平均单笔亏损）。\n两大流派平衡哲学：\n1. 高胜率低盈亏比流派（如套利交易、网格做市、高频做T）：胜率往往达到 70%~90%，但单笔盈利极薄，必须配备严格止损，否则一次极端亏损就会全额抹杀几十次小盈利；\n2. 低胜率高盈亏比流派（如趋势跟踪、风险投资、大波段动量）：胜率可能只有 35%~45%，但依托严格止损（小亏）和让利润奔跑（单笔盈利是亏损的3倍至5倍），整体依然能够实现长期丰厚复利；\n误区警示：单方面追求 100% 极高胜率是投资陷阱，成熟交易者永远追求系统数学期望的最大化。"
  },
  {
    "id": "concept_slippage_market_impact",
    "title": "滑点与市场冲击成本",
    "category": "quant_finance",
    "keywords": "滑点,冲击成本,撮合成交,流动性摩擦,容量上限",
    "text": "【金融常识·量化交易】滑点（Slippage）与市场冲击成本（Market Impact）\n定义与产生机制：\n1. 滑点（Slippage）：交易策略发出的理论指令价格与最终交易所系统实际撮合成交价格之间的负向偏差。例如策略在 10.00 元触发买入信号，但盘口卖单瞬间被抢光，最终成交均价为 10.05 元，0.05 元的差额即为滑点；\n2. 市场冲击成本（Market Impact）：大额资金的买入或卖出需求本身由于订单量巨大，吃掉了订单簿深度的多档挂单，从而自行动量推高买价或砸低卖价所付出的隐性额外成本；\n量化实战：滑点与冲击成本是决定高频策略与短线因子能否承载大资金容量（Capacity）的致命瓶颈，很多在回测中漂亮的策略一旦实盘由于滑点摩擦全部由赚转亏。"
  },
  {
    "id": "concept_lookahead_overfitting",
    "title": "未来函数与过拟合（回测陷阱）",
    "category": "quant_finance",
    "keywords": "未来函数,过拟合,前瞻偏差,样本外失效,幸存者偏差,回测作弊",
    "text": "【金融常识·量化风险】未来函数与过拟合（量化回测两大致命陷阱）\n定义与排查：\n1. 未来函数（Lookahead Bias / 前瞻偏差）：在历史模拟回测中，策略在第 T 天决策时非法使用了第 T 天收盘后才产生的数据（如使用了当天的最高价、收盘价甚至后续公布的财务报告、高转增送股信息）。带未来函数的策略回测曲线近乎完美的45度暴利，实盘必死；\n2. 过拟合（Overfitting）：策略为了迎合历史特定噪音数据，通过添加大量复杂的条件参数进行过度调优，使得样本内（In-Sample）回测年化翻倍、夏普极高，但在未见过的样本外（Out-of-Sample）新市场中完全失效；\n3. 幸存者偏差：回测股票池仅包含当前还活着的公司，自动排除了历史上爆雷退市的股票，导致回测收益严重高估。"
  },
  {
    "id": "concept_omo_reverse_repo",
    "title": "央行逆回购与公开市场操作（OMO）",
    "category": "macro_policy",
    "keywords": "逆回购,央行公开市场操作,OMO,流动性投放,资金利率,DR007",
    "text": "【金融常识·宏观货币】央行逆回购（Reverse Repo）与公开市场操作\n运作机制：中国人民银行向一级交易商（主要商业银行）购买有价证券（通常是国债），并约定在未来特定日期将有价证券卖还给一级交易商的交易行为。\n本质内涵：\n- 央行逆回购是**向市场投放流动性**（央行出资金、商业银行拿走资金并押出国债）；\n- 逆回购到期则是**自然回笼流动性**（商业银行还钱赎回国债）；\n- 期限品种：以 7 天期逆回购为主，季末年末配合 14 天期等；\n- 7天逆回购利率是央行的核心短期政策利率，直接指导银行间市场基准利率（如 DR007），其利率上调或下调分别代表短期加息或降息。"
  },
  {
    "id": "concept_total_social_financing",
    "title": "社会融资规模（社融）",
    "category": "macro_policy",
    "keywords": "社融,社会融资规模,信贷周期,实体经济融资,表外融资,政府债券",
    "text": "【金融常识·宏观信用】社会融资规模（Total Social Financing，简称社融）\n定义：指一定时期内实体经济（境内非金融企业和住户）从整个金融体系获得的资金总额，是由中国人民银行编制的中国特有的宏观信用供给全貌指标。\n核心构成支柱：\n1. 表内信贷：人民币贷款和外币贷款（最大基本盘）；\n2. 表外融资：委托贷款、信托贷款、未贴现的银行承兑汇票；\n3. 直接融资：企业债券净融资、境内股票融资（IPO与再融资）；\n4. 政府债券：国债和地方政府专项债净发行；\n先行指标意义：社融存量增速通常领先于名义GDP与企业盈利改善约 1~2 个季度，是判断信贷扩张周期（信用宽/紧）和实体经济见底复苏最重要的晴雨表。"
  },
  {
    "id": "concept_hedging_market_neutral",
    "title": "对冲与市场中性策略（Market Neutral）",
    "category": "quant_finance",
    "keywords": "对冲,市场中性,股票量化对冲,股指期货,Alpha剥离,绝对收益",
    "text": "【金融常识·量化策略】股票量化对冲与市场中性策略（Alpha 策略）\n构建逻辑：追求无论大盘上涨还是暴跌都能获得稳健绝对回报的资产管理策略。\n具体操作步骤：\n1. 多头端：利用多因子选股模型构建一篮子预期跑赢大盘的股票多头组合（产生 Beta + Alpha 收益）；\n2. 空头端：卖出同等市值的股指期货（如沪深300期指IF、中证500期指IC、中证1000期指IM）进行做空对冲（产生 -Beta 收益）；\n3. 盈亏对冲抵消：总收益 = （Beta + Alpha） + （-Beta） = 纯粹的超额 Alpha 收益；\n收益来源与风险：主要赚取选股战胜指数的超额溢价，风险主要来自于期货基差贴水大幅波动成本或多头组合在极端风格切换时跑输指数（负Alpha）。"
  },
  {
    "id": "concept_order_book_depth",
    "title": "内盘外盘、委比量比与买卖盘口",
    "category": "trading_rules",
    "keywords": "内盘,外盘,委比,量比,买卖盘口,主动买单,主动卖单",
    "text": "【金融常识·盘口指标】内盘、外盘、委比与量比的盘口微观含义\n定义与研判：\n1. 外盘（B / 红色）：以卖出报价主动成交的交易量，反映买方急于成交愿意接受卖方要价，体现**主动买盘力量强弱**；\n2. 内盘（S / 绿色）：以买入报价主动成交的交易量，反映卖方急于变现愿意降价成交，体现**主动卖盘打压力度**；\n3. 委比（Commission Ratio） = （委买总手数 - 委卖总手数） / （委买总手数 + 委卖总手数） × 100%。委比为正且很大说明当前买盘挂单积极；但需警惕大资金「虚假挂单撤单」诱多骗线；\n4. 量比（Volume Ratio） = 现成交总手 / （过去5日同时段平均成交手）。量比大于 2~3 倍通常代表当日成交异动放量，资金高度关注。"
  }
]


def main():
    print("=" * 65)
    print("  构建金融基础常识与财务概念词典向量库")
    print("=" * 65)

    # 1. 读取基础词典
    if not DATA_FILE.exists():
        print(f"错误: 找不到概念数据文件 {DATA_FILE}")
        return 1

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    # 合并扩展概念（去重）
    existing_ids = {c["id"] for c in concepts}
    for c in CONCEPTS_EXTENDED:
        if c["id"] not in existing_ids:
            concepts.append(c)
            existing_ids.add(c["id"])

    # 回写完整版
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)

    print(f"[1/4] 整理金融概念词条：共 {len(concepts)} 个核心概念")

    # 2. 检查 ChromaDB
    print(f"[2/4] 连接 ChromaDB: {CHROMA_DB_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    initial_count = col.count()
    print(f"  当前库内文档数: {initial_count}")

    # 3. 向量编码
    print(f"[3/4] 加载 Embedding 模型 ({MODEL_NAME}) 进行向量化...")
    t0 = time.time()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    texts = [c["text"] for c in concepts]
    embeddings = ef(texts)
    print(f"  向量化完成：{len(embeddings)} 条向量（维度 {len(embeddings[0])}），耗时 {time.time() - t0:.1f}s")

    # 4. 写入 ChromaDB
    print(f"[4/4] 写入 ChromaDB (type='concept') ...")
    ids = [c["id"] for c in concepts]
    metas = [
        {
            "type": "concept",
            "category": c.get("category", "general"),
            "title": c.get("title", ""),
            "keywords": c.get("keywords", ""),
            "scope": "concept",
            "company": "金融常识",
            "stock_code": "CONCEPT"
        }
        for c in concepts
    ]

    col.upsert(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeddings
    )

    final_count = col.count()
    print("=" * 65)
    print(f"[OK] 金融概念库注入完成！")
    print(f"  文档总数: {initial_count} -> {final_count}（新增概念词条: {len(concepts)}）")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
