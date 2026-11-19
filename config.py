"""RAG 金融问答系统 - 配置文件"""

import sys
import os

# 接入配置（支持独立运行与共享环境）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.config import MYSQL_CONFIG, PROXY, CHROMA_DB_PATH, COLLECTION_NAME
except ImportError:
    CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db_zh")
    COLLECTION_NAME = "finance_qa"
    MYSQL_CONFIG = {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", 3306)),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "database": os.environ.get("MYSQL_DB", "finance_db_v2")
    }
    PROXY = os.environ.get("PROXY", "")

# ChromaDB 配置：路径单一来源在 shared/config.py（归 rag 所有）

# Embedding 模型配置
# 生产库已迁到 BAAI/bge-small-zh-v1.5（中文区分度远好于英文 MiniLM）。
# chroma collection 元数据里已写死该模型，引擎 get_collection 自动复用，
# 本变量仅作文档说明，运行期不实例化 EF。
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# LLM 配置（使用 ModelScope 免费 API）
# 获取方式：https://modelscope.cn → 右上角头像 → Access Token
LLM_API_BASE = "https://api-inference.modelscope.cn/v1"
LLM_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
LLM_MODEL = "stepfun-ai/Step-3.5-Flash"  # 免费档实测可用；Qwen/Qwen3-8B 已下架（HTTP 400 no provider）
# Step-3.5-Flash 是**推理模型**：先流 reasoning_content 再流正文，旧值 1024 在长 RAG
# 提示词下被思考吃光 ⇒ content 恒空（2026-09-24 V6 线上实测）。关思考参数
# （enable_thinking/thinking:{type:"disabled"}）该端点静默忽略，只能给足预算。
# 两枪实测定值：4096 下茅台题思考涨到 6741 字被截（finish_reason=length，正文 0 字）；
# 8192 下同题 26s 正常收束（completion 共 765 token，思考约 6.5k 字后出正文）。
# 真被截断时引擎按 finish_reason 报 reasoning_exhausted_budget，不再误指 API Key。
#
# 2026-10-15 实测：若拿到 **403 + "please complete your profile via email verification"**，
# 与 API Key 无关 —— 是 ModelScope 账号**未完成邮箱验证**（Magicubes 只对已验证账号开放）。
# 修法：登录 https://modelscope.cn/my/settings/account 完成邮箱验证，无需换 Key。
# 别再走"重装 SDK / 换 key / 查代理"这条弯路，那三条都治不了这个错。
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8192"))


# Web 界面配置
WEB_HOST = "127.0.0.1"  # 2026-09-26 收敛：原值 0.0.0.0 从未被使用（两侧 app.run 均硬编码 127.0.0.1），留着是地雷
WEB_PORT = 5679  # 避免和 n8n 的 5678 冲突

# 检索配置
TOP_K = 5  # 检索最相关的 5 条记录
MAX_CONTEXT_LENGTH = 2000  # 上下文最大 token 数

# ===== 检索管线升级配置（2026-09-11）=====
# 设计原则：**分数量纲显式化** —— score 在管线里被反复覆盖，每覆盖一次量纲就换一次，
# 阈值必须按量纲分开配，禁止跨量纲比较。各段量纲与本库实测/设计值：
#   vector_sim : 1/(1+distance)，0~1（MiniLM 是英文模型，对中文区分度差，
#                相关片段经验值 0.46~0.77，无关片段也常落在 0.5~0.6 —— 不可全信）
#   bm25       : BM25 原始分，无上界 → 只记录名次，不设阈值（与 0~1 量纲不可比）
#   rrf        : 名次融合分，约 1/(RRF_K+rank)，只表名次先后 → 不设阈值
#   rerank     : 启发式复合分，0~1.x，非校准概率 → 只记录，不设阈值
# 总开关：off 时 rag_engine_v2 回退到升级前行为（供评测跑基线用）。
RAG_V2_UPGRADES = os.environ.get("RAG_V2_UPGRADES", "on") != "off"
# 向量兜底通道的相似度阈值，0 = 不过滤只记录。等评测数据出来再定非零默认值，
# 现在拍脑袋设 0.5 之类会把兜底通道整个杀掉（无关片段与相关片段区间重叠严重）。
VECTOR_SIM_THRESHOLD = float(os.environ.get("RAG_VECTOR_SIM_THRESHOLD", "0"))

# BM25 / RRF 参数
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # 经典取值：名次第 1 名 ≈ 1/61，两路同时命中 ≈ 2/61

# 无实体命中时词法兜底二选一：
#   True  = BM25 接管兜底（真实语料下通常优于向量）
#   False = 维持向量兜底（当前默认，实测依据见下）
# 实测（2026-09-11 _rag_eval E 类）：本库是**合成语料**，20 份行业研报共享同一套
# 模板词表 —— 「锂电池」「银行业」「家电」在每个行业研报里都出现，BM25 对它们
# 无区分度（三个不同行业问法返回同一批研报，E01 召回 1.00 → 0.00）；
# 而向量兜底反而把电力设备研报排对第一。语料换成真实研报后再把此项打开。
BM25_AS_FALLBACK = os.environ.get("RAG_BM25_AS_FALLBACK", "off") == "on"

# 父子分块：命中下列类型的子块后，回填**整个父文档**给模型生成（小块用来找，大块用来答）。
# 研报块的父文档 = 同 title 的全部 chunk（入库时带 title+chunk 元数据，无需重新入库）。
# 结构化块（overview/financial/stock）本身是原子块，不参与回填。
PARENT_EXPAND_TYPES = ("research_report",)
PARENT_MAX_CHARS = 1800  # 单个父文档回填的字符上限（防单篇研报吃光上下文预算）

# ===== HyDE 查询改写（2026-09-28，REQ-092）=====
# HyDE = 先让 LLM 写一段"假设这份资料库里存在"的答案段落，再拿它去检索。
# 开关沿用 RRF 的习惯：在调用点读环境变量 `RAG_V2_HYDE=on`（默认 off），
# 这样评测跑批器能在 import 引擎前后任意时刻切换，不受 config 导入顺序影响。
#
# 本库只把改写结果喂给**向量腿**，关键词/实体腿一律用原查询 —— 假设文档里
# 幻觉出来的公司名一旦进关键词腿，就会把 H 类字根混淆（长三角→长江电力）
# 重新灌进 top-k，那是精确率护栏，历史上真栽过。
#
# 默认 off 的两条理由：
#   1) 开启会给**每次检索**前置一次 LLM 调用。本端点（Step-3.5-Flash）是推理模型，
#      实测长提示词下可达数十秒，而 5679 是生产服务，延迟账要先算清。
#   2) 与 BM25_AS_FALLBACK 同理：当前语料是合成模板，22 个计入召回的用例已经
#      100% 命中（见 _rag_eval/eval_baseline.json 的 upgraded 标签），没有上升空间，
#      A/B 只能证明"不变差"。等语料换成真实研报后再评估是否默认开。
HYDE_MAX_TOKENS = int(os.environ.get("RAG_HYDE_MAX_TOKENS", "1024"))
HYDE_MAX_CHARS = int(os.environ.get("RAG_HYDE_MAX_CHARS", "600"))
HYDE_TIMEOUT = int(os.environ.get("RAG_HYDE_TIMEOUT", "60"))

# ===== 检索置信度提示线（2026-10-15 标定，脚本 _rag_eval/calibrate_refusal.py）=====
# 标定结论（**当日两次标定，结论相反，第二次才是对的**）：
#
# 第一次（负例仅 2 条）：正例 top1 0.30~0.95，负例 0.481/0.448 落在正例集群内部，
#   扫 0~2.0 全部门槛综合分无一为正 → 当时判定"不存在可用的拒答门槛"。
#   那个结论是**样本不足导致的假否定**，已作废。
#
# 第二次（负例扩到 12 条，I01-I10 + H01-H02）：
#   正例 22 条：0.30~0.95，其中 20 条 >= 0.515
#   负例 12 条：全部 <= 0.495（分布 0.39~0.495，高度集中）
#   → **0.50 是干净分离点**：低于它 12/12 全是库外问题，高于它 0/12 漏过
#   综合分（负例收益x1 - 正例误杀x3）在 0.50 档 = +6.0，是唯一为正的点
#
# 为什么不抄原文的 0.15：那是另一套校准 rerank 的量纲。本项目 config.py 自己写了
# "启发式复合分，0~1.x，非校准概率"，且本项目明令禁止跨量纲抄阈值。
#
# 语义：top1 < REFUSAL_RERANK → 判定"资料不足"，**不调 LLM**（省 token 且从源头防幻觉），
#       返回明确拒答 + 最接近的片段供人工判断。
# 代价：误杀 2 个正例（0.459 / 0.30）—— 这两个本就是弱命中，代价可接受。
REFUSAL_RERANK = float(os.environ.get("RAG_REFUSAL_RERANK", "0.50"))
