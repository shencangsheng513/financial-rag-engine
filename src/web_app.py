"""
RAG 金融问答系统 - Web 界面

2026-09-11 安全加固。背景：线上 /rag/api/* 全部端点曾**公网免鉴权**
（实测裸 curl 即 200 返回全量数据；/api/ask 的 use_llm 默认 True ⇒ 任何人可匿名烧 LLM 配额）。
本文件的处置：

  - `/api/ingest` 是**写**接口 —— 必须带 key（`RAG_WRITE_API_KEY`），未配置则 **fail-closed**
    拒掉全部写请求（返回 503），绝不"没配 key 就放行"。
  - `/api/ask` / `/api/stats` / `/api/suggest` 受**可选**总开关 `RAG_API_KEY` 保护：
    **设了才校验**，不设则维持原样。默认关是有意的 —— 这是个公开演示页，
    一上来就强制 key 会把页面打死；要收口时在平台环境变量设 `RAG_API_KEY` 即可。
  - `/api/health` 与 `/` **始终放行**：前者是平台端口健康检查（拦了会被判不健康而摘流量），
    后者要能加载，否则页面没法提示用户输入 key。
  - LLM 花费封顶：每分钟 / 每天令牌桶，超限**降级为纯检索**（`use_llm=False` + `degraded: true`）
    而不是返 5xx —— "答案质量降级"比"整站报错"更可用，成本仍然是硬的。
    （按 IP 限流在这里没有意义：router.py 只转发 Content-Type/Accept，不传 X-Forwarded-For，
    后端看到的 client 永远是 127.0.0.1。要做精确到 IP 的限制得改 router.py，那是另一层的事。）
"""

import sys
import os
import hmac
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from functools import wraps
from flask import Flask, render_template, request, jsonify
from rag_engine_v2 import FinanceRAGEngineV2
from config import WEB_HOST, WEB_PORT, CHROMA_DB_PATH

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
            static_folder=_STATIC_DIR, static_url_path="/static")

# Flask 默认 ensure_ascii=True：一个汉字写成「反斜杠 u + 四位十六进制」共 6 B，UTF-8 只要 3 B。
# 本应用四条 JSON 路由实测省 5,598 B（现网 14,704 → 9,106 B，占 38.1%；探针
# _scratch/probe_090b_backend_escape_census.py，大头在 /api/ask 的检索臂 4,212 B 与 /api/stats 1,083 B。
# 普查表的"税"列写的是 5,602 B —— 那里面每条路由还含 1 B 尾换行，`jsonify` 的尾换行省不掉，别把它当承诺值）。
# moml-cloud 的 **agent 出口** 自 REQ-20260927-080 起已脱税 ⇒ 这一行只省「服务端 → 本演示页 / 第三方调用者」
# 那一跳的传输，**不省 token**，不是 080 的重复记账。工单 REQ-20260928-103（守卫 tests/test_json_encoding.py）。
# ⚠ 部署包副本 moml-cloud/data-plane/rag-finance-qa/src/web_app.py 与根侧**整文件逐字节相同**
# （这一对没有声明过分叉，锁步单位是整文件 —— 和 finance-webapp 那一对"按块判定、禁止 cp"正相反）
# ⇒ 改完这一份要 `cp` 过去；只改根、线上零变化是本项目最贵的陷阱（守卫 test_06 钉这一条）。
app.json.ensure_ascii = False

# 全局 RAG 引擎实例
engine = None


# ===== 鉴权 =====
WRITE_API_KEY = os.environ.get("RAG_WRITE_API_KEY", "")   # 写接口，未配 = 503 fail-closed
READ_API_KEY = os.environ.get("RAG_API_KEY", "")          # 读接口总开关，未配 = 保持公开


def _presented_key():
    """取调用方出示的 key：优先 X-API-Key，其次 Authorization: Bearer。"""
    k = request.headers.get("X-API-Key", "")
    if not k:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            k = auth[7:]
    return k


def _key_matches(expected):
    """恒定时间比较，防时序侧信道。"""
    return bool(expected) and hmac.compare_digest(_presented_key(), expected)


def require_key(scope="write"):
    """scope="write"：必须配 `RAG_WRITE_API_KEY`，没配直接 503（fail-closed）。
    scope="read" ：仅当 `RAG_API_KEY` 非空时才校验；为空则放行（默认不启用）。"""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            expected = WRITE_API_KEY if scope == "write" else READ_API_KEY
            if not expected:
                if scope == "write":
                    return jsonify({
                        "error": "服务端未配置 RAG_WRITE_API_KEY，已拒绝全部写请求（fail-closed）",
                        "how": "在平台环境变量设置 RAG_WRITE_API_KEY 后重启",
                    }), 503
                return fn(*args, **kwargs)
            if not _key_matches(expected):
                return jsonify({"error": "未授权：缺少或错误的 X-API-Key"}), 401
            return fn(*args, **kwargs)
        return wrapper
    return deco


# ===== LLM 花费上限（全局令牌桶）=====
LLM_PER_MIN = int(os.environ.get("RAG_LLM_PER_MIN", "30"))
LLM_PER_DAY = int(os.environ.get("RAG_LLM_PER_DAY", "300"))  # 0 = 该维度不限
_llm_lock = threading.Lock()
_llm_min = [0.0, 0]                                  # [窗口起始, 窗口内计数]
_llm_day = [time.strftime("%Y-%m-%d"), 0]            # [日期, 当日计数]


def llm_budget_ok():
    """申请一次 LLM 配额，返回 (是否允许, 拒绝原因)。两个维度都 <=0 时不限。"""
    if LLM_PER_MIN <= 0 and LLM_PER_DAY <= 0:
        return True, ""
    now = time.time()
    with _llm_lock:
        if LLM_PER_MIN > 0:
            if now - _llm_min[0] >= 60:
                _llm_min[0], _llm_min[1] = now, 0
            if _llm_min[1] >= LLM_PER_MIN:
                return False, f"每分钟 LLM 调用上限 {LLM_PER_MIN} 已到"
            _llm_min[1] += 1
        if LLM_PER_DAY > 0:
            today = time.strftime("%Y-%m-%d")
            if _llm_day[0] != today:
                _llm_day[0], _llm_day[1] = today, 0
            if _llm_day[1] >= LLM_PER_DAY:
                return False, f"当日 LLM 调用上限 {LLM_PER_DAY} 已到"
            _llm_day[1] += 1
    return True, ""


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
@require_key("read")
def api_ask():
    """问答 API"""
    data = request.json or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "请输入问题"}), 400

    # 长度上限：防止用超长 query 拖长上下文、放大单次成本
    max_len = int(os.environ.get("RAG_MAX_QUERY_LEN", "200"))
    if len(query) > max_len:
        return jsonify({"error": f"问题过长（{len(query)} 字），上限 {max_len} 字"}), 400

    use_llm = data.get("use_llm", True)
    degraded = False
    notice = ""
    if use_llm:
        allowed, why = llm_budget_ok()
        if not allowed:
            # 超预算不报错，降级为纯检索：可用性与成本两头都保住
            use_llm, degraded, notice = False, True, why

    result = engine.ask(query, use_llm=use_llm)

    payload = {
        "query": result["query"],
        "answer": result["answer"],
        "sources": [
            {
                "id": s["id"],
                "text": s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"],
                "type": s["metadata"].get("type", "?"),
                # company 缺失时回退到 sector：行业研报（report_081–089）不属于任何公司，
                # 只带 sector/scope，直接显示 "?" 会让引用一行看不出讲的是什么；概念常识块展示词条名称。
                "company": (
                    f"金融常识·{s['metadata']['title']}" if s["metadata"].get("type") == "concept" and s["metadata"].get("title")
                    else s["metadata"].get("company") or (
                        f"{s['metadata']['sector']}（行业）" if s["metadata"].get("sector") else "?")
                ),
                "stock_code": (
                    s["metadata"].get("category", "基础概念") if s["metadata"].get("type") == "concept"
                    else s["metadata"].get("stock_code", "?")
                ),
                "scope": s["metadata"].get("scope", "company"),
                # 聚合/排行命中的 distance 是 None（它没走向量检索，没有"距离"可言），
                # 直接 `1 - None` 会 TypeError → 整个 /api/ask 500。
                # 这类结果有意义的信号是**名次**，不是相似度，故另给 rank 字段，
                # 前端优先显示"第N名"（见 templates/index.html）。
                "score": (round(1 - s["distance"], 4)
                          if s.get("distance") is not None else None),
                "rank": s.get("agg_rank"),
                # 行业清单（REQ-050）也带 agg_rank，但那是**序号**不是名次——名单刻意按公司名排序。
                # 前端靠这个标记把徽标从"第N名"换成"第N家"，否则页面会把无排行的清单读成排行。
                "roster": bool(s.get("_roster")),
                # StageTrace（2026-09-11）：每个来源的分数量纲链条，调试/展示用。
                # 缺失（如聚合结果旧缓存）时为 null，前端按无 trace 渲染。
                "trace": s.get("trace"),
            }
            for s in result["sources"]
        ]
    }
    if degraded:
        payload["degraded"] = True
        payload["notice"] = f"{notice}，本次已降级为纯检索（未调用大模型）。"
    return jsonify(payload)


@app.route("/api/stats")
@require_key("read")
def api_stats():
    """数据库统计"""
    stats = engine.stats()
    return jsonify(stats)


@app.route("/api/health")
def api_health():
    """数据健康检查：RAG 引擎与向量库状态。

    规约同 market-monitor：只读；独立降级，engine 未就绪不抛 500。
    """
    import datetime

    now = datetime.datetime.now()
    sources = []
    ok = True
    if engine is None:
        ok = False
        sources.append({"label": "RAG 引擎", "status": "empty",
                        "human": "引擎未初始化", "rows": 0, "latest": None})
    else:
        try:
            count = engine.collection.count()
            sources.append({
                "label": "向量知识库",
                "status": "fresh" if count > 0 else "empty",
                "human": f"共 {count} 个文档块",
                "rows": count,
                "latest": None,
            })
        except Exception as e:
            ok = False
            sources.append({"label": "向量知识库", "status": "empty",
                            "human": f"查询失败：{type(e).__name__}", "rows": 0, "latest": None})

    return jsonify({
        "service": "rag-finance-qa",
        "checked_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "db_ok": ok,
        "sources": sources,
        "ingest": dict(INGEST_STATUS),
    })


@app.route("/api/suggest")
@require_key("read")
def api_suggest():
    """推荐问题"""
    suggestions = [
        "科技行业有哪些公司？",
        "哪家公司的市值最高？",
        "苹果公司最近的财务状况怎么样？",
        "哪家公司的ROE最高？",
        "最近哪只股票涨得最好？",
        "哪些公司的资产负债率比较高？",
        "哪家公司员工最多？",
        "最近有大额交易的公司是哪家？",
    ]
    return jsonify(suggestions)


# 入库并发锁：防止定时器与手动触发同时重建向量库
_ingest_lock = threading.Lock()

# 入库状态回填：202 只代表"已开始"，成败要能在 /api/health 查到（不留静默失败）
INGEST_STATUS = {"status": "idle", "detail": None, "finished_at": None}


@app.route("/api/ingest", methods=["POST"])
@require_key("write")
def api_ingest():
    """触发研报入库到 Chroma 向量库（供 CloudBase 定时函数调用）。

    ingest 可能数分钟，丢后台线程，立即返 202。

    ⚠️ 两个必须知道的限制（2026-09-11）：
      1. 向量库是**随镜像打包**的（Dockerfile `COPY rag-finance-qa/`），没有挂载持久卷，
         所以这里写进去的块**重启即失**。它的价值是"本次运行期内补数据"，
         要持久化得先在平台上挂卷、再把 RAG_CHROMA_DB_PATH 指过去。
      2. 入库后必须重建 engine，否则运行中的 collection 仍持旧的内存索引、新块查不到。

    路径来自 ingest_reports 读的环境变量，这里用 shared.config 的值兜底，
    保证容器里不落到源码里那个开发机的 D:/ 默认值。
    """
    if not _ingest_lock.acquire(blocking=False):
        return jsonify({"error": "入库正在运行，请稍后", "status": "running"}), 409

    def _job():
        global engine
        import datetime
        INGEST_STATUS.update(status="running", detail=None, finished_at=None)
        try:
            # ingest_reports 在 **import 时** 读环境变量，所以必须 import 之前设好
            os.environ.setdefault("RAG_CHROMA_DB_PATH", CHROMA_DB_PATH)
            from tools.ingest_reports import main as ingest_main
            ingest_main()
            engine = FinanceRAGEngineV2()   # 重建，让新块对查询可见
            print("[ingest] done, engine rebuilt", flush=True)
            INGEST_STATUS.update(status="ok", detail=None,
                                 finished_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            print(f"[ingest] 失败: {type(e).__name__}: {e}", flush=True)
            INGEST_STATUS.update(status="failed", detail=f"{type(e).__name__}: {e}"[:200],
                                 finished_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        finally:
            _ingest_lock.release()

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"ok": True, "status": "started"}), 202


def create_app():
    """创建 Flask 应用"""
    global engine, WRITE_API_KEY, READ_API_KEY, LLM_PER_MIN, LLM_PER_DAY
    print("=" * 60)
    print("  RAG 金融问答系统")
    print("=" * 60)
    
    # 加载 .env 文件
    # .env 查找顺序：本目录（旧布局兼容）→ 项目根（2026-09-11 移出，防明文 Key 进 code.zip）
    env_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), ".env"),
    ]
    env_path = next((p for p in env_candidates if os.path.exists(p)), None)
    if env_path:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)
        api_key = os.environ.get("MODELSCOPE_API_KEY", "")
        print(f"  API Key: {'已配置' if api_key else '未配置'} ({len(api_key)} chars)")
    else:
        print("  [WARN] .env 文件不存在")

    # 安全相关环境变量在上面的 .env 加载**之后**再读一次：
    # 模块级读取发生在 import 时，那时 .env 还没载入，本地跑会全部读成空。
    # （容器里由平台 EnvParams 注入，两种顺序都能拿到；.env 走 setdefault，平台优先。）
    WRITE_API_KEY = os.environ.get("RAG_WRITE_API_KEY", "")
    READ_API_KEY = os.environ.get("RAG_API_KEY", "")
    LLM_PER_MIN = int(os.environ.get("RAG_LLM_PER_MIN", "30"))
    LLM_PER_DAY = int(os.environ.get("RAG_LLM_PER_DAY", "300"))
    print(f"  [auth] 写接口 RAG_WRITE_API_KEY : {'已配置' if WRITE_API_KEY else '未配置（写接口一律 503）'}")
    print(f"  [auth] 读接口 RAG_API_KEY       : {'已配置（/api/ask|stats|suggest 需 key）' if READ_API_KEY else '未配置（读接口公开）'}")
    print(f"  [cost] LLM 限额                 : {LLM_PER_MIN}/分钟, {LLM_PER_DAY}/天（0=不限）")
    print(f"  [cost] 单问上限                 : {os.environ.get('RAG_MAX_QUERY_LEN', '200')} 字")

    engine = FinanceRAGEngineV2()
    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n[WEB] 界面启动：http://localhost:{WEB_PORT}")
    print("   按 Ctrl+C 停止\n")
    # 安全加固：debug=False + 仅本机监听（原 debug=True 且 0.0.0.0 存在 RCE 风险）
    app.run(host="127.0.0.1", port=WEB_PORT, debug=False, use_reloader=False)
