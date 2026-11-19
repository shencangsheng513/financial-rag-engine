# -*- coding: utf-8 -*-
"""REQ-20260928-103（丙三-B）· rag-finance-qa 服务端 JSON 出口去 ASCII 转义。

要钉的事实：Flask 默认 `ensure_ascii=True` 把一个汉字写成「反斜杠 u + 四位十六进制」共 6 B，
UTF-8 只要 3 B。本轮实测（2026-09-28，`_scratch/probe_090b_backend_escape_census.py --self rag`，
两跑载荷逐字节可复算 ⇒ 这一簇的内容是确定性的）四条 JSON 路由 14,704 → 9,106 B，净省 **5,598 B / 38.1%**：

    /api/ask（检索臂）11,092 → 6,880 B（−4,212）    /api/stats   2,757 → 1,674 B（−1,083）
    /api/suggest        575 → 302 B（−273）         /api/health    280 → 250 B（−30）

普查表「税」列写的是 5,602 B：那里面每条路由还含 1 B 尾换行（`jsonify` 必写，这一行省不掉）
⇒ 4 B 不许算进收益，引用时用上面那四个减数。

**这只省传输，不省 token**：喂给对话模型那一份自 REQ-20260927-080 起就是 `tools._utf8_json`
脱过税的，本文件不是 080 的替代品。受益者是演示页 `templates/index.html`（三处 `r.json()`）
与任何直连 5679 的第三方（`shared/watch/engine.py:36` 是一路真消费者）。

两臂分工（沿用 REQ-090 / 丙三-A 的"夹具钉形状 + 真引擎钉现实"）：
- `RagEncodingOffline`（零模型载入，可进每次提交）：运行时开关、**同进程两臂无损对照**、
  消费方解码（requests / httpx 两个真消费者的规则）、旁路闸（AST）、整文件锁步。
  engine 走桩 ⇒ 不调 `create_app()`、不碰向量库、不花额度。
- `RagEncodingLive`：真走 `create_app()`（`engine` 是模块级 `None`，绕过工厂只会量到 500 页），
  只钉"现网载荷里一个转义都不剩 + 载荷确实有肉"。两臂对照**不在真臂做**：每条路由都要过一遍
  向量检索，两臂就是两次 ⇒ 无损那一半由夹具臂在同进程里负责。
"""

import ast
import copy
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT.parent
SRC = ROOT / "src" / "web_app.py"
MIRROR = SUITE / "moml-cloud" / "data-plane" / "rag-finance-qa" / "src" / "web_app.py"
DRIFT_MANIFEST = SUITE / "moml-cloud" / "drift-manifest.json"

MARK = "app.json.ensure_ascii = False"
REQ = "REQ-20260928-103"
ESCAPE_TOKEN = re.compile(r"\\u[0-9a-fA-F]{4}")

for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import web_app as appmod
except ModuleNotFoundError as _exc:      # 只让"缺件"跳过；语法错/属性错必须红
    raise unittest.SkipTest("需 rag 依赖链可 import（用 D:/ML_Projects/.venv 解释器跑）：%s" % _exc)

try:
    import httpx
except ModuleNotFoundError:
    httpx = None

import requests.utils as _req_utils


# ===== 夹具：形状照真 /api/ask 的载荷造，中文密度照现网量 =====
_ASK_FIXTURE = {
    "query": "白酒板块估值逻辑是什么？",
    "answer": "白酒板块的估值锚主要是渠道库存与批价：贵州茅台直销占比提升后毛利率抬升。",
    "sources": [
        {   # 聚合/排行臂：distance 是 None（没走向量检索），前端靠 rank 显示"第N名"
            "id": "report_081",
            "text": "高端白酒的批价与动销数据说明渠道去库存仍在进行。" * 3,
            "metadata": {"type": "research_report", "sector": "白酒", "scope": "industry"},
            "distance": None,
            "agg_rank": 1,
        },
        {   # 常规文档块：带 company / stock_code / trace
            "id": "doc_002",
            "text": "贵州茅台 2025 年年度报告：营业收入 1,741 亿元，归母净利润 862 亿元。",
            "metadata": {"type": "annual_report", "company": "贵州茅台",
                         "stock_code": "SH600519", "scope": "company"},
            "distance": 0.2134,
            "trace": {"vector": 0.7866, "bm25": 12.3, "rerank": 0.91},
        },
    ],
}
_STATS_FIXTURE = {"总文档块": 12345, "公司数": 88,
                  "行业分布": {"白酒": 12, "科技": 34, "医药": 21},
                  "说明": "数值来自本地 Chroma 向量库，随入库轮次变化"}


class _StubCollection:
    def count(self):
        return 12345


class _StubEngine:
    """桩 engine：只负责把四条路由的响应体喂出来，不含任何检索逻辑。

    ⚠️ 每次返回都拷开 —— 两臂对照要在同进程里把同一份夹具取两次，
    丙三-A 那轮就是被 `src/app.py` 的一句就地 `pop` 把第二次读数啃成了 0（教训 93）。
    """

    def __init__(self):
        self.collection = _StubCollection()

    def ask(self, query, use_llm=True):
        return copy.deepcopy(_ASK_FIXTURE)

    def stats(self):
        return copy.deepcopy(_STATS_FIXTURE)


# 本文件认领并**真打**的 JSON 路由（末位是期望状态；名单与 test_05 的 url_map 对账）
CASES = (
    ("/api/suggest", "get", {}, 200),
    ("/api/health", "get", {}, 200),
    ("/api/stats", "get", {}, 200),
    ("/api/ask", "post", {"json": {"query": "白酒板块估值逻辑是什么？", "use_llm": False}}, 200),
    # /api/ingest 的**错误臂**：`require_key("write")` 的两条出口（未配 RAG_WRITE_API_KEY ⇒ 503
    # fail-closed；配了但 key 不符 ⇒ 401）都在抢 `_ingest_lock` 和起后台线程**之前** return，
    # 所以这一条既覆盖了一条真路由的 JSON 错误体，又保证零副作用（绝不可能触发重建向量库）。
    ("/api/ingest", "post",
     {"json": {}, "headers": {"X-API-Key": "__definitely_not_the_configured_key__"}}, (401, 503)),
)
# 墙钟/易变键的豁免**按路由**列（丙三-A 踩过：全局表会把没有该键的路由判成红）
WALL_CLOCK_KEYS = {
    "/api/health": ("checked_at",),
    "/api/suggest": (),
    "/api/stats": (),
    "/api/ask": (),
    "/api/ingest": (),
}
CASE_BY_PATH = {c[0]: c for c in CASES}


def _drop_volatile(obj, path):
    out = copy.deepcopy(obj)
    for key in WALL_CLOCK_KEYS[path]:
        assert key in out, "%s 声明豁免了墙钟键 %s，但它不在响应里 ⇒ 豁免表过期了" % (path, key)
        out.pop(key)
    return out


class RagEncodingOffline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = appmod.app
        cls.client = cls.app.test_client()
        cls._engine_before = appmod.engine
        # 两个 key 在 import 时从环境读（web_app.py:43-44）⇒ 机器上配了 key，这四条正常臂就变 401、
        # /api/ingest 的错误臂也会在 401/503 之间漂。夹具臂只判编码形状，这里临时清空并原样还回。
        cls._keys_before = (appmod.READ_API_KEY, appmod.WRITE_API_KEY)
        appmod.READ_API_KEY = appmod.WRITE_API_KEY = ""
        appmod.engine = _StubEngine()

    @classmethod
    def tearDownClass(cls):
        appmod.READ_API_KEY, appmod.WRITE_API_KEY = cls._keys_before
        appmod.engine = cls._engine_before

    def _get(self, case):
        path, method, kwargs, expected = case
        r = getattr(self.client, method)(path, **kwargs)
        codes = expected if isinstance(expected, tuple) else (expected,)
        self.assertIn(r.status_code, codes, "%s 实得 HTTP %s（期望 %s）" % (path, r.status_code, codes))
        return r

    def test_01_runtime_switch(self):
        """问运行时对象本人，不读源码字符串猜。"""
        self.assertIs(False, self.app.json.ensure_ascii,
                      "app.json.ensure_ascii 必须是 False（Flask 3 的正解；老写法 json_encoder 已废弃）")
        self.assertIsNone(getattr(self.app, "json_encoder", None),
                          "不许同时用已废弃的 json_encoder 覆盖 app.json 这一层")

    def test_02_no_escape_left(self):
        """五条真路由（含 /api/ingest 的错误臂）的 body 里一个转义段都不剩。"""
        for case in CASES:
            raw = self._get(case).get_data(as_text=True)
            self.assertEqual([], ESCAPE_TOKEN.findall(raw), "%s 还在写转义段" % case[0])

    def test_03_lossless_shape_only(self):
        """只动编码形状，不动内容：同进程把开关拨回 True 再取一份对账。"""
        for case in CASES:
            path = case[0]
            clean = self._get(case)
            clean_raw = clean.get_data(as_text=True)
            self.app.json.ensure_ascii = True
            try:
                esc = self._get(case)
            finally:
                self.app.json.ensure_ascii = False
            esc_raw = esc.get_data(as_text=True)
            esc_segs = ESCAPE_TOKEN.findall(esc_raw)
            self.assertGreater(len(esc_raw.encode("utf-8")), len(clean_raw.encode("utf-8")),
                               "%s 的转义臂不比脱税臂大 ⇒ 这一路由压根没有中文载荷，本表在骗人" % path)
            self.assertGreater(len(esc_segs), 0, "%s 转义臂段数为 0，同上" % path)
            self.assertEqual(_drop_volatile(clean.get_json(), path),
                             _drop_volatile(esc.get_json(), path),
                             "%s 两臂内容不等 ⇒ 这一行改坏了内容，不是改形状" % path)
            self.assertEqual(0, len(ESCAPE_TOKEN.findall(clean_raw)),
                             "%s 脱税臂仍有转义段" % path)
        self.assertIs(False, self.app.json.ensure_ascii, "拨完没拨回来（finally 里那句也算这条闸）")

    def test_04_transport_contract(self):
        """传输契约没变：合法 UTF-8、Content-Type 仍是 application/json、尾换行形状不变，
        且**这个头**在两个真消费者的解码规则下都判成 utf-8。"""
        for case in CASES:
            r = self._get(case)
            body = r.get_data()
            self.assertEqual("application/json", r.headers.get("Content-Type"),
                             "%s 的 Content-Type 变了 ⇒ 下游按什么字符集解要重新谈" % case[0])
            body.decode("utf-8")            # 炸 = 非 UTF-8
            self.assertTrue(body.endswith(b"\n"), "%s 尾换行形状变了" % case[0])

        # 头里没有 charset，所以"能不能读对中文"完全取决于消费者怎么猜。
        # 两个真消费者：requests（各评估脚本）与 httpx（moml-cloud/agent_service/tools.py::_get）。
        r = self._get(CASE_BY_PATH["/api/suggest"])
        ct = r.headers.get("Content-Type")
        body = r.get_data()
        self.assertTrue(any(b > 127 for b in body),
                        "body 全 ASCII ⇒ 这一路没有中文载荷，下面的解码判据是空的")
        self.assertEqual("utf-8", _req_utils.get_encoding_from_headers({"content-type": ct}),
                         "requests 对这个头不是判 utf-8 ⇒ 脱税后的 body 会被它解成乱码")
        if httpx is None:
            self.skipTest("本环境没有 httpx，tools.py 那条消费链的解码规则没测到（不是通过）")

        def _echo(request):
            return httpx.Response(200, headers={"content-type": ct}, content=body)

        with httpx.Client(transport=httpx.MockTransport(_echo)) as client:
            got = client.get("http://stub/api/suggest")
        self.assertEqual(body.decode("utf-8"), got.text,
                         "httpx 按这个头解不出原文 ⇒ agent 侧 _get(r.text) 会拿到乱码")

    def test_05_no_bypass(self):
        """旁路闸（AST）：这个 app 里没有任何"绕过 app.json 自己拼 body"的站点，
        且每条 /api/* 视图的 return 都没被提前拼成字符串（走 jsonify 或裸 dict/list 都算合格）。

        名单走本文件自己写的 CASES（五条 /api/* 全部真打），不抄实现；新增路由不登记就红。
        """
        tree = ast.parse(SRC.read_text(encoding="utf-8"), filename=str(SRC))

        def _dotted(node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))

        dumps, bypass, render = [], [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if name in ("Response", "make_response", "response_class", "app.response_class"):
                bypass.append("%s@%d" % (name, node.lineno))
            # 按**最后一段**判，不按点号全形判：`from json import dumps` 之后裸调 `dumps(...)`
            # 与 `json.dumps(...)` 是同一件事（变异 M6 的第一版就是靠别名绕过了全形判据）。
            if name.split(".")[-1] in ("dumps", "dump"):
                dumps.append(node.lineno)
            if name == "render_template":
                render.append(node.lineno)

        self.assertEqual([], dumps,
                         "出现了手写 json.dumps（行 %s）⇒ 它绕开 app.json，这行开关对它无效" % dumps)
        self.assertEqual([], bypass,
                         "出现了 Response/make_response/response_class 自建 body（%s）⇒ 同上，须逐个登记 mimetype" % bypass)

        # 1) 路由名单：url_map 里所有 /api/* 必须被本文件的 CASES 认领（每条都真打）
        api_rules = {str(rule) for rule in self.app.url_map.iter_rules() if str(rule).startswith("/api/")}
        claimed = {c[0] for c in CASES}
        self.assertEqual(claimed, api_rules,
                         "路由名单与本文件认领的不一致：url_map 独有 %s / 名单独有 %s"
                         % (sorted(api_rules - claimed), sorted(claimed - api_rules)))

        # 2) 每条 /api/* 视图的 return 形状必须仍经过 app.json
        #    （REQ-090 §3.4 的纠正：`return jsonify(x)` 与 `return x`（dict/list）都经过 app.json，
        #     "必须走 jsonify"是过严的判据 —— 会把一次合法重构判成红。真正的绕过只有"提前序列化成字符串"。）
        view_names = []
        for rule in self.app.url_map.iter_rules():
            path = str(rule)
            if not path.startswith("/api/"):
                continue
            view_names.append(self.app.view_functions[rule.endpoint].__name__)
        funcs = {n: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in [node.name]}
        allowed = {"jsonify", "name", "dict", "list", "tuple(jsonify)", "tuple(name)",
                   "tuple(dict)", "tuple(list)"}

        def _shape(node):
            if isinstance(node, ast.Tuple):
                return "tuple(%s)" % _shape(node.elts[0])
            if isinstance(node, ast.Call):
                return _dotted(node.func) or type(node.func).__name__
            if isinstance(node, ast.Dict):
                return "dict"
            if isinstance(node, ast.List):
                return "list"
            if isinstance(node, ast.Name):
                return "name"
            return type(node).__name__

        for name in view_names:
            fn = funcs.get(name)
            self.assertIsNotNone(fn, "视图函数 %s 在源码里找不到（被重构走了？）" % name)
            returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
            self.assertGreater(len(returns), 0, "%s 一个 return 都没有" % name)
            for ret in returns:
                shape = _shape(ret.value)
                self.assertIn(shape, allowed,
                              "%s 第 %d 行返回的是 %s ⇒ body 可能已经拼好，这行开关对它无效" % (name, ret.lineno, shape))

        # 3) 被排除的面也要有正向计数（不是"它不在名单所以看不见"）
        non_api = [str(rule) for rule in self.app.url_map.iter_rules()
                   if not str(rule).startswith("/api/")]
        self.assertEqual(["/", "/static/<path:filename>"], sorted(non_api),
                         "非 JSON 出口的形状变了（现在只有主页 HTML 与静态资源）")
        self.assertEqual(1, len(render), "render_template 调用数不是 1 ⇒ HTML 出口在长")

    def test_06_lockstep_whole_file(self):
        """这一对的锁步单位是**整文件**"""
        if not MIRROR.exists():
            self.skipTest("Standalone environment: deployment mirror not present")
        self.assertTrue(MIRROR.exists(), "部署包副本不存在：%s" % MIRROR)
        root_bytes = SRC.read_bytes()
        mirror_bytes = MIRROR.read_bytes()
        self.assertEqual(root_bytes, mirror_bytes,
                         "根侧与部署包整文件不一致（根 %d B / 包 %d B）⇒ 线上跑的是包那份"
                         % (len(root_bytes), len(mirror_bytes)))
        for label, text in (("根侧", root_bytes.decode("utf-8")), ("部署包", mirror_bytes.decode("utf-8"))):
            self.assertEqual(1, text.count(MARK), "%s 的开关行应当恰好一次，实得 %d 次"
                             % (label, text.count(MARK)))
            self.assertIn(REQ, text, "%s 的注释块没带工单号 %s" % (label, REQ))
        self.assertTrue(DRIFT_MANIFEST.exists(), "漂移清单不在了，test_06 的收回路径说明失效：%s" % DRIFT_MANIFEST)


class RagEncodingLive(unittest.TestCase):
    """真引擎臂：走 `create_app()`（绕过工厂只量到 500 页）。载不动就明说没测到。"""

    # 地板取现网量值的一半（丙三-B 现量：suggest 302 / health 250 / stats 1,674 / ask 6,880 B）
    FLOORS = {"/api/suggest": 200, "/api/health": 150, "/api/stats": 800, "/api/ask": 3000}

    @classmethod
    def setUpClass(cls):
        try:
            cls.app = appmod.create_app()
        except Exception as e:
            raise unittest.SkipTest("没测到：真引擎起不来（%s: %s）" % (type(e).__name__, str(e)[:160]))
        cls.client = cls.app.test_client()
        # 读接口默认不配 key（现测 RAG_API_KEY 长度 0）；配了就得带，否则四条全是 401 的空账
        cls.headers = {"X-API-Key": appmod.READ_API_KEY} if appmod.READ_API_KEY else {}

    def test_07_live_payload_has_no_escape(self):
        self.assertIs(False, self.app.json.ensure_ascii)
        total = 0
        for path, floor in sorted(self.FLOORS.items()):
            _claimed, method, kwargs, _expected = CASE_BY_PATH[path]
            r = getattr(self.client, method)(path, headers=self.headers, **kwargs)
            self.assertEqual(200, r.status_code,
                             "%s 实得 HTTP %s（读接口 key 长度 %d）"
                             % (path, r.status_code, len(appmod.READ_API_KEY)))
            body = r.get_data(as_text=True)
            size = len(body.encode("utf-8"))
            total += size
            json.loads(body)                        # 炸 = 这条路由不再回合法 JSON
            self.assertGreaterEqual(size, floor,
                                    "%s 只有 %d B（地板 %d）⇒ 载荷空了，0 转义不算证据" % (path, size, floor))
            self.assertEqual([], ESCAPE_TOKEN.findall(body), "%s 现网载荷里还有转义段" % path)
        self.assertGreater(total, 8000, "四条路由合计 %d B，丙三-B 的账是 9,106 B 量级" % total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
