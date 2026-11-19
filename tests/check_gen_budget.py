# -*- coding: utf-8 -*-
"""生成侧预算修复的确定性守卫 —— 零网络，rc 0/1。

病灶（2026-09-24 V6 线上实测）：Step-3.5-Flash 是推理模型，旧 payload
max_tokens=1024 被 reasoning_content 吃光 ⇒ content 恒空，引擎却报
「请检查 API Key 和模型名称」把排障方向带偏。

守卫两层（文本闸会瞎，行为闸才算数 —— 见套件记忆规则 21）：
  S1 静态：root+bundle 两侧 config 定义 LLM_MAX_TOKENS；src 里不再有写死的
     "max_tokens": 1024；v2/v1 都捕获 finish_reason 与 reasoning_content；
     「检查 API Key」在每份引擎里至多 1 处（不许回潮成唯一空答案出口）；
     v2 两侧逐字节锁步。
  S2 行为：用假 SSE 流直接驱动 generate_with_modelscope 三态——
     a) 思考+正文都有 ⇒ 正文返回、error=None；
     b) 只有思考 + finish_reason=length ⇒ 诚实的「预算花在推理上」提示
        （且必须引用真实预算数字与思考字数），error=reasoning_exhausted_budget；
     c) 啥都没有、无 finish ⇒ 才允许说 API Key，error=empty_answer。
     三态同时断言 log_generation 收到的 error 值。

跑：cd rag-finance-qa && PYTHONIOENCODING=utf-8 python tests/check_gen_budget.py
"""
import io
import json
import re
import sys
import traceback
from pathlib import Path

APP = Path(__file__).resolve().parent.parent          # rag-finance-qa/
BUN = APP.parent / "moml-cloud" / "data-plane" / "rag-finance-qa"
sys.path.insert(0, str(APP))          # config.py 在这里
sys.path.insert(0, str(APP / "src"))  # 引擎模块在这里

FAILS = []


def check(name, cond, detail=""):
    tag = "OK  " if cond else "[FAIL] "
    print(f"{tag}{name}" + (f"  —— {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---------- S1 静态闸 ----------

def static_gates():
    cfg_root = (APP / "config.py").read_text(encoding="utf-8")
    cfg_bun = (BUN / "config.py").read_text(encoding="utf-8")
    for side, txt in (("root", cfg_root), ("bundle", cfg_bun)):
        m = re.search(r'LLM_MAX_TOKENS\s*=\s*int\(os\.environ\.get\("LLM_MAX_TOKENS",\s*"(\d+)"\)\)', txt)
        dv = int(m.group(1)) if m else None
        check(f"S1[{side}] config 定义 LLM_MAX_TOKENS（env 可覆盖，默认 >=8192）",
              dv is not None and dv >= 8192,
              f"实际默认 {dv}（4096 下茅台题思考 6741 字被截，8192 实测收束）")

    for side, src_dir in (("root", APP / "src"), ("bundle", BUN / "src")):
        for f in ("rag_engine.py", "rag_engine_v2.py"):
            st = (src_dir / f).read_text(encoding="utf-8")
            check(f"S1[{side}/{f}] 不再有写死的 max_tokens=1024",
                  '"max_tokens": 1024' not in st)
            check(f"S1[{side}/{f}] payload 用 LLM_MAX_TOKENS",
                  '"max_tokens": LLM_MAX_TOKENS' in st)
            check(f"S1[{side}/{f}] 捕获 finish_reason（赋值式，注释不算）",
                  '_finish_reason = choices[0]["finish_reason"]' in st)
            check(f"S1[{side}/{f}] 计数 reasoning_content",
                  "_reasoning_chars += len(delta[\"reasoning_content\"])" in st)
            n = st.count("请检查 API Key")
            check(f"S1[{side}/{f}] 「检查 API Key」只剩 1 处兜底", n == 1, f"实际 {n} 处")
    v2r = (APP / "src" / "rag_engine_v2.py").read_bytes()
    v2b = (BUN / "src" / "rag_engine_v2.py").read_bytes()
    check("S1 锁步：v2 root == bundle 逐字节", v2r == v2b)


# ---------- S2 行为闸（假 SSE，不碰网） ----------

class _FakeResp:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def _sse(reasoning_pieces, content_pieces, finish, usage=None):
    """按 ModelScope 线格式（2026-09-24 实测：delta 里有 reasoning_content/
    content 两种键，finish_reason 在 choices[0]）拼 SSE 字节行。"""
    out = []
    for p in reasoning_pieces:
        out.append(b"data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": p}, "finish_reason": None}]}
        ).encode())
    for p in content_pieces:
        out.append(b"data: " + json.dumps(
            {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
        ).encode())
    if finish:
        out.append(b"data: " + json.dumps(
            {"choices": [{"delta": {}, "finish_reason": finish}]}).encode())
    if usage:
        out.append(b"data: " + json.dumps({"choices": [], "usage": usage}).encode())
    out.append(b"data: [DONE]")
    return out


class _FakeRequests:
    def __init__(self, lines):
        self.lines = lines
        self.bodies = []

    def post(self, url, **kw):
        self.bodies.append(kw.get("json"))
        return _FakeResp(self.lines)


def behavioral_gates():
    import rag_engine_v2 as eng

    logged = {}

    def fake_log(**kw):
        logged.clear()
        logged.update(kw)

    orig_requests, orig_log = eng.requests, eng.log_generation
    eng.log_generation = fake_log
    e = eng.FinanceRAGEngineV2.__new__(eng.FinanceRAGEngineV2)   # 绕开 init（不开 chroma）
    try:
        # a) 正常：思考 + 正文
        eng.requests = _FakeRequests(_sse(["思考一下。"], ["正文回答：营收 922.78 亿元。"], "stop",
                                         {"completion_tokens": 30}))
        ans = e.generate_with_modelscope("测试问题", "【上下文】", "sk-fake")
        check("S2a 正常流：正文原样返回", "正文回答" in ans, ans[:60])
        check("S2a 正常流：log error=None", logged.get("error") is None, str(logged.get("error")))
        check("S2a 正常流：log 记到 completion_tokens", logged.get("completion_tokens") == 30)

        # b) 预算被推理吃光：只有 reasoning + finish_reason=length
        eng.requests = _FakeRequests(_sse(["想" * 500], [], "length"))
        ans = e.generate_with_modelscope("测试问题", "【上下文在兜底里要原样给出】", "sk-fake")
        check("S2b 预算耗尽：说「花在推理上」而不是 API Key",
              "推理" in ans and "API Key" not in ans, ans[:80])
        check("S2b 预算耗尽：引用真实预算与思考字数",
              str(eng.LLM_MAX_TOKENS) in ans and "500" in ans, ans[:120])
        check("S2b 预算耗尽：context 兜底附后", "【上下文在兜底里要原样给出】" in ans)
        check("S2b 预算耗尽：log error=reasoning_exhausted_budget",
              logged.get("error") == "reasoning_exhausted_budget", str(logged.get("error")))
        # payload 必须带预算字段（不许静默漏给服务端默认值）
        check("S2b payload 带 max_tokens=LLM_MAX_TOKENS",
              eng.requests.bodies[0].get("max_tokens") == eng.LLM_MAX_TOKENS)

        # c) 真空：无内容无 finish —— 才允许指回 API Key
        eng.requests = _FakeRequests(_sse([], [], None))
        ans = e.generate_with_modelscope("测试问题", "【上下文】", "sk-fake")
        check("S2c 真空：报 API Key 提示", "API Key" in ans, ans[:60])
        check("S2c 真空：log error=empty_answer",
              logged.get("error") == "empty_answer", str(logged.get("error")))
    finally:
        eng.requests, eng.log_generation = orig_requests, orig_log


def v1_gate():
    """v1（评测基线引擎）同病灶同修：至少行为闸 b 必须诚实。"""
    sys.path.insert(0, str(APP / "src"))
    import rag_engine as eng1
    orig_requests = eng1.requests
    e = eng1.FinanceRAGEngine.__new__(eng1.FinanceRAGEngine)
    try:
        eng1.requests = _FakeRequests(_sse(["想" * 300], [], "length"))
        ans = e.generate_with_modelscope("q", "CTX-兜底", "sk-fake")
        check("S2[v1] 预算耗尽：诚实提示且附资料",
              "推理" in ans and "API Key" not in ans and "CTX-兜底" in ans, ans[:80])
        eng1.requests = _FakeRequests(_sse([], ["正文v1"], "stop"))
        ans = e.generate_with_modelscope("q", "CTX", "sk-fake")
        check("S2[v1] 正常流：正文返回", "正文v1" in ans, ans[:60])
    finally:
        eng1.requests = orig_requests


def main():
    try:
        static_gates()
        behavioral_gates()
        v1_gate()
    except Exception:
        traceback.print_exc()
        FAILS.append("守卫自身炸了（异常）")
    print(f"\ncheck_gen_budget: {len(FAILS)} 条红" + ("".join(f"\n  - {f}" for f in FAILS) or "，全绿"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
