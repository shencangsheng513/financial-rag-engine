# -*- coding: utf-8 -*-
"""引用可核性守卫 `src/cite_guard.py` 的确定性守卫 —— 零网络，rc 0/1。

为什么要有这一发（病灶）：`build_context()` 把窗口拼成 `[1] [2] …`，prompt 要求答案标角标，
但线上**没有任何一处**检查"标了没标对"或"答案里的标识符真在那段代码里"。幻觉符号与悬空角标
一路畅通，而且带角标的幻觉比真答案更像有依据。这一族在评测侧已经钉过（`cov==0` 地板闸），
本发钉的是它的线上版：**"没核到"不许被读成"核过了"**。

风格照 `check_gen_budget.py`：纯文本 + 纯函数，`check()` 打点，末行汇总，红则 rc 1。
刻意**不 import** `rag_engine_v2`：那个文件此刻在别的会话手里改，而且 import 它会拉起
chroma 与网络侧依赖——一发守卫不该依赖它要保护的东西活着。

跑：cd rag-finance-qa && PYTHONIOENCODING=utf-8 python tests/check_cited_symbols.py
"""
import ast
import sys
import traceback
from pathlib import Path

APP = Path(__file__).resolve().parent.parent          # rag-finance-qa/
sys.path.insert(0, str(APP / "src"))

import cite_guard as CG                                # noqa: E402

FAILS = []

W1 = "def __init__(self, loader):\n        self.batch_sampler = DynamicBatchSampler(loader)"
W2 = "敏感词过滤在 sensitive_word_filter 里做，见 utils/infer.py:318 的调用。"
W3 = "这一窗口讲 f5_tts.models.utils 的重采样。"
CTX = [W1, W2, W3]

# 文件出口：只放**不会和 `dict.get()` 撞名**的那些（第一版收了 "get"，自己假红过）。
# 网络侧不必在这儿问：上面那一格已经要求 import 名单恰为 {re}，requests/socket 压根进不来。
IO_ATTRS = {"open", "read_text", "read_bytes", "write_text", "write", "writelines",
            "urlopen", "Popen", "connect"}


def check(name, cond, detail=""):
    tag = "OK  " if cond else "[FAIL] "
    print(f"{tag}{name}" + (f"  —— {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---------- S1 提取：什么算"代码形状的符号" ----------

def extraction_gates():
    pos = [("下划线 sensitive_word 命中", "sensitive_word", ["sensitive_word"]),
           ("点号 f5_tts.models.utils 命中", "f5_tts.models.utils", ["f5_tts.models.utils"]),
           ("驼峰两大写 DynamicBatchSampler 命中", "DynamicBatchSampler", ["DynamicBatchSampler"]),
           ("路径带行号 utils/infer.py:318 整体命中", "utils/infer.py:318", ["utils/infer.py:318"]),
           ("行号 foo_bar:12 命中", "foo_bar:12", ["foo_bar:12"]),
           ("中文夹英文 get_epss_timesteps 命中", "k5 的 get_epss_timesteps 在", ["get_epss_timesteps"]),
           ("括号 load_items() 只留标识符", "load_items()", ["load_items"])]
    for name, text, want in pos:
        got = CG.extract_symbols(text)
        check("S1 " + name, got == want, "实际 %s" % (got,))

    neg = [("散文里的 Python 不算", "Python 是一门语言"),
           ("单首大写 Config 不算", "改 Config 就行"),
           ("短碎片 a_b 不算", "a_b 这种命名"),
           ("裸扩展名 py 不算", "写在 py 文件里"),
           ("纯中文不算", "这套代码没有做说话人分离")]
    for name, text in neg:
        got = CG.extract_symbols(text)
        check("S1 " + name, got == [], "实际 %s" % (got,))

    # 三条正则会有覆盖；按包含关系丢短的，否则同一说法数两遍会把分母吹大。
    once = CG.extract_symbols("utils/infer.py:318 与 utils/infer.py 与 infer.py:318")
    check("S1 同一说法只数一次（包含去重，分母不吹大）", once == ["utils/infer.py:318"],
          "实际 %s" % (once,))
    dup = CG.extract_symbols("DynamicBatchSampler 又一次 DynamicBatchSampler")
    check("S1 原文重复两遍也只数一次", dup == ["DynamicBatchSampler"], "实际 %s" % (dup,))


# ---------- S2 三态判定 ----------

def state_gates():
    r = CG.verify("用了 DynamicBatchSampler [1]，还有 sensitive_word_filter [2]。", CTX)
    check("S2 全部核到 -> PASS", r["state"] == CG.STATE_PASS, str(r))
    check("S2 PASS 的分母是现算的（窗口数/符号数/被引窗口数）",
          (r["n_windows"], r["n_symbols"], r["n_cited_windows"]) == (3, 2, 2), str(r))

    r = CG.verify("用了 NonexistThing [1]。", CTX)
    check("S2 幻觉符号 -> FLAG 且点名",
          r["state"] == CG.STATE_FLAG and r["missing"] == ["NonexistThing"], str(r))

    r = CG.verify("用了 DynamicBatchSampler [9]。", CTX)
    check("S2 悬空角标 -> FLAG 且点名角标",
          r["state"] == CG.STATE_FLAG and r["dangling"] == [9], str(r))

    r = CG.verify("它读 f5_tts.models.utils [1]。", CTX)
    check("S2 符号只在**未被引用**的窗口里 -> FLAG（引用面必须有牙齿）",
          r["state"] == CG.STATE_FLAG and r["missing"] == ["f5_tts.models.utils"], str(r))
    check("S2 上一条的被引窗口数确实只有 1（不是全池）", r["n_cited_windows"] == 1, str(r))

    r = CG.verify("这套代码没有做说话人分离。", CTX)
    check("S2 无可核符号 -> UNVERIFIED，不许折算成通过",
          r["state"] == CG.STATE_UNVERIFIED and r["n_symbols"] == 0, str(r))

    for shape, ctx in (("空列表", []), ("None", None)):
        r = CG.verify("用了 DynamicBatchSampler [1]。", ctx)
        check("S2 没有可引用窗口（%s）-> UNVERIFIED" % shape,
              r["state"] == CG.STATE_UNVERIFIED and r["n_windows"] == 0, str(r))

    # 两档：判据走全局（跨窗口拼起来的事实不算编造），就近档只报不判。
    r = CG.verify("构造 DynamicBatchSampler [1]。过滤在 sensitive_word_filter [2]。"
                  "这行 utils/infer.py:318 也算引过 [1]。", CTX)
    check("S2 跨窗口事实在全局档核到 -> PASS",
          r["state"] == CG.STATE_PASS and r["missing"] == [], str(r))
    check("S2 同一符号在就近档点名（差值只报不判）",
          r["near_missing"] == ["utils/infer.py:318"], str(r))

    r = CG.verify("顺带一提 sensitive_word_filter 也在这里。它见 [2]，流程见 DynamicBatchSampler [1]。", CTX)
    check("S2 符号所在句没角标 -> 点名 unattributed 但不判红",
          r["state"] == CG.STATE_PASS and r["unattributed"] == ["sensitive_word_filter"], str(r))

    # 生产侧 retrieved 是 dict；str 与 dict 必须同判，未知形状不许炸。
    # 答案**必须连第二个窗口一起引**：只引 [1] 的话 `_WINDOW_KEYS` 少认一个键也没人喊（变异会存活）。
    as_dict = [{"text": W1}, {"content": W2}, {"page_content": W3}]
    ans = "用了 DynamicBatchSampler [1]。过滤在 sensitive_word_filter [2]。"
    rd, rs = CG.verify(ans, as_dict), CG.verify(ans, CTX)
    check("S2 dict 形槽位与 str 形同判（生产 payload 形状，两个窗口都要认）",
          rd["state"] == rs["state"] == CG.STATE_PASS
          and rd["n_windows"] == rs["n_windows"] == 3
          and rd["n_cited_windows"] == 2 and rd["missing"] == [], str(rd))

    r = CG.verify("过滤在 Sensitive_Word_Filter [2]。", CTX)
    check("S2 大小写被模型改写仍算核到（比对两侧都走 lower）",
          r["state"] == CG.STATE_PASS and r["missing"] == [], str(r))
    r = CG.verify("用了 DynamicBatchSampler [1]。", [{"no_such_key": 1}, 42])
    check("S2 认不出的槽位形状回空文本而不是抛异常",
          r["state"] == CG.STATE_FLAG and r["missing"] == ["DynamicBatchSampler"], str(r))


# ---------- S3 三态互斥（判据的真值表，不是举例） ----------

FIXTURES = [
    ("全核到", "用了 DynamicBatchSampler [1]。", CTX),
    ("幻觉", "用了 NonexistThing [1]。", CTX),
    ("悬空", "用了 DynamicBatchSampler [7]。", CTX),
    ("无符号", "纯散文一句。", CTX),
    ("无窗口", "用了 DynamicBatchSampler [1]。", []),
    ("两个都有", "NonexistThing [7] 与 DynamicBatchSampler [1]。", CTX),
    ("未知形状", "用了 DynamicBatchSampler [1]。", [{"a": 1}]),
    ("既无符号又有悬空角标（钉住 FLAG 优先于 UNVERIFIED）", "参见 [9]。", CTX),
]


def mutex_gates():
    bad, seen = [], set()
    for name, ans, ctx in FIXTURES:
        r = CG.verify(ans, ctx)
        st = r["state"]
        seen.add(st)
        if st not in CG.STATES:
            bad.append("%s：state=%s 不在三态名单里" % (name, st))
            continue
        # 三态各自的**充要**条件（照 verify() 的判定顺序，含"谁先谁后"）：
        # 判据改了但注释没改，会在这儿露出来。
        want = (CG.STATE_UNVERIFIED if r["n_windows"] == 0
                else CG.STATE_FLAG if (r["dangling"] or r["missing"])
                else CG.STATE_UNVERIFIED if not r["symbols"]
                else CG.STATE_PASS)
        if st != want:
            bad.append("%s：state=%s 但按条件应为 %s" % (name, st, want))
    check("S3 三态互斥且穷尽（%d 份夹具逐份对判据）" % len(FIXTURES),
          not bad and seen == set(CG.STATES),
          "; ".join(bad) or "夹具只打出 %s" % sorted(seen))
    check("S3 UNVERIFIED 不参与 PASS 计数（名单里就没有第四态）",
          CG.STATES == ("PASS", "FLAG", "UNVERIFIED") and len(CG.STATES) == 3, str(CG.STATES))


# ---------- S4 静态闸：零出口 + 接线状态诚实 ----------

def static_gates():
    st = (APP / "src" / "cite_guard.py").read_text(encoding="utf-8")
    tree = ast.parse(st)
    # 用 AST 而不是 grep：`__doc__` 里 `D:\\ML_Projects` 会被 Python 反转义成 `D:\...`，
    # 拿它去 replace 原文永远不命中（我第一版就是这么假红的）。
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
    check("S4 cite_guard 的全部 import 只有 re（零第三方/零网络/零引擎，一条说完）",
          mods == {"re"}, "实际 %s" % sorted(mods))
    io_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "open":
            io_calls.append("%d:open" % node.lineno)
        elif isinstance(f, ast.Attribute) and f.attr in IO_ATTRS:
            io_calls.append("%d:%s" % (node.lineno, f.attr))
    check("S4 cite_guard 里没有文件/网络出口（只吃调用方给进来的文本）",
          not io_calls, "出现 %s" % sorted(set(io_calls)))
    # 接线状态：今天**故意**没有调用点。接上那天这一格会红，逼下一轮把断言翻过来
    # （改成"点名调用点"），而不是留下一个假装已经接上的模块 + 一份过期的"没接线"说明。
    callers = [p.name for p in sorted((APP / "src").glob("*.py"))
               if p.name != "cite_guard.py" and "cite_guard" in p.read_text(encoding="utf-8")]
    check("S4 生产侧调用点数 == 0（未接线是本发的**声明**，不是漏；接线那轮换掉这一格）",
          callers == [], "已出现调用点 %s -> 把本格改成点名调用点并同步工单" % callers)


def main():
    try:
        extraction_gates()
        state_gates()
        mutex_gates()
        static_gates()
    except Exception:
        traceback.print_exc()
        FAILS.append("守卫自身炸了（异常）")
    print(f"\ncheck_cited_symbols: {len(FAILS)} 条红" + ("".join(f"\n  - {f}" for f in FAILS) or "，全绿"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
