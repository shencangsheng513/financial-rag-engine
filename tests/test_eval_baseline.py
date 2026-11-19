# -*- coding: utf-8 -*-
"""评测基线门禁 —— REVIEW.md §5 第 3 条的落地。

为什么需要
----------
`_rag_eval/run_eval.py` 能跑出 `overall_recall_abc` 与 `h_violations`，
但"低于基线就拦下"一直是人肉动作。检索改动最容易的失败模式不是报错，
而是**静默变差**：服务能起、守卫全绿、召回率掉了几个点，没人发现。

设计取舍
--------
- **不重跑 eval**：重跑要 ChromaDB + embedding 模型，不适合塞进每次 unittest。
  本门禁只读已落盘的 `output/eval_report_*.json`。
- **两级判定**（沿用 `test_static_guards` 的做法）：
  * 通用红线：任何报告 `overall_recall_abc < floor_recall` → FAIL
  * 回归比对：tag 已登记在快照里 → recall 不得低于快照（容差内）、
    `h_violations` 不得增加
  * tag 未登记（新实验标签）→ **只提示不拦**，避免 WIP 实验把门禁刷红
- **改基线是显式动作**：快照在 `_rag_eval/eval_baseline.json`。
  降基线必须写理由 —— 否则等于拆门禁。

两个已知盲区（别误以为这门禁能防住一切）
--------------------------------------
1. 报告是手动跑出来的。**改了检索代码但没重跑 eval，门禁仍然绿。**
   `test_report_covers_current_code` 只提示不拦 —— 拦了会变成"改一行代码
   就得重跑全量评测"，那样门禁会被人为绕过，比没有更糟。
2. 只校验已登记 tag 的数值，**不校验用例集本身是否被改弱**。
   `_rag_eval/eval_cases.json` 的改动需要人工 review。
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
SNAP = ROOT / "_rag_eval" / "eval_baseline.json"

# 检索核心文件：这些比报告新，说明报告可能已过期（只提示）
CORE = ("src/rag_engine_v2.py", "src/rag_engine.py", "config.py",
        "_rag_eval/run_eval.py", "_rag_eval/eval_cases.json")


def _snapshot():
    return json.loads(SNAP.read_text(encoding="utf-8"))


def _reports():
    """{tag: 报告 dict}；无法解析的按文件名登记，带 _error。"""
    out = {}
    for p in sorted(OUT.glob("eval_report_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("顶层不是对象")
            d["_file"] = p.name
            d["_mtime"] = p.stat().st_mtime
        except Exception as e:
            d = {"_file": p.name, "_error": str(e)}
        tag = d.get("tag") or p.stem[len("eval_report_"):]
        out[tag] = d
    return out


class TestEvalBaseline(unittest.TestCase):

    def test_snapshot_is_readable(self):
        """快照本身要能被解析 —— 否则后面的比对全是假绿。"""
        snap = _snapshot()
        self.assertIn("tags", snap, "eval_baseline.json 缺 tags")
        self.assertIn("floor_recall", snap, "eval_baseline.json 缺 floor_recall")
        self.assertTrue(snap["tags"], "基线快照里没有任何 tag")

    def test_at_least_one_report_exists(self):
        reps = _reports()
        self.assertTrue(
            reps,
            f"{OUT} 下没有 eval_report_*.json。门禁需要至少一份报告，"
            f"跑 `python _rag_eval/run_eval.py` 生成。")
        for tag, rep in reps.items():
            self.assertNotIn("_error", rep, f"{rep.get('_file')} 解析失败: {rep.get('_error')}")

    def test_recall_not_below_baseline(self):
        """核心断言：召回率不得低于快照，且不得跌破绝对底线。"""
        snap = _snapshot()
        tol = snap.get("tolerance", 0.01)
        floor = snap.get("floor_recall", 0.80)
        regressed, below_floor, unregistered = [], [], []

        for tag, rep in _reports().items():
            r = rep.get("overall_recall_abc")
            if r is None:
                continue
            if r < floor:
                below_floor.append(f"{tag}: {r:.2%} < 底线 {floor:.0%}（{rep['_file']}）")
            base = (snap["tags"].get(tag) or {}).get("overall_recall_abc")
            if base is None:
                unregistered.append(f"{tag}: recall={r:.2%}（未登记在快照里，本次不拦）")
            elif r < base - tol:
                regressed.append(
                    f"{tag}: {r:.2%} < 基线 {base:.2%}（容差 {tol:.0%}）"
                    f" → 报告 {rep['_file']}。若是有意接受的回退，需显式改 eval_baseline.json 并说明理由")

        if unregistered:
            print("\n[提示] 未登记的 tag（新实验，未纳入回归比对）：")
            for u in unregistered:
                print("  -", u)

        self.assertEqual([], below_floor,
                         "召回率跌破绝对底线：\n  " + "\n  ".join(below_floor))
        self.assertEqual([], regressed,
                         "召回率相对基线回退：\n  " + "\n  ".join(regressed))

    def test_h_violations_not_increased(self):
        """H 类反向检查违规数不得比基线多 —— 这条历史上真出过事。"""
        snap = _snapshot()
        worse, unregistered = [], []
        for tag, rep in _reports().items():
            v = rep.get("h_violations")
            if v is None:
                continue
            base_entry = snap["tags"].get(tag) or {}
            base = base_entry.get("h_violations")
            if base is None:
                unregistered.append(f"{tag}: h_violations={v}（未登记，本次不拦）")
            elif v > base:
                worse.append(
                    f"{tag}: {v} 处违规 > 基线 {base} 处 → 报告 {rep['_file']}。"
                    f"反向检查违规是精确率护栏，出现新的必须查（历史上"
                    f"「长江三角洲→长江电力」「民生问题→民生银行」都栽过）")
        if unregistered:
            print("\n[提示] h_violations 未登记 tag：", "; ".join(unregistered))
        self.assertEqual([], worse, "反向检查违规增加：\n  " + "\n  ".join(worse))

    def test_report_covers_current_code(self):
        """软检查：报告是否早于检索核心文件的改动。只提示，不 fail。"""
        snap = _snapshot()
        tol = snap.get("tolerance", 0.01)
        newest_report = max((r.get("_mtime", 0) for r in _reports().values()), default=0)
        if not newest_report:
            self.skipTest("没有可用的报告 mtime")
        stale = []
        for rel in CORE:
            p = ROOT / rel
            if p.exists() and p.stat().st_mtime > newest_report:
                stale.append(rel)
        if stale:
            print("\n[提示] 以下文件比最新 eval 报告更新，报告可能已过期"
                  "（这不是失败，但结论要打折）：")
            for s in stale:
                print("  -", s)
            print("  重跑：cd rag-finance-qa && python _rag_eval/run_eval.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
