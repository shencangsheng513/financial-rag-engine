# -*- coding: utf-8 -*-
"""Context Recall 评测跑批器（集合运算版，不依赖裁判模型）。

设计（2026-09-11，蒸馏自 Agentic RAG 评测实践）：
- **Context Recall 用集合运算，不用模型判**：评测用例直接声明期望来源的
  (company, type) 二元组，检索结果里核对命中 —— 参照精确到条目，不靠裁判
  模型打分，零 LLM 成本、可重复、无随机性。
- A_公司档案 / B_财务明细 / C_研报内容 三类计入**总召回率**（逐用例
  recall = 命中期望数 / 期望总数，再对用例求平均）。
- D_指标排行 是聚合型问题，走 structured_rank 结构化路径，词法召回不适用，
  单独记录：是否走了结构化路径（id 前缀 agg:）、返回了哪些公司。
- H_反向检查：expect_none 里的公司**绝不允许**出现在 top-k —— 这是精确率
  的护栏，历史上"长江三角洲→长江电力""民生问题→民生银行"都栽过。

用法（在 rag-finance-qa/ 下，用 D:/ML_Projects/.venv 的解释器）：
    python _rag_eval/run_eval.py                       # 升级后（默认 on）
    python _rag_eval/run_eval.py --baseline            # 升级前基线（upgrades off）
    python _rag_eval/run_eval.py --engine v1           # 生产旧引擎对照
    python _rag_eval/run_eval.py --hyde --tag hyde-on  # HyDE 查询改写臂（需 MODELSCOPE_API_KEY）
    python _rag_eval/run_eval.py --top-k 10            # 覆盖 top_k

结果写 output/eval_report_<tag>.json，同时打印逐用例明细。
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _load_engine(which):
    if which == "v1":
        from rag_engine import FinanceRAGEngine
        return FinanceRAGEngine()
    from rag_engine_v2 import FinanceRAGEngineV2
    return FinanceRAGEngineV2()


def _doc_key(doc):
    """文档 -> 可核对的来源键集合：{(company, type)}；行业研报另带 (sector, type)。"""
    meta = doc.get("metadata") or {}
    keys = set()
    if meta.get("company"):
        keys.add((meta["company"], meta.get("type") or "?"))
    if meta.get("sector"):
        keys.add((f"sector:{meta['sector']}", meta.get("type") or "?"))
    return keys


def _match(expect, doc):
    """单个期望是否被该文档满足。company/sector + type 精确相等即算命中。"""
    keys = _doc_key(doc)
    ent = expect.get("company") or (f"sector:{expect['sector']}" if expect.get("sector") else None)
    return (ent, expect.get("type")) in keys


def run_case(engine, case, top_k):
    docs = engine.retrieve(case["query"], top_k=top_k)
    companies = [((d.get("metadata") or {}).get("company") or "") for d in docs]
    used_structured = any(str(d.get("id", "")).startswith("agg:") for d in docs)

    result = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "top_companies": companies,
        "used_structured": used_structured,
        "returned": [
            {"id": d.get("id"), "company": c, "type": (d.get("metadata") or {}).get("type"),
             "agg_rank": d.get("agg_rank")}
            for d, c in zip(docs, companies)
        ],
    }

    expects = case.get("expect_any") or []
    if expects:
        hits = [e for e in expects if any(_match(e, d) for d in docs)]
        result["recall"] = len(hits) / len(expects)
        result["missed"] = [e for e in expects if e not in hits]

    nones = case.get("expect_none") or []
    if nones:
        result["violations"] = [c for c in nones if c in companies]

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v1", "v2"], default="v2")
    ap.add_argument("--baseline", action="store_true",
                    help="RAG_V2_UPGRADES=off 跑基线（仅对 v2 有意义）")
    ap.add_argument("--rrf", action="store_true",
                    help="RAG_V2_RRF=on 走 RRF 对等融合（默认关键词优先/向量兜底）")
    ap.add_argument("--hyde", action="store_true",
                    help="RAG_V2_HYDE=on 开 HyDE 查询改写（只改向量腿的查询文本）。"
                         "需要 MODELSCOPE_API_KEY；每个用例前置一次 LLM 调用，跑批会明显变慢")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--db", type=str, default=None,
                    help="覆盖 ChromaDB 库路径（换 embedding 后做 A/B：同一份代码、换库跑，"
                         "把「只改 embedding」这一个变量隔离出来）")
    ap.add_argument("--tag", type=str, default=None, help="报告文件名标签，默认按 baseline/engine 推断")
    args = ap.parse_args()

    if args.baseline:
        # config 在 import 时读环境变量，必须赶在引擎加载之前设好
        os.environ["RAG_V2_UPGRADES"] = "off"

    if args.rrf:
        # RRF 融合开关：向量侧永远进池（不再是关键词 0 命中才兜底），
        # 必须配 rerank 规则 6f 反字根混淆护栏。赶在引擎加载前设好。
        os.environ["RAG_V2_RRF"] = "on"

    if args.hyde:
        # HyDE 开关在 hybrid_search 里按调用时读（不像 config 是 import 时绑死），
        # 但仍赶在引擎加载前设好，避免"半个进程内开着"这种难复现的状态。
        os.environ["RAG_V2_HYDE"] = "on"

    cases_doc = json.loads((ROOT / "_rag_eval" / "eval_cases.json").read_text(encoding="utf-8"))
    top_k = args.top_k or cases_doc.get("top_k", 5)

    # 换库 A/B：必须先 import 引擎模块、再改模块级全局量、最后才实例化。
    # 引擎是 `from config import CHROMA_DB_PATH`，值绑在它自己的命名空间里；
    # 改 config 模块对已 import 的引擎无效。（先 import config 会 ModuleNotFoundError）
    if args.db:
        for mod_name in ("rag_engine_v2", "rag_engine"):
            try:
                mod = __import__(mod_name)
            except Exception:
                continue
            if hasattr(mod, "CHROMA_DB_PATH"):
                mod.CHROMA_DB_PATH = args.db
        print(f"[db] 覆盖库路径 -> {args.db}")

    t0 = time.time()
    engine = _load_engine(args.engine)
    print(f"[engine] {args.engine} 加载 {time.time() - t0:.1f}s，top_k={top_k}，"
          f"upgrades={'off' if args.baseline else 'on'}，"
          f"rrf={'on' if args.rrf else 'off'}，"
          f"hyde={'on' if args.hyde else 'off'}\n")

    # 期望公司必须在库里真实存在 —— 防用例笔误静默拉低召回（宁报错，不虚标）
    stats = engine.stats()
    in_corpus = set(stats.get("by_company", {}).keys())
    for case in cases_doc["cases"]:
        for e in (case.get("expect_any") or []) + [
                {"company": c} for c in (case.get("expect_none") or [])]:
            if e.get("company") and e["company"] not in in_corpus:
                print(f"[WARN] {case['id']}: 期望公司「{e['company']}」不在当前库中"
                      f"（共 {len(in_corpus)} 家），该用例结果不可信，请修用例或重新入库")

    results = [run_case(engine, case, top_k) for case in cases_doc["cases"]]

    # HyDE 分母自证：改写**实际生效**了几次。ok==0 时本轮与关闭 HyDE 的结果
    # 必然逐位相同，"召回没变化"必须读成**没验成**，不能读成"HyDE 没用"。
    hyde_stats = dict(getattr(getattr(engine, "retriever", None), "hyde_stats", None) or {})
    if args.hyde:
        n_ok = hyde_stats.get("ok", 0)
        print(f"\n[hyde] 计数 {hyde_stats}")
        print(f"[hyde] 分母自证：{len(results)} 个用例，改写生效 {n_ok} 次")
        if n_ok == 0:
            print("[hyde] WARN 生效 0 次 —— 本轮结果与 HyDE 关闭时完全一致是必然的，"
                  "结论是「没验成」而不是「HyDE 无效果」。先查 MODELSCOPE_API_KEY、"
                  "网络、以及 RAG_HYDE_MAX_TOKENS 是否被推理过程吃光")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print("=" * 72)
    headline = []
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        if cat == "D_指标排行":
            n_struct = sum(1 for r in rs if r["used_structured"])
            print(f"\n[{cat}] 结构化路径命中 {n_struct}/{len(rs)}（词法召回不适用，不计入总分）")
            for r in rs:
                print(f"  {r['id']} 「{r['query']}」 structured={r['used_structured']} "
                      f"-> {r['top_companies'][:3] or '(空)'}")
            continue
        if cat in ("H_反向检查", "I_库外问题"):
            print(f"\n[{cat}] 反向检查（只看违规，无召回指标）")
            for r in rs:
                bad = r.get("violations") or []
                flag = "VIOLATION" if bad else "OK"
                print(f"  {r['id']} [{flag}] 「{r['query']}」 返回: "
                      f"{r['top_companies'] or '(空)'}")
            continue
        for r in rs:
            # I 类负例（expect_any 为空）没有 recall 字段，不能直接 :.2f 格式化
            rec = r.get("recall")
            if not isinstance(rec, (int, float)):
                continue
            headline.append(rec)   # 总召回率只统计有 recall 的正例类
            miss = r.get("missed") or []
            print(f"  {r['id']} recall={rec:.2f} "
                  f"{'缺:' + str(miss) if miss else ''}「{r['query']}」")
        # 平均也只统计有 recall 的用例，别把 n/a 当 0 拉低均值
        with_recall = [r["recall"] for r in rs if isinstance(r.get("recall"), (int, float))]
        cat_avg = sum(with_recall) / len(with_recall) if with_recall else None
        print(f"  -- [{cat}] 平均 {cat_avg:.2f}" if cat_avg is not None
              else f"  -- [{cat}] 无召回指标（负例类，只看违规数）")

    overall = sum(headline) / len(headline) if headline else 0.0
    violations = sum(1 for c in ("H_反向检查", "I_库外问题") for r in by_cat.get(c, []) if r.get("violations"))
    print("\n" + "=" * 72)
    print(f"总召回率（A/B/C，{len(headline)} 用例）：{overall:.2%}")
    print(f"负例类（H+I）违规数：{violations}")

    tag = args.tag or ("baseline" if args.baseline else ("v1" if args.engine == "v1" else "upgraded"))
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_report_{tag}.json"
    out_path.write_text(json.dumps({
        "tag": tag, "engine": args.engine, "top_k": top_k,
        "upgrades": "off" if args.baseline else "on",
        "rrf": "on" if args.rrf else "off",
        "hyde": "on" if args.hyde else "off",
        # 分母自证：hyde=on 但 ok=0 时，这份报告的指标等同于 HyDE 关闭，
        # 不能拿它当"HyDE 无效"的证据。
        "hyde_stats": hyde_stats or None,
        "overall_recall_abc": round(overall, 4),
        "h_violations": violations,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
