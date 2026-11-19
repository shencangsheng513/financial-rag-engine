# -*- coding: utf-8 -*-
"""
拒答阈值标定（2026-10-15 加）

蒸馏文档 04 §2③ 主张双重拒答：硬阈值 + LLM 裁判，硬阈值用 0.15。
**那个 0.15 不能直接抄** —— 它是原文 rerank 分量纲下的值，本项目 config.py 明写
"各段量纲显式化，禁止跨量纲比较"，rerank_score 是启发式复合分（0~1.x，非校准概率）。

标定方法（与 jev-laya-eval/sweep_threshold.py 同一套方法论）：
    正例 = eval_cases.json 里 expect_any 非空的用例（库里确实有答案）
    负例 = H 类反向检查（库里没有答案 / 故意诱导）
    扫 rerank_score 门槛，看"正例不被误杀 + 负例被抓到"的分离点。

用法：python calibrate_refusal.py [--top-k 5]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    os.environ.setdefault("RAG_V2_UPGRADES", "on")
    from rag_engine_v2 import FinanceRAGEngineV2

    cases = json.load(open(os.path.join(HERE, "eval_cases.json"), encoding="utf-8"))["cases"]
    engine = FinanceRAGEngineV2()

    pos, neg = [], []
    for c in cases:
        expects = c.get("expect_any") or []
        nones = c.get("expect_none") or []
        kind = "neg" if (not expects and nones) else ("pos" if expects else "skip")
        if kind == "skip":
            continue
        try:
            docs = engine.retrieve(c["query"], top_k=args.top_k)
        except Exception as e:
            print(f"[warn] {c['id']} 检索失败: {e}")
            continue
        if not docs:
            (neg if kind == "neg" else pos).append({"id": c["id"], "q": c["query"], "top1": None})
            continue
        # 取 best rerank_score（None 当 0 处理，但要单独计数）
        scores = [d.get("rerank_score") for d in docs if d.get("rerank_score") is not None]
        top1 = max(scores) if scores else None
        (neg if kind == "neg" else pos).append(
            {"id": c["id"], "q": c["query"], "top1": top1, "n": len(docs)})

    print(f"正例 {len(pos)} 条 ｜ 负例 {len(neg)} 条 ｜ top_k={args.top_k}")
    print()
    pv = sorted([x["top1"] for x in pos if x["top1"] is not None], reverse=True)
    nv = sorted([x["top1"] for x in neg if x["top1"] is not None], reverse=True)
    print("正例 top1 rerank 降序：", [round(v, 3) for v in pv])
    print("负例 top1 rerank 降序：", [round(v, 3) for v in nv])
    print()

    if not pv:
        print("正例无有效分数，无法标定")
        return 1

    # 扫门槛：正例被误杀率 + 负例被抓率
    print(f"{'门槛':>6} {'正例误杀':>10} {'负例被抓':>10} {'综合':>8}")
    print("-" * 44)
    best = None
    for t in [round(x * 0.05, 2) for x in range(0, 41)]:
        fp = sum(1 for v in pv if v < t)                    # 正例误杀（有答案却拒答）
        tp = sum(1 for v in nv if v < t)                    # 负例被抓（没答案且拒答）
        # 综合分：抓负例的收益要大于误杀正例的代价（正例是主线，权重更高）
        score = tp * 1.0 - fp * 3.0
        if best is None or score > best[1]:
            best = (t, score, fp, tp)
        print(f"{t:>6.2f} {fp:>10} {tp:>10} {score:>8.1f}")

    print("-" * 44)
    t, score, fp, tp = best
    print(f"推荐门槛 ≈ {t:.2f}（正例误杀 {fp} / 负例 {len(nv)} 条中抓到 {tp}）")
    print()
    print("读法：门槛越高，越多正例被误杀（有答案却拒答）—— 那是可用性损失；")
    print("     门槛越低，越多负例漏过（没答案却硬答）—— 那是幻觉。")
    print("     综合分按 负例收益 x1 - 正例误杀 x3 算：**拒答错一个真问题的代价，")
    print("     约等于漏过三个假问题。** 这个 3:1 是拍的，你可以按业务改。")
    print()
    print("注意：本标定只覆盖硬阈值层。LLM 裁判层需要可用的 API Key 才能标定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
