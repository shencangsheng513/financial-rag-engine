# -*- coding: utf-8 -*-
"""Reranker 数值可答性规则（6g）的确定性守卫。

钉住的回归（2026-09-22 Jev 巡检发现）：
「板凳龙问题三的最小螺距pmin是多少厘米？」的真实答案在**摘要块**（1485 字，
吃 >1000 字长度惩罚 ×0.9），向量分还略低于四篇"问题分析"诱饵块 —— 纯向量+
长度排序下答案块跌破 top-5，模型只能说"资料未给出具体数值"。
规则 6g：query 问数值 且 文本含"数字+单位/等号"形态的答案时小幅加分。

断言设计：
1. 数值问 + 诱饵向量分更高 → 修复后含答案块必须赢（6g 生效）。
2. 非数值问 + 相同两块 → 排序不得翻转（6g 不是无差别偏置）。
3. 数值问但文本没有任何数字 → 不加分（护栏不靠猜）。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from rag_engine_v2 import Reranker  # noqa: E402
except ModuleNotFoundError as _exc:  # mt 环境无 chromadb；本守卫须在 .venv 下跑
    raise unittest.SkipTest(f"需 chromadb 环境（用 D:/ML_Projects/.venv 解释器跑）: {_exc}")

DECOY = {
    "id": "decoy-analysis",
    "text": "【板凳龙建模·2.4 问题三分析】问题三将螺距 p 视为决策变量，要求龙头在各节"
            "板凳抬升过程中互不碰撞。本节建立空间坐标系与刚体运动学方程，给出约束推导。" * 3,
    "metadata": {"type": "research_report", "company": "", "title": "板凳龙"},
    "vector_score": 0.7542,
}
# 生产真实形态：Popo 排版在数字/标识符内也插空格（"44 . 7656"、"p min"），
# 查询侧的连续 token（pmin）在块内配不上 → 加分项全空，只剩向量分还挨长度惩罚。
ANSWER = {
    "id": "abstract-with-number",
    "text": "【板凳龙建模·摘要】针对 板凳龙 闹元宵问题，建立运动学与优化模型。利用 "
            "Lingo 求解非线性规划，得到问题三的最小螺距 p min = 44 . 7656 厘 米，"
            "此时相邻板凳互不干涉。" * 12,  # >1000 字，复刻真实块的长度惩罚场景
    "metadata": {"type": "research_report", "company": "", "title": "板凳龙"},
    "vector_score": 0.7209,
}


def _top_id(reranker, query, docs):
    return reranker.rerank(query, [dict(d) for d in docs], top_k=2)[0]["id"]


class TestNumericAnswerability(unittest.TestCase):
    def setUp(self):
        self.r = Reranker()

    def test_numeric_query_prefers_chunk_containing_value(self):
        got = _top_id(self.r, "板凳龙问题三的最小螺距pmin是多少厘米？", [DECOY, ANSWER])
        self.assertEqual(got, ANSWER["id"], "数值问必须把含答案数字的块顶到第一")

    def test_non_numeric_query_order_unchanged(self):
        got = _top_id(self.r, "板凳龙问题三是如何建模的？", [DECOY, ANSWER])
        self.assertEqual(got, DECOY["id"], "非数值问不得因 6g 翻转排序")

    def test_numeric_query_no_number_in_any_chunk(self):
        plain = dict(DECOY)
        plain["text"] = "本节只讲符号约定与术语定义，不含任何结论。" * 8
        got = _top_id(self.r, "板凳龙问题三的最小螺距是多少厘米？", [plain, DECOY])
        self.assertEqual(got, DECOY["id"], "两侧都无数字时 6g 不应改变任何东西")


class TestBuildContextBudget(unittest.TestCase):
    """超长块只能节选、不得整块丢弃（巡检回归的第二半）。"""

    def test_partial_fit_keeps_answer_head(self):
        from config import MAX_CONTEXT_LENGTH
        from rag_engine_v2 import FinanceRAGEngineV2
        big = dict(DECOY)
        big["text"] = "本节推导约束并给出符号约定。" * 70  # ~910 字，整块放得下
        ctx = FinanceRAGEngineV2.build_context(None, [big, dict(ANSWER)])
        self.assertIn("44 . 7656", ctx, "预算只容得下答案块开头时，必须节选保留而非整块丢弃")
        self.assertIn("节选", ctx)


if __name__ == "__main__":
    unittest.main()
