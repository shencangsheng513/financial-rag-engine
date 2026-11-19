# -*- coding: utf-8 -*-
"""ingest_popo.build_docs 的确定性守卫。

钉住的约定（都是本仓库踩过的坑，改坏了这里会红）：
1. 短块被显式拒收并上报，不静默丢（engine <50 字过滤的入库侧对应物）。
2. popo 块永不写 company/stock_code（行业研报先例：防 by_company 污染）。
3. id 稳定（内容 hash）→ 重跑幂等，不留影子文档。
4. 父子回填三件套：type=research_report + title + 连续 chunk 序号。
5. metadata 值只允许 str/int（chroma 标量约束，pages/block_ids 必须预先拼串）。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ingest_popo  # noqa: E402


def _chunk(hp, text, pages=(1,), level=2, block_ids=(3, 4)):
    return {"heading_path": hp, "text": text, "pages": list(pages),
            "level": level, "block_ids": list(block_ids), "type": "text"}


LONG = "这是一段足够长的正文内容，" * 10  # 140 字


class TestIngestPopoBuild(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            _chunk("论文A > 摘要", LONG, pages=(1,)),
            _chunk("论文A > 目录", "短到不行的块", pages=(2,)),          # <50 → 拒收
            _chunk("论文A > 一、正文 > 1.1", LONG, pages=(4, 5), level=3),
        ]
        self.ids, self.docs, self.metas, self.dropped = ingest_popo.build_docs(
            self.chunks, "grok")

    def test_short_chunk_rejected_not_silent(self):
        self.assertEqual(len(self.dropped), 1)
        self.assertIn("目录", self.dropped[0][0])
        self.assertEqual(len(self.ids), 2)

    def test_no_company_keys_anywhere(self):
        for m in self.metas:
            self.assertNotIn("company", m)
            self.assertNotIn("stock_code", m)

    def test_ids_stable_and_prefixed(self):
        again = ingest_popo.build_docs(self.chunks, "grok")[0]
        self.assertEqual(self.ids, again)
        for i in self.ids:
            self.assertTrue(i.startswith("popo_grok_"))
            self.assertEqual(len(i), len("popo_grok_") + 12)

    def test_parent_expand_triplet(self):
        for m in self.metas:
            self.assertEqual(m["type"], "research_report")
            self.assertEqual(m["title"], "论文A")
        chunks = [m["chunk"] for m in self.metas]
        self.assertEqual(chunks, sorted(chunks))
        self.assertEqual(set(chunks), {0, 1})

    def test_metadata_scalars_only(self):
        for m in self.metas:
            for k, v in m.items():
                self.assertIsInstance(v, (str, int), f"{k} 不是标量: {v!r}")

    def test_pages_and_blockids_joined(self):
        last = self.metas[-1]
        self.assertEqual(last["pages"], "4,5")
        self.assertEqual(last["block_ids"], "3,4")
        self.assertEqual(last["level"], 3)

    def test_doc_head_prefix_in_document(self):
        self.assertTrue(self.docs[0].startswith("【论文A·摘要】(p.1)"))

    def test_min_chars_floor_matches_engine_filter(self):
        # engine（rag_engine_v2）按 text_len<50 丢块；入库侧下限必须同值，
        # 谁把 50 改小，这里立刻红。
        self.assertGreaterEqual(ingest_popo.MIN_CHARS, 50)


if __name__ == "__main__":
    unittest.main()
