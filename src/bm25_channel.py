# -*- coding: utf-8 -*-
"""BM25 关键词通道：jieba 分词 + 内存 BM25 索引（2026-09-11）。

为什么需要它 —— rag_engine_v2.HybridRetriever._extract_keywords 的已知短板：
没有实体命中时，原关键词通道对纯中文问句**一个 token 都抽不出来**
（原实现只退英文数字串），`$contains` 拿整句去匹配必然 0 命中，
关键词通道对这类 query 等于全程关闭。BM25 在**全库分词索引**上打分，
不依赖实体表，专补这类问法（如「锂电池行业怎么样」「这家公司负债高吗」）。

设计约束（来自 2026-09-10 实测教训，别改回去）：
- 向量检索只作兜底：embedding 是英文模型 all-MiniLM-L6-v2，对中文区分度差，
  若让它参与 RRF 加权融合，向量第 1 名会压过关键词第 1 名（实测 SH600003
  之类无关文档挤进 TOP3）。BM25 只与**同族词法信号**（实体关键词通道）融合。
- BM25 分数无上界，与 0~1 的向量相似度量纲不同 → 融合只看名次（RRF），
  不直接比较分数。
- 当前库 664 块，全量内存索引开销可忽略；通过 doc 数量做版本号，库变化后惰性重建。

本模块**不 import config**（依赖方向：引擎层把参数传进来），保持可独立测试。
"""

import math
import re
from collections import Counter

try:
    import jieba
    jieba.setLogLevel(60)  # 关掉「Building prefix dict」刷屏
    _HAS_JIEBA = True
except ImportError:  # jieba 缺失时通道静默关闭，引擎退回原行为（不炸生产）
    _HAS_JIEBA = False

# 与 _norm 同口径：全角→半角、去空白、小写。中文字符不受影响。
_FW2HW = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}


def _norm(s: str) -> str:
    s = "".join(_FW2HW.get(ch, ch) for ch in (s or ""))
    return re.sub(r"\s+", "", s).lower()


# 中文场景：词长 >=2 才保留（单字几乎全是噪声）；英文数字串 >=2 保留。
# 这是词法通道，宁缺毋滥 —— 单字 token 会把「市值」拆成「市」「值」级别的垃圾召回。
def tokenize(text: str, stop_words=None) -> list:
    if not _HAS_JIEBA:
        return []
    text = _norm(text)
    toks = []
    for t in jieba.lcut(text):
        if len(t) < 2:
            continue
        if stop_words and t in stop_words:
            continue
        toks.append(t)
    # jieba 可能保留 GTE3000 这种连写，补一刀字母数字切分
    for seg in re.findall(r"[a-z0-9]{2,}", text):
        toks.append(seg)
    return toks


class BM25Index:
    """全库内存 BM25。ensure() 惰性构建，search() 返回 [(doc_id, score, rank)]。"""

    def __init__(self):
        self._records = []   # [{id, text, metadata, tokens}]
        self._idf = {}
        self._avgdl = 0.0
        self._version = None  # 构建时的 collection.count()

    def ensure(self, collection, stop_words=None, k1=1.5, b=0.75):
        count = collection.count()
        if self._version == count and self._records:
            return
        got = collection.get(include=["documents", "metadatas"])
        records, df = [], Counter()
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            toks = tokenize(text, stop_words)
            records.append({"id": doc_id, "text": text,
                            "metadata": meta or {}, "tokens": toks})
            for t in set(toks):
                df[t] += 1
        n = max(len(records), 1)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avgdl = sum(len(r["tokens"]) for r in records) / n
        self._records = records
        self._version = count
        self._k1, self._b = k1, b

    def search(self, query: str, top_n: int = 20):
        """返回 [(record, score)]，按 BM25 降序，最多 top_n 条。"""
        if not _HAS_JIEBA or not self._records:
            return []
        q_toks = tokenize(query)
        if not q_toks:
            return []
        k1, b = self._k1, self._b
        scored = []
        for r in self._records:
            tf = Counter(r["tokens"])
            dl = len(r["tokens"])
            if dl == 0:
                continue
            s = 0.0
            for t in q_toks:
                if t not in tf:
                    continue
                s += self._idf.get(t, 0.0) * tf[t] * (k1 + 1) / (
                    tf[t] + k1 * (1 - b + b * dl / self._avgdl))
            if s > 0:
                scored.append((r, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
