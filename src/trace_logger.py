# -*- coding: utf-8 -*-
"""检索全链路追踪（2026-09-12 加）

为什么要有这个：
    `_rag_eval/` 解决的是**离线评测**——跑一批固定 case，对比 baseline / upgraded。
    但它回答不了线上问题：用户刚才那次问答为什么答错了？是**检索没召回**，
    还是**召回了但模型没用好**？没有留痕就只能靠猜。

    本模块做的是**在线追踪**：每次 retrieve 落一行结构化 JSONL，记下
    query / 意图 / 走的哪条路径 / 命中了哪些块 / 各块得分 / 耗时。
    出问题时翻这一行就能定位到具体环节。

设计约束（遵守项目的既有风格）：
    - **永远不因追踪失败而影响主流程**：所有写盘异常静默吞掉。
    - **不重复实现意图判断**：从被装饰函数的 `__globals__` 里取 `_NEWS_HINTS`
      等常量（避免 trace_logger → rag_engine_v2 的循环导入）。
    - **落 JSONL 不落数据库**：一天一个文件，可直接 grep / jq，也能被别的进程读。
    - **开关**：环境变量 `RAG_TRACE=off` 关闭（默认开）。

产出路径：
    output/trace/retrieval-YYYYMMDD.jsonl
"""

import json
import os
import time
import functools
from datetime import datetime

try:
    from config import OUTPUT_DIR
except Exception:  # config 里没有 OUTPUT_DIR 时退回相对路径
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

TRACE_DIR = os.path.join(OUTPUT_DIR, "trace")
_ENABLED = os.environ.get("RAG_TRACE", "on").lower() != "off"

# 单条追踪里最多记几个命中块（防止长列表把日志撑爆）
_MAX_DOCS = int(os.environ.get("RAG_TRACE_MAX_DOCS", "10"))


def _doc_digest(doc):
    """把一条命中块压成可诊断的最小字段集。

    只取**.get()**，不假设字段名——检索各路径（关键词 / 向量 / 聚合）返回的
    dict 字段并不一致，取不到就记 None，绝不让追踪层反过来约束检索层。
    """
    if not isinstance(doc, dict):
        return {"text": str(doc)[:80]}
    text = (doc.get("text") or "")
    meta = doc.get("metadata") or {}
    return {
        "id": doc.get("id"),
        "type": meta.get("type") if isinstance(meta, dict) else None,
        "title": (meta.get("title") if isinstance(meta, dict) else None),
        # 各路得分分开记：量纲不同，**禁止跨量纲比较**（与 config.py 的约定一致）
        "score": doc.get("score") or doc.get("similarity") or doc.get("vector_sim"),
        "rerank_score": doc.get("rerank_score"),
        "_agg_line": doc.get("_agg_line"),
        "text_head": text[:60].replace("\n", " "),
    }


def _detect_intents(query, module_globals):
    """复用 rag_engine_v2 的意图词表，不改也不重实现。

    从被装饰函数的 `__globals__` 拿 `_XXX_HINTS`，拿不到就返回空——
    追踪层对主流程必须是**只读且可缺席**的。
    """
    mapping = {
        "news": "_NEWS_HINTS",
        "stock": "_STOCK_HINTS",
        "report": "_REPORT_HINTS",
        "profile": "_PROFILE_HINTS",
        "macro": "_MACRO_HINTS",
    }
    hits = []
    for name, const in mapping.items():
        hints = module_globals.get(const)
        if hints and any(h in query for h in hints):
            hits.append(name)
    return hits


def _route_of(docs):
    """判断这次检索走的是哪条路径（这是排障时最想先知道的一件事）。"""
    if not docs:
        return "empty"
    first = docs[0]
    if isinstance(first, dict) and first.get("_roster"):
        return "industry_roster"    # 行业→公司清单（REQ-050，按公司去重，不吃检索）
    if isinstance(first, dict) and first.get("_agg_header"):
        return "structured_rank"   # 聚合/排行路径（ORDER BY 类问题）
    return "hybrid"                 # 关键词优先 + 向量兜底


def _write(record):
    if not _ENABLED:
        return
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, "retrieval-%s.jsonl" % datetime.now().strftime("%Y%m%d"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 追踪写不进去绝不能影响问答


def trace_retrieval(fn):
    """给 `RAGEngine.retrieve` 用的装饰器。

    用法（rag_engine_v2.py 里只加这两行，不动函数体）：
        from src.trace_logger import trace_retrieval
        @trace_retrieval
        def retrieve(self, query, top_k=TOP_K):
    """
    @functools.wraps(fn)
    def wrapper(self, query, top_k=None):
        t0 = time.perf_counter()
        error = None
        docs = None
        try:
            docs = fn(self, query, top_k) if top_k is not None else fn(self, query)
            return docs
        except Exception as e:
            error = "%s: %s" % (type(e).__name__, e)
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            docs = docs if isinstance(docs, list) else []
            _write({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "top_k": top_k,
                "intents": _detect_intents(query, getattr(fn, "__globals__", {})),
                "route": _route_of(docs),
                "n_docs": len(docs),
                "elapsed_ms": elapsed_ms,
                "error": error,
                "docs": [_doc_digest(d) for d in docs[:_MAX_DOCS]],
            })
    return wrapper


def log_generation(query, *, model=None, prompt_chars=0, answer_chars=0,
                   prompt_tokens=None, completion_tokens=None, elapsed_ms=None, error=None):
    """生成侧追踪（可选接入）。

    检索侧已经自动落盘；生成侧要记 token，得在调用 LLM 的地方显式调一次。
    单独开一个文件 `generation-YYYYMMDD.jsonl`，与检索侧用 ts+query 对齐。
    """
    if not _ENABLED:
        return
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, "generation-%s.jsonl" % datetime.now().strftime("%Y%m%d"))
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "model": model,
            "prompt_chars": prompt_chars,
            "answer_chars": answer_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
