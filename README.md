# rag-finance-qa · 金融 RAG 问答

**端口** `5679`（明确避开 n8n 的 5678）· **入口** `src/web_app.py` · **向量库** ChromaDB

## 路由

`/` · `/api/ask` · `/api/health` · `/api/ingest` · `/api/stats` · `/api/suggest`

## 模块

| 文件 | 职责 |
|---|---|
| `src/rag_engine.py` / `rag_engine_v2.py` | 检索引擎 |
| `src/bm25_channel.py` | BM25 通道 |
| `src/data_extractor.py` | 数据抽取 |
| `src/enrich_database.py` | 库扩充 |
| `src/trace_logger.py` | 调用追踪 |
| `setup_api.py` | API 配置 |

## 启动

```bash
D:\ML_Projects\.venv\Scripts\python.exe D:\ML_Projects\rag-finance-qa\src\web_app.py
```

## 依赖

`flask` `chromadb==1.5.9` `pymysql` `requests` `sqlalchemy` `openai`

> ⚠️ **本文件是全仓对 `chromadb` 唯一声明正确的那份**（1.5.9）。
> 根 `requirements.txt` 在 2026-09-25 前写的是 `0.6.3` —— 跨两个大版本，
> 装了它本项目的向量库直接不可用。根文件已修正，但子项目 `requirements.txt` 里
> `flask==3.1.1` / `pymysql==1.1.1` / `openai==1.82.0` 仍低于实装（`3.1.3` / `1.2.0` / `2.50.0`）。

## 已知问题

- `openai` SDK 实装是 **v2.50.0**，而本目录声明 `1.82.0`。v1/v2 API 不兼容，
  若代码按 v1 写而环境是 v2，**调用会失败**。引用前确认代码里的调用姿势。

## 评测

`_rag_eval/` 下为评测集；`_tools/watchdog_rag5679.py` 是看门狗（带 `*_selftest.py`）。
