"""
RAG 问答引擎：检索 + 生成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
import requests
import json
import re

# 语料正文的金额单位是万元，而 8B 模型每次都要自行 ÷10000 —— 实测会错 1~2 个数量级
# （2026-09-10 线上：语料「市值约 41487292.75 万元」被答成「41,487.29 万元，即约 414.87 亿元」）。
# 这里在**检索后、送 LLM 前**把 >=1 亿元的金额补上亿元口径，让模型只需照抄。
# 不改写向量库文本：embedding 不受影响，逻辑集中在 build_context，随时可回撤。
#   (?<!（) 用于幂等：已转换的形态是「4,148.73亿元（41,487,292.75万元）」，
#   括号内那个万元数紧跟「（」，不会被再次匹配。
# 语料正文的金额单位是万元，而 8B 模型每次都要自行 ÷10000 —— 实测会错 1~2 个数量级
# （2026-09-10 线上：语料「市值约 41487292.75 万元」被答成「41,487.29 万元，即约 414.87 亿元」）。
# 这里在**检索后、送 LLM 前**把 >=1 亿元的金额**直接替换**成亿元口径，让模型只需照抄。
# 不改写向量库文本：embedding 不受影响，逻辑集中在 build_context，随时可回撤。
#
# ⚠️ 为什么是「替换」而不是「补一个括号」：
# 先试过后一种——输出「4,148.73亿元（41,487,292.75万元）」，结果模型放着现成的亿元不用，
# 又去拿括号里的万元数自己做除法，算成「4.15亿元」（仍错 1000 倍）。
# **上下文里同时出现两种单位 = 引诱模型重算**；只留一个权威数字才稳。
# 幂等性也因此天然成立：替换后文本里不再有「万元」，重复执行不再匹配。
_AMOUNT_WAN = re.compile(r"([\d,]+(?:\.\d+)?)万元")

# 问题里出现这些词，说明用户要的是「研报内容」，实体过滤时优先取 research_report 块
_REPORT_HINTS = ("研报", "报告", "研究", "解读", "观点", "评级", "点评", "深度")


def _with_yi(text: str) -> str:
    """『市值约 41487292.75 万元』→『市值约 4,148.73 亿元』（详见上方注释）"""
    def repl(m):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            return m.group(0)
        if abs(v) >= 10000:
            return f"{v / 1e4:,.2f}亿元"
        return m.group(0)   # < 1 亿元：原样保留，小数额用万元更直观

    return _AMOUNT_WAN.sub(repl, text)

# 加载 .env 文件
# .env 查找顺序：本应用根目录（旧布局）→ 套件仓库根 .env。
# 2026-09-23 自 data-plane 副本跟进（部署侧 2026-09-11 修）：部署包里本目录 .env 已移出——
# 它会随 code.zip 整包上云（.dockerignore 只挡镜像构建、不挡 zip），明文 Key 不应随源码走；
# 云上由平台 EnvParams 注入 MODELSCOPE_API_KEY，这里的 setdefault 只是本地兜底。
# 候选层级按源码布局适配：源码侧仓库根=上两级（部署侧镜像里是 /ml，多一层）。
_env_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))), ".env"),
]
_env_path = next((p for p in _env_candidates if os.path.exists(p)), None)
if _env_path:
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, MAX_CONTEXT_LENGTH
from config import LLM_API_BASE, LLM_MODEL, LLM_MAX_TOKENS, PROXY


class FinanceRAGEngine:
    """金融 RAG 问答引擎"""
    
    def __init__(self):
        print("正在初始化 RAG 引擎...")
        
        # 连接 ChromaDB（使用内置 Embedding）
        print("  连接 ChromaDB...")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        
        doc_count = self.collection.count()
        print(f"  数据库中共 {doc_count} 个文档块")

        # 已收录的公司名清单，供 retrieve() 做「实体过滤」用（一次读取，常驻内存）。
        # 为什么需要：见 retrieve() 注释——纯向量检索在中文短查询上几乎取不到研报块。
        try:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            self.known_companies = sorted({m.get("company") for m in metas if m.get("company")},
                                          key=len, reverse=True)  # 长名优先，避免「嘉实科技」被「嘉实」误配
        except Exception:
            self.known_companies = []
        print(f"  已收录公司 {len(self.known_companies)} 家")
        print("[OK] RAG 引擎就绪！\n")

    def retrieve(self, query, top_k=TOP_K):
        """检索最相关的文档：**向量召回 + 实体过滤**两路合并。

        ⚠️ 为什么不能只用向量召回：
        内置 embedding 是 all-MiniLM-L6-v2（**英文模型**），对中文的区分度很差——
        实测 269 块的库里，无论问「鹏华电子的研报里说了什么」还是直接问报告标题
        「鹏华电子深度研究报告」，返回的前 5 条距离都在 0.78~1.0（近乎正交），
        **research_report 类型的块几乎永远进不了 top-k**，等于 89 份研报白灌。
        短的结构化块（overview 等）反而更容易命中，因为公司名密度高、模板化强。

        所以这里补一路确定性的实体过滤：问题里若出现已收录的公司名，
        就直接按 metadata 把该公司的块取出来（这才是「问哪家答哪家」），
        问的是「研报/报告/研究」时再让 research_report 排前面。
        两路合并时实体块优先，向量块补足到 top_k。
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        vector_hits = [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

        matched = [c for c in getattr(self, "known_companies", []) if c in query]
        if not matched:
            return vector_hits

        want_report = any(h in query for h in _REPORT_HINTS)
        entity_hits = []
        for c in matched:
            got = self.collection.get(where={"company": c},
                                      include=["documents", "metadatas"])
            rows = [
                {
                    "id": got["ids"][i],
                    "text": got["documents"][i],
                    "metadata": got["metadatas"][i],
                    # 实体精确命中，不是距离度量；置 0 让 score(=1-distance) 显示为满分
                    "distance": 0.0,
                }
                for i in range(len(got["ids"]))
            ]
            # 问研报时把 research_report 提到前面；否则保持库内顺序
            rows.sort(key=lambda d: 0 if (not want_report or d["metadata"].get("type") == "research_report") else 1)
            entity_hits.extend(rows)

        merged, seen = [], set()
        for d in entity_hits + vector_hits:
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            merged.append(d)
            if len(merged) >= top_k:
                break
        return merged
    
    def build_context(self, retrieved_docs):
        """将检索到的文档拼接成上下文"""
        context_parts = []
        total_length = 0
        
        for doc in retrieved_docs:
            text = _with_yi(doc["text"].strip())
            if total_length + len(text) > MAX_CONTEXT_LENGTH:
                break
            context_parts.append(text)
            total_length += len(text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_with_modelscope(self, query, context, api_key):
        """使用 ModelScope API 生成回答"""
        prompt = f"""你是一个专业的金融分析师助手。请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 只根据参考资料回答，不要编造数据
2. 如果参考资料中没有相关信息，诚实说明
3. 回答要专业、准确、有条理
4. 【金额单位】参考资料里的金额**已经换算好了**：大于等于 1 亿元的写作「亿元」，
   小于 1 亿元的写作「万元」。请**直接照抄该数字和它后面的单位**，
   **不要再做任何换算、乘除或单位推断**。例如资料写「4,148.73亿元」就答「4,148.73亿元」，
   不要改写成别的数。若同一指标出现多个互相矛盾的数值，请如实指出矛盾。
5. 【百分比】ROE、毛利率、资产负债率等字段已是百分数，直接引用即可，不要再乘 100。
6. 涉及数字时要精确引用

【回答】"""
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的金融分析师助手，擅长分析公司财务数据、股票行情和行业趋势。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": LLM_MAX_TOKENS,
            }
            
            proxies = {"http": PROXY, "https": PROXY} if PROXY else {"http": None, "https": None}
            
            # 使用流式模式（Qwen3 推理模型需要）
            payload["stream"] = True
            resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=120
            )
            resp.raise_for_status()
            
            # 收集流式响应
            content_parts = []
            _sse_parse_errors = 0
            _reasoning_chars = 0
            _finish_reason = None
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    chunk_str = line.decode("utf-8")[6:]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            if choices[0].get("finish_reason"):
                                _finish_reason = choices[0]["finish_reason"]
                            delta = choices[0].get("delta", {})
                            # Qwen3 推理模型：reasoning_content 是思考过程，content 是最终答案
                            if delta.get("reasoning_content"):
                                _reasoning_chars += len(delta["reasoning_content"])
                            if "content" in delta and delta["content"]:
                                content_parts.append(delta["content"])
                    except Exception:
                        _sse_parse_errors += 1  # 单块解析失败跳过，汇总在流结束后可见

            result = "".join(content_parts)
            if _sse_parse_errors:
                print(f"[llm] SSE 流有 {_sse_parse_errors} 个数据块解析失败（已跳过）", flush=True)
            if result:
                return result
            if _finish_reason == "length":
                # 与 v2 同修：预算被 reasoning 吃光不是 Key 问题（2026-09-24 V6 实测）
                return (f"[模型把 token 预算全部花在推理上（max_tokens={LLM_MAX_TOKENS}，"
                        f"思考约 {_reasoning_chars} 字，正文 0 字），本次无法生成回答。]\n\n"
                        f"以下是检索到的相关资料，请自行参考：\n\n{context}")
            return "[模型未返回有效内容，请检查 API Key 和模型名称]"
        
        except Exception as e:
            return f"[LLM 调用失败：{e}]\n\n以下是检索到的相关资料，请自行参考：\n\n{context}"
    
    def generate_without_llm(self, query, context, retrieved_docs):
        """不使用 LLM，直接返回检索结果（离线模式）"""
        answer_parts = [
            f"## 检索到 {len(retrieved_docs)} 条相关信息\n",
            "### 参考资料\n",
        ]
        
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc["metadata"]
            score = 1 - doc["distance"]  # 转换为相似度
            answer_parts.append(
                f"**[{i}] {meta.get('company', '?')}（{meta.get('stock_code', '?')}）"
                f"- {meta.get('type', '?')}** （相似度：{score:.2%}）\n"
                f"{doc['text']}\n"
            )
        
        answer_parts.append("\n> 💡 以上是从数据库中检索到的最相关信息。如需更智能的分析，请配置 LLM API。")
        return "\n".join(answer_parts)
    
    def ask(self, query, use_llm=True):
        """完整 RAG 流程：检索 → 构建上下文 → 生成回答"""
        print(f"[Q] {query}\n")
        
        # 1. 检索
        print("[检索] 相关文档...")
        retrieved = self.retrieve(query)
        print(f"   找到 {len(retrieved)} 条相关记录\n")
        
        # 2. 构建上下文
        context = self.build_context(retrieved)
        
        # 3. 生成回答
        api_key = os.environ.get("MODELSCOPE_API_KEY", "")
        if use_llm and api_key:
            print("[LLM] 调用生成回答...")
            answer = self.generate_with_modelscope(query, context, api_key)
        else:
            print("[离线] 返回检索结果...")
            answer = self.generate_without_llm(query, context, retrieved)
        
        return {
            "query": query,
            "answer": answer,
            "sources": retrieved,
        }
    
    def stats(self):
        """返回数据库统计信息"""
        count = self.collection.count()
        
        # 按类型统计
        all_docs = self.collection.get(include=["metadatas"])
        type_counts = {}
        company_counts = {}
        
        for meta in all_docs["metadatas"]:
            t = meta.get("type", "unknown")
            c = meta.get("company", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            company_counts[c] = company_counts.get(c, 0) + 1
        
        return {
            "total_documents": count,
            "by_type": type_counts,
            "by_company": company_counts,
        }


if __name__ == "__main__":
    # 测试
    engine = FinanceRAGEngine()
    
    # 测试问题
    test_queries = [
        "科技行业有哪些公司？",
        "哪家公司的市值最高？",
        "最近哪只股票涨得最好？",
        "哪家公司的ROE最高？",
    ]
    
    for q in test_queries:
        result = engine.ask(q, use_llm=False)
        print(f"\n{'='*60}")
        print(f"问题：{result['query']}")
        print(f"回答：\n{result['answer']}")
        print(f"{'='*60}\n")
