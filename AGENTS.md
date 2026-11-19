# AGENTS.md — rag-finance-qa 操作护栏

给下一个改动这个仓库的智能体看。这些规矩是 2026-09-10 交付审计
（intended-vs-implemented）钉住的，违反任何一条都会把已修好的洞重新打开。

## 这个服务是什么
端口 5679 的 Flask 后端（金融 RAG 问答）。正在迁移到腾讯云 CloudBase Run。
**它是生产服务，不是本地脚本。**

## 硬规矩（不许违反）

1. **不许硬编码绝对路径**（`D:/ML_Projects/...` 之类）。一律用相对路径或环境变量推导。
   容器里绝对路径必然失效。

2. **不许明文密钥**——数据库密码、API key、token 一律走环境变量。
   `os.environ.get("XXX", "明文默认值")` 这种写法**尤其禁止**：失败是静默的。

3. **新增任何写接口（POST/PUT/DELETE）必须鉴权**。本项目当前全只读 GET，
   一旦加写接口，必须加 `@require_key("write")` 类鉴权，否则任何人能改生产数据。

4. **监听地址不许写死 `0.0.0.0`** 在 `app.run` 里。容器部署由平台注入，
   本地一律 `HOST=127.0.0.1`（见第 9 条）。安全性由鉴权 + 网关承担，不是监听地址。

5. **不许提交 `*.bak` / `.env`**。`.dockerignore` 已排除它们，不要绕过。
   明文密码/密钥曾因此被 `COPY .` 打进镜像，已发生多次泄露。

6. **SQL 必须参数化 `%s`**，不许字符串拼接。
   （守卫 `TestSqlIsParameterized`：抓 f-string / `.format()` / `%` 三种拼接，2026-09-13 起自动查）

7. **不许引入 `subprocess` / `os.system` / `eval`**，当前为零，保持。
   （守卫 `TestNoShellOrEval`。注意写法 `(?<![.\w])eval\s*\(` 会放行 `model.eval()` 这类库 API——
   以后若本服务引入模型推理，别把这条正则改回 `\beval\s*\(`，会误杀。）

8. **不许删测试、也不许改测试让它变绿。** 测试红了就修代码，不是修测试。
   改完跑 `python -m unittest discover -s tests -p "test_*.py"`，必须全绿
   （当前 **17 项** = 12 守卫 + 5 评测基线门禁）。
   （守卫 `TestGuardsNotWeakened` 钉住这一点：预期守卫类少一个就 FAIL，加 `@unittest.skip` 也 FAIL。
   要移除某条守卫必须显式改 `REQUIRED` 清单，会在 diff 里留下记录。）

9. **本地启动必须 `HOST=127.0.0.1`**。本机网卡挂**公网 IP 153.35.55.197**，
   `0.0.0.0` 会把服务直接暴露到公网。只有容器/云部署才用 `0.0.0.0`。

## 改动后必做

```bash
cd D:/ML_Projects/rag-finance-qa
python -m unittest discover -s tests -p "test_*.py"   # 必须全绿（17 项）
python _rag_eval/run_eval.py                          # 重跑评测 → output/eval_report_<tag>.json
python -m unittest tests.test_eval_baseline           # 跑完立刻验基线，低于就红
```

评测说明（2026-09-11 建）：`_rag_eval/eval_cases.json` 用例按 (company, type)/(sector, type)
粒度做集合运算，不依赖裁判模型。A/B/C/E 计入召回，D 类是结构化排行（看是否走
structured_rank），H 类是反向检查（expect_none 的公司绝不允许出现）。
**改检索逻辑必须先跑 `--baseline` 再跑升级版对比，H 出现 VIOLATION 一律算失败。**

基线门禁（2026-09-13 加，`REVIEW.md` §5 第 3 条）：`tests/test_eval_baseline.py` 读
`_rag_eval/eval_baseline.json` 快照比对 —— 召回低于快照（容差 0.01）、跌破绝对底线 0.80、
或 H 违规数增加，任一即 FAIL。**改基线是显式动作**：确认指标提升后手改快照文件，
降基线必须在提交信息里写理由。未登记的 tag 只提示不拦（给新实验留路）。
注意它的固有盲区：**改了检索代码但没重跑 eval，门禁仍然绿** —— 报告是静态文件。

已知缺口：E03「家电行业的研报观点」两种兜底都召不回（合成语料限制），留作哨兵。

启动：`HOST=127.0.0.1 python src/web_app.py`（生产入口，v2 引擎）。
不配置密钥时，任何写接口应被拒绝（fail-closed），而不是放行。

## 检索管线现状（2026-09-11 四件套升级后）

- `src/bm25_channel.py`：BM25 词法通道，与实体关键词路做 RRF 融合；
  `BM25_AS_FALLBACK` 默认 off（合成语料模板词表让 BM25 无区分度，依据见 config 注释）。
- 分数量纲显式化：每条检索结果带 `doc["trace"]["stages"]`（stage/dim/value），
  阈值只允许配在 0~1 量纲（`VECTOR_SIM_THRESHOLD`，默认 0=只记录）。
- 父子分块：研报子块命中后 `build_context` 回填整个父文档（同 title 按 chunk 序拼接），
  受 `PARENT_MAX_CHARS` / `MAX_CONTEXT_LENGTH` 约束。
- `structured_rank` 的正则已按**真实语料模板**校准（全角冒号 + 万元后缀 +
  括号注释），改语料模板必须同步改 `_FIN_FIELDS`/`_OVW_CAP` 并重跑评测。
- `RAG_V2_UPGRADES=off` 可整体回退升级（评测基线开关），bug 修复不受它控制。

## 已知待办（人类部分）
- [ ] 确认谁在触发这个服务的数据更新（仓库内无调度器代码）。
- [ ] 若接了外部 API key，确认 key 不在仓库/镜像里（曾发生 .env 泄露）。
