# -*- coding: utf-8 -*-
"""引用可核性守卫：答案里声称来自代码的东西，必须能在**被引用的那段窗口**里核到。

病：`rag_engine_v2.build_context()` 把检索窗口拼成 `[1] [2] …` 交给模型，prompt 要求答案标
`[编号]`。而今天没有任何一处检查"标了没标对"，也没有任何一处检查"答案里的标识符真在那段代码里"。
⇒ 幻觉出来的 `DynamicBatchSampler`、指向不存在的 `[9]` 的角标，一路畅通到用户眼前，
而且看起来比真答案更像有依据（它带着角标）。
这是本仓在评测侧已经钉过的那一族的线上版：**"没核到"被读成"核过了"**。

三态（互斥，故意没有第四态"看起来没问题"）：
  PASS        有可核符号，且全部在引用面里核到了
  FLAG        有可核符号，但有核不到的 / 有悬空角标 —— 逐条点名
  UNVERIFIED  压根没有可核符号，或没有可引用的窗口。**这一态不许折算成通过**
              （它就是评测侧那条 `cov==0` 地板闸的反面：一个依据都没有时，"没查出问题"最贵）

口径（为什么这么判，判据只住这一个文件）：
  什么算"代码形状的符号"（**结构判据，不列词表**，词表一定会腐烂）：
    含下划线且长度 >= 4      `sensitive_word`
    点号分隔                 `f5_tts.models.utils`
    带文件后缀或行号         `utils/infer.py` / `utils/infer.py:318`
    驼峰且大写字母 >= 2 个   `DynamicBatchSampler`
    单首大写的普通英文词不算（`Python`、`Config`）⇒ 散文里的词不会变成"编造证据"。
  核到哪里：只在**答案真正引用了的那些窗口**里核，不是整个检索池、更不是整个语料。
    符号只出现在没被引用的窗口里 = FLAG（"引用面"这三个字必须有牙齿）。
  两档一起报：全局档（在被引用窗口的并集里找得到）与就近档（在该句自己引用的那个窗口里）。
    判据走**全局**那宽的一档（跨窗口拼起来的事实不该演成编造），
    差值与"无归属句"只报不判 —— 它们是给接线那一轮看的，不是今天的红。

不做什么（刻意的边界，接线前请先读这段）：
  * 不发网络、不读语料、不 import `rag_engine_v2`（那个文件此刻正在别的会话手里改）。
  * 不改任何生产代码：本模块今天**没有调用点**。它的调用点在工单里登记成待办，
    而不是在这里假装已经接上。
  * 不做"语义对不对"：这里只回答"这句话里那个标识符，你引的那段代码里有没有"。

跑它的守卫（零网络、可复跑）：
    cd rag-finance-qa && D:\\ML_Projects\\.venv\\Scripts\\python.exe -B tests/check_cited_symbols.py
"""
import re

STATE_PASS = "PASS"
STATE_FLAG = "FLAG"
STATE_UNVERIFIED = "UNVERIFIED"
STATES = (STATE_PASS, STATE_FLAG, STATE_UNVERIFIED)

FILE_EXT = r"(?:py|js|ts|tsx|jsx|json|ya?ml|toml|md|sh|sql|html|css|csv|txt|ipynb)"

# 三条候选，逐条对应"真会进答案的形状"；提取必须一遍完成（分多遍会互相吞：
# 先抠驼峰再把 `f5_tts.models` 拆成两个词）。路径与行号各有自己那条（下面两条正则），
# 所以这里只认「有分隔符的」和「两个以上大写的」两种形状 —— 单首大写的普通英文词
# （Python、Config）压根进不了候选，不需要词表。
SYM_CAND = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|_[A-Za-z0-9_]+)+"
    r"|[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)")
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_\-]+\.(?:%s)(?::\d+)?" % FILE_EXT)
ROW_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*:\d+\b")
CITE_RE = re.compile(r"\[(\d{1,3})\]")
SENT_SPLIT = re.compile(r"[。！？\n]|\?\?|!+")

_WINDOW_KEYS = ("text", "content", "window", "page_content")


def _text_of(item):
    """一个检索槽位 -> 文本。生产侧 `retrieved` 是 dict，测试与人工核查也常直接给 str。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in _WINDOW_KEYS:
            v = item.get(k)
            if isinstance(v, str):
                return v
    return ""


def looks_like_symbol(tok):
    """结构判据（唯一一处，词表一律不用）：这一串值不值得拿去核。

    四种形状分别对应：`sensitive_word`（下划线）/ `utils/infer.py:318`（路径或行号）/
    `f5_tts.models.utils`（点号）/ `DynamicBatchSampler`（两个以上大写）。
    长度下限挡掉 `a_b`、`py` 这种碎片；单首大写的普通英文词四条都不命中，所以不会误伤散文。
    """
    if len(tok) < 4:
        return False
    if "/" in tok or "_" in tok or ":" in tok:
        return True
    if re.search(r"[A-Za-z0-9_]\.[A-Za-z0-9_]", tok):
        return True
    return sum(1 for c in tok if c.isupper()) >= 2 and any(c.islower() for c in tok)


def extract_symbols(text):
    """文本 -> 顺序去重的"代码形状"标识符清单（不判真假，只负责"这一串值得去核"）。

    三条正则会有覆盖（`foo_bar:12` 同时被行号那条和下划线那条各抠一次），
    这里按"包含关系"丢短的：同一句话里的同一个说法数两遍会把分母吹大。
    """
    out = []
    for rx in (PATH_RE, ROW_RE, SYM_CAND):
        for m in rx.finditer(text):
            tok = m.group(0).strip("._-")
            if not tok or not looks_like_symbol(tok):
                continue
            if any(tok == x or tok in x or x in tok for x in out):
                continue
            out.append(tok)
    return out


def cited_indices(text):
    """答案里出现过的所有角标（去重、保序）。"""
    seen = []
    for m in CITE_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def _sentence_of(text, pos):
    """按字符位置回找该符号所在的那一句（就近档用这个来找"这句话引了哪几个窗口"）。"""
    start = 0
    for m in SENT_SPLIT.finditer(text):
        if m.start() >= pos:
            break
        start = m.end()
    end = len(text)
    for m in SENT_SPLIT.finditer(text):
        if m.start() > pos:
            end = m.start()
            break
    return text[start:end]


def verify(answer, contexts):
    """答案 + 检索槽位 -> 三态判定与逐条点名（纯函数，无副作用）。

    返回 dict 里 `n_windows` / `n_symbols` / `n_cited_windows` 都是**现算**的分母：
    接线那一轮要把它们打进屏面，不许回头抄常数。
    """
    windows = [_text_of(c) for c in (contexts or [])]
    nwin = len(windows)
    cites = cited_indices(answer)
    dangling = [c for c in cites if not 1 <= c <= nwin]
    # 引用面：只有被角标点到的窗口参与"核得到"这件事
    cited_idx = [c - 1 for c in cites if 1 <= c <= nwin]
    union = "\n".join(windows[i] for i in cited_idx)
    symbols = extract_symbols(answer)
    missing = [s for s in symbols if s.lower() not in union.lower()]
    near_missing, unattributed = [], []
    for s in symbols:
        pos = answer.find(s)
        sent = _sentence_of(answer, pos if pos >= 0 else 0)
        own = [int(m.group(1)) - 1 for m in CITE_RE.finditer(sent) if 1 <= int(m.group(1)) <= nwin]
        if not own:
            unattributed.append(s)
            continue
        if not any(s.lower() in windows[i].lower() for i in own):
            near_missing.append(s)
    if nwin == 0:
        state = STATE_UNVERIFIED
    elif dangling or missing:
        state = STATE_FLAG
    elif not symbols:
        state = STATE_UNVERIFIED
    else:
        state = STATE_PASS
    return dict(state=state, n_windows=nwin, n_cited_windows=len(cited_idx),
                n_symbols=len(symbols), n_cites=len(cites), symbols=symbols,
                cites=cites, dangling=dangling, missing=missing,
                near_missing=near_missing, unattributed=unattributed)
