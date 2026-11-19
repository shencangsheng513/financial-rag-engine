"""通用静态守卫 —— 零依赖，任何环境（含容器构建）都能跑。

钉住 AI 写代码最容易重演的洞（已在 quant-platform / market-monitor 等验证）：
- 明文密钥进仓库/镜像
- .dockerignore 没挡住 *.bak / .env
- 监听地址写死 0.0.0.0（本机挂公网 IP，会直接暴露）
- web 入口硬编码绝对路径（容器内必然失效）

不依赖具体入口文件名：自动 grep `app.run(` 定位监听地址。
CLI/工具类文件（ingest、tools、report_downloader 等）里的硬编码路径是本地 bug、
非 web 暴露面，降级为 warning 不阻断（避免误杀合法本地脚本）。

运行： python -m unittest tests.test_static_guards -v
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ABS_PATH = re.compile(r'["\'](?:D|C):[/\\]', re.IGNORECASE)
SECRET = re.compile(
    r'(?:password|passwd|secret|api_key|apikey|token)\s*["\']?\s*[:=]\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# CLI / 工具类文件：硬编码路径是本地 bug，不是 web 暴露面，降级为 warning
CLI_HINT = re.compile(r'(cli|ingest|tools?|report_downloader|docs_to_md)', re.I)


def _code_lines(p):
    out = []
    for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        out.append((ln, line.split("#", 1)[0]))
    return out


def _py_files():
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts or "tests" in p.parts:
            continue
        yield p


class TestNoPlaintextSecrets(unittest.TestCase):
    """C2：明文密钥是上云的头号雷（已多次进镜像）。"""

    def test_no_plaintext_secret(self):
        offenders = []
        for f in list(_py_files()) + list(ROOT.glob("*.bak")):
            for ln, code in _code_lines(f):
                m = SECRET.search(code)
                if not m:
                    continue
                v = m.group(1)
                if not v or v.startswith(("os.", "{", "$")) or "environ" in v:
                    continue
                offenders.append(f"{f.relative_to(ROOT)}:{ln}: {code.strip()[:70]}")
        self.assertEqual([], offenders, "发现明文密钥赋值。")


class TestDockerignoreBlocksSecrets(unittest.TestCase):
    """C2 镜像侧防线。"""

    def test_exists(self):
        self.assertTrue((ROOT / ".dockerignore").exists(), "缺 .dockerignore")

    def test_entries(self):
        c = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        miss = [e for e in ("*.bak", ".env")
                if not any(l.strip() == e for l in c.splitlines())]
        self.assertEqual([], miss, f".dockerignore 缺: {miss}")


class TestBindAddressNotWideOpen(unittest.TestCase):
    """H1：本机挂公网 IP，app.run(host='0.0.0.0') 直接暴露。
    只查 app.run 实际调用，不查 config.py 里的 WEB_HOST 死配置。"""

    def test_app_run_not_0_0_0_0(self):
        offenders = []
        for f in _py_files():
            for ln, code in _code_lines(f):
                if re.search(r'app\.run\([^)]*host\s*=\s*["\']0\.0\.0\.0["\']', code):
                    offenders.append(f"{f.relative_to(ROOT)}:{ln}")
        self.assertEqual([], offenders, "监听地址写死 0.0.0.0 会暴露公网。")


class TestNoHardcodedAbsPathInWeb(unittest.TestCase):
    """H2：web 入口路径必须可移植；CLI 工具降级为 warning。"""

    def test_web_paths(self):
        warnings = []
        for f in _py_files():
            rel = str(f.relative_to(ROOT))
            is_cli = bool(CLI_HINT.search(rel))
            for ln, code in _code_lines(f):
                if ABS_PATH.search(code):
                    if is_cli:
                        warnings.append(f"{rel}:{ln}")
                    else:
                        self.fail(f"{rel}:{ln} web 代码含硬编码绝对路径: {code.strip()[:60]}")
        if warnings:
            print("\n[WARN] 以下 CLI/工具文件含硬编码路径（不阻断，建议改相对）:")
            for w in warnings:
                print("  " + w)


class TestWriteEndpointsAreGuarded(unittest.TestCase):
    """C1：/api/ingest（定时函数调用的写接口）必须带 @require_key('write')。
    rag 原有全站零鉴权，新增的入库端点若漏装饰器会重新裸奔。"""

    WRITE_ENDPOINTS = ("api_ingest",)

    def test_require_key_on_write_endpoints(self):
        src = (ROOT / "src" / "web_app.py").read_text(encoding="utf-8")
        for fn in self.WRITE_ENDPOINTS:
            m = re.search(r'((?:@[^\n]+\n)*)def ' + fn + r'\b', src)
            self.assertIsNotNone(m, f"找不到 {fn} 的定义")
            self.assertIn("require_key", m.group(1),
                          f"{fn} 缺少 @require_key('write') —— 定时入库接口会裸奔。")

    def test_require_key_is_fail_closed(self):
        src = (ROOT / "src" / "web_app.py").read_text(encoding="utf-8")
        block = re.search(r'def require_key.*?(?=\n(?:@app|def |class ))', src, re.S)
        self.assertIsNotNone(block, "找不到 require_key 定义")
        self.assertIn("503", block.group(0),
                      "require_key 未实现 fail-closed —— 忘配密钥时会放行写请求。")


class TestNoShellOrEval(unittest.TestCase):
    """硬规矩 7：不许引入 subprocess / os.system / eval，当前为零，保持。
    2026-09-13 补 —— 此前该条只有文字规定，没有守卫。"""

    FORBIDDEN = re.compile(
        r'\b(?:import|from)\s+subprocess\b|\bsubprocess\s*\.'
        r'|\bos\.system\s*\(|\bos\.popen\s*\(|\bcommands\.getoutput\s*\('
        r'|(?<![.\w])eval\s*\(|(?<![.\w])exec\s*\('
    )
    # 注意 (?<![.\w]) —— 排除 model.eval() / pd.eval() / df.eval() 这类合法的
    # 库 API（quant-platform 与 ML 项目大量使用）。内置 eval 前不会是「.」。
    # 另注意**不匹配裸词 subprocess** —— 2026-09-13 实测：写 `\bsubprocess\b` 会把
    # docstring / 字符串里「提到」subprocess 的行也判违规（探针自爆验证过）。
    # 真实使用必然带 `import` 或 `.`，所以只认这两种形态。

    def test_no_shell_or_eval(self):
        offenders = []
        for f in _py_files():
            for ln, code in _code_lines(f):
                if self.FORBIDDEN.search(code):
                    offenders.append(f"{f.relative_to(ROOT)}:{ln}: {code.strip()[:70]}")
        self.assertEqual([], offenders, "引入了 shell/eval —— 本项目当前为零，保持。")


class TestSqlIsParameterized(unittest.TestCase):
    """硬规矩 6：SQL 的「值」必须参数化 %s。

    分两级（2026-09-13 二次修正 —— 第一版把 f-string 一刀切判 fail，实测发现会误报）：
      - **fail**：`%` 拼接 / `.format()` —— 这两种在 SQL 场景几乎只用于拼「值」，置信度高
      - **warning**：f-string —— 合法场景真实存在（拼表名、拼 `%s` 占位符数量），
        正则无法区分「拼表名」与「拼值」，只提示不阻断，由人确认

    实测反例（四处均为**正确**写法，第一版会全部误杀）：
      `finance-webapp/src/app.py:230`（拼 placeholders 个数）
      `quant-platform/src/data_fetcher.py:151/171`、`expand_data.py:108`（拼表名）
      `stock-predictor/src/fetch_akshare.py:211/298`（拼模块常量 TABLE）
    """

    FAIL_PATTERNS = (
        (re.compile(r'execute\w*\s*\([^\n]{0,200}?\.format\s*\('), ".format()"),
        (re.compile(r'execute\w*\s*\([^\n]{0,200}?["\']\s*%\s*[\(\w]'), "% 拼接"),
    )
    WARN_FSTRING = re.compile(r'execute\w*\s*\(\s*[fF]["\']')

    def test_no_sql_value_concatenation(self):
        offenders = []
        for f in _py_files():
            for ln, code in _code_lines(f):
                for pat, kind in self.FAIL_PATTERNS:
                    if pat.search(code):
                        offenders.append(
                            f"{f.relative_to(ROOT)}:{ln} [{kind}]: {code.strip()[:60]}"
                        )
        self.assertEqual([], offenders, "SQL 值出现拼接 —— 必须改参数化 %s。")

    def test_fstring_sql_listed_for_review(self):
        """f-string SQL 只列清单不阻断（合法用法存在，见类 docstring）。"""
        hits = []
        for f in _py_files():
            for ln, code in _code_lines(f):
                if self.WARN_FSTRING.search(code):
                    hits.append(f"{f.relative_to(ROOT)}:{ln}")
        if hits:
            print("\n[WARN] f-string SQL —— 仅允许拼「表名/占位符」，值必须走 %s 参数化；"
                  "表名建议加白名单校验：")
            for h in hits:
                print("  " + h)


class TestGuardsNotWeakened(unittest.TestCase):
    """硬规矩 8：不许删测试、也不许改测试让它变绿。
    只钉「守卫本身」：预期守卫类必须都在，且不许用 skip 绕过。
    合法新增测试不受影响；要移除某条守卫必须显式修改本清单（会留下 diff 记录）。
    2026-09-13 补。"""

    REQUIRED = (
        "TestNoPlaintextSecrets",
        "TestDockerignoreBlocksSecrets",
        "TestBindAddressNotWideOpen",
        "TestNoHardcodedAbsPathInWeb",
        "TestWriteEndpointsAreGuarded",
        "TestNoShellOrEval",
        "TestSqlIsParameterized",
    )

    def _src(self):
        return Path(__file__).read_text(encoding="utf-8")

    def test_guard_classes_present(self):
        src = self._src()
        miss = [c for c in self.REQUIRED if f"class {c}(" not in src]
        self.assertEqual([], miss, f"守卫类被删除/改名: {miss} —— 不许删测试让它变绿。")

    def test_no_skip_escape(self):
        # 只拦「装饰器级 skip」（会整体禁用守卫）。不拦运行时 self.skipTest ——
        # 那是条件性跳过，有合法场景（如前置资源缺失），拦了会误杀。
        bad = re.findall(r'@unittest\.(?:skip|expectedFailure)\b', self._src())
        self.assertEqual([], bad, "守卫测试被 skip —— 等于没有守卫。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
