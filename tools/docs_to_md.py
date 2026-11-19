#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把研报/公告/财报等原始文档批量转成 Markdown，供 rag-finance-qa 的 RAG 索引。

为什么需要它：
  rag-finance-qa 吃的是文本。研报多为 PDF、公告多为 docx/xlsx，
  手写解析器又脆又慢。markitdown 统一转成 Markdown，省掉这一层。

依赖：markitdown（已装在托管 venv）
运行（必须用这个解释器）：
  C:/Users/shenh/.workbuddy/binaries/python/envs/default/Scripts/python.exe tools/docs_to_md.py <输入目录> [--out DIR] [--recursive]

支持格式：pdf, docx, doc, xlsx, xls, pptx, ppt, html, htm, csv, json, xml,
          epub, zip, 以及图片（jpg/png，会走 OCR/图像描述，较慢）
输出：默认写到 <项目>/data/docs_md/，同名 .md；另写 _failed.txt 记录失败文件。
"""

import argparse
import sys
import time
from pathlib import Path

SUFFIXES = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".html", ".htm", ".csv", ".json", ".xml", ".epub", ".zip",
    ".jpg", ".jpeg", ".png", ".txt", ".md",
}
# 超过这个大小直接跳过，避免卡死（默认 80MB）
MAX_BYTES = 80 * 1024 * 1024


def convert_one(md_converter, src: Path, out_dir: Path) -> tuple[bool, str]:
    """转换单个文件。返回 (是否成功, 说明)。"""
    dst = out_dir / (src.stem + ".md")
    if dst.exists():
        return True, "skip(exists)"
    try:
        result = md_converter.convert(str(src))
        text = getattr(result, "text_content", None)
        if not text or not text.strip():
            return False, "empty output"
        dst.write_text(text, encoding="utf-8")
        return True, f"{len(text)} chars"
    except Exception as e:  # noqa: BLE001 - 批量任务，单个失败不能中断整体
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="批量文档转 Markdown（markitdown）")
    ap.add_argument("src", help="输入文件或目录")
    ap.add_argument("--out", default=None, help="输出目录，默认 <项目>/data/docs_md")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 md")
    args = ap.parse_args()

    try:
        from markitdown import MarkItDown
    except ImportError:
        print("[FATAL] 找不到 markitdown。请用这个解释器运行：")
        print("  C:/Users/shenh/.workbuddy/binaries/python/envs/default/Scripts/python.exe")
        return 2

    src = Path(args.src)
    if not src.exists():
        print(f"[FATAL] 路径不存在: {src}")
        return 2

    project_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else project_root / "data" / "docs_md"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in out_dir.glob("*.md"):
            p.unlink()

    files: list[Path] = []
    if src.is_file():
        files = [src]
    else:
        it = src.rglob("*") if args.recursive else src.glob("*")
        files = [p for p in it if p.is_file() and p.suffix.lower() in SUFFIXES]

    if not files:
        print(f"[WARN] 没找到可转换的文件（后缀白名单: {sorted(SUFFIXES)}）")
        return 1

    print(f"[INFO] 待转换 {len(files)} 个文件 -> {out_dir}")
    md = MarkItDown()
    ok, failed, t0 = 0, [], time.time()

    for i, f in enumerate(files, 1):
        size = f.stat().st_size
        if size > MAX_BYTES:
            failed.append((f, "too large"))
            print(f"  [{i}/{len(files)}] SKIP(大文件) {f.name}")
            continue
        success, note = convert_one(md, f, out_dir)
        if success:
            ok += 1
            print(f"  [{i}/{len(files)}] OK   {f.name}  ({note})")
        else:
            failed.append((f, note))
            print(f"  [{i}/{len(files)}] FAIL {f.name}  ({note})")

    dt = time.time() - t0
    print(f"\n[DONE] 成功 {ok} / 共 {len(files)}，用时 {dt:.1f}s")
    print(f"[OUT ] {out_dir}")

    if failed:
        log = out_dir / "_failed.txt"
        log.write_text(
            "\n".join(f"{p}\t{why}" for p, why in failed), encoding="utf-8"
        )
        print(f"[WARN] {len(failed)} 个失败，清单: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
