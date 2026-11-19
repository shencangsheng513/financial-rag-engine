# -*- coding: utf-8 -*-
"""用 requests（走本机代理）把 bge-small-zh-v1.5 拉到本地，绕开 huggingface_hub 的代理坑。

下载到本地目录后，build_zh_vector_db.py 用 --model <本地目录> 加载，全程不再联网。
"""
import os
import sys
import time
from pathlib import Path

import requests

PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
MODEL = "BAAI/bge-small-zh-v1.5"
OUT = Path("D:/ML_Projects/models/bge-small-zh-v1.5")
OUT.mkdir(parents=True, exist_ok=True)
API_URL = f"https://hf-mirror.com/api/models/{MODEL}"
RESOLVE = f"https://hf-mirror.com/{MODEL}/resolve/main"


def main():
    s = requests.Session()
    s.trust_env = False  # 本机代理 7897 已挂，直接连 hf-mirror.com（沙箱直连可达）
    s.headers.update({"User-Agent": "Mozilla/5.0"})

    r = s.get(API_URL, timeout=30)
    r.raise_for_status()
    sibs = r.json().get("siblings", [])
    print(f"[list] {len(sibs)} 个文件，写入 {OUT}")
    total = 0
    for item in sibs:
        fn = item["rfilename"]
        dst = OUT / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  skip {fn} ({dst.stat().st_size} bytes)")
            continue
        t0 = time.time()
        url = f"{RESOLVE}/{fn}"
        resp = s.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(1 << 16):
                if chunk:
                    f.write(chunk)
        total += dst.stat().st_size
        print(f"  ok {fn} -> {dst.stat().st_size} bytes ({time.time()-t0:.1f}s)")
    print(f"[done] 本地模型目录就绪：{OUT}（共 {total>>20} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
