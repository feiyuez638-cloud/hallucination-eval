"""下载中文维基 2023-11 快照（HF wikimedia/wikipedia 20231101.zh，6 个 parquet，约 1.6 GiB）。

越越 2026-10-21 同意：「同意下载第 1、2 项（维基 2023-11 快照 1.6 GiB 和依赖包），bge-m3 等第二步再问我；先搭 BM25 检索原型」

  py rag/fetch_wiki.py

- 文件清单与 sha256 取自 HF API（LFS oid），下载用 curl 断点续传（-C -），走系统代理。
- 判成败不看 curl 退出码，只看落盘文件的大小和 sha256。已下好且校验通过的文件跳过。
"""
import hashlib, subprocess, sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/kb/wiki-zh-20231101"
API = "https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/20231101.zh"
BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [x for x in httpx.get(API, timeout=30).json() if x["path"].endswith(".parquet")]
    print(f"共 {len(files)} 个文件，{sum(x['size'] for x in files) / 2**30:.2f} GiB，存到 {OUT}", flush=True)
    bad = []
    for x in files:
        dst = OUT / Path(x["path"]).name
        want = x["lfs"]["oid"]
        if dst.exists() and dst.stat().st_size == x["size"] and sha256(dst) == want:
            print(f"已有 {dst.name}", flush=True)
            continue
        for attempt in range(1, 4):
            r = subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "--retry-delay", "5",
                                "-o", str(dst), BASE + x["path"]],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            size = dst.stat().st_size if dst.exists() else 0
            if size == x["size"]:
                break
            print(f"{dst.name} 第 {attempt} 次未下完 {size}/{x['size']} {r.stderr.strip()[:200]}", flush=True)
        ok = dst.exists() and dst.stat().st_size == x["size"] and sha256(dst) == want
        print(f"{dst.name} {x['size'] / 2**20:.0f} MiB  {'校验通过' if ok else '校验失败'}", flush=True)
        if not ok:
            bad.append(dst.name)
    print("全部校验通过" if not bad else f"校验失败：{bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
