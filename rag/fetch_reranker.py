"""下载重排模型 BAAI/bge-reranker-v2-m3（Apache-2.0，约 2.14 GiB）到 data/models/bge-reranker-v2-m3。

越越 2026-10-21 同意：「同意下载 GPU 方案：bge-reranker-v2-m3（2.14 GiB）、torch 2.14.1 CUDA 12.6 版（2.42 GiB）和 transformers 等小依赖；先验重排命中率，小样试答，全量前再问我」

  py rag/fetch_reranker.py

- 只下推理要用的 6 个文件（权重、分词、配置），图片和 README 不下。
- 文件清单取自 HF API：大文件核 sha256（LFS oid），小文件核 git blob sha1。curl 断点续传，走系统代理。
- 判成败不看 curl 退出码，只看落盘文件的大小和哈希。已下好且校验通过的跳过。
"""
import hashlib, subprocess, sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REPO = "BAAI/bge-reranker-v2-m3"
OUT = ROOT / "data/models/bge-reranker-v2-m3"
API = f"https://huggingface.co/api/models/{REPO}/tree/main"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
WANT = {"model.safetensors", "tokenizer.json", "sentencepiece.bpe.model", "tokenizer_config.json",
        "special_tokens_map.json", "config.json"}


def digest(p, x):
    if x.get("lfs"):
        h = hashlib.sha256()
        want = x["lfs"]["oid"]
    else:
        h = hashlib.sha1(f"blob {x['size']}\0".encode())
        want = x["oid"]
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest() == want


def ok(p, x):
    return p.exists() and p.stat().st_size == x["size"] and digest(p, x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [x for x in httpx.get(API, timeout=30).json() if x["type"] == "file" and x["path"] in WANT]
    assert {x["path"] for x in files} == WANT, [x["path"] for x in files]
    print(f"共 {len(files)} 个文件，{sum(x['size'] for x in files) / 2**30:.2f} GiB，存到 {OUT}", flush=True)
    bad = []
    for x in files:
        dst = OUT / x["path"]
        if ok(dst, x):
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
        good = ok(dst, x)
        print(f"{dst.name} {x['size'] / 2**20:.1f} MiB  {'校验通过' if good else '校验失败'}", flush=True)
        if not good:
            bad.append(dst.name)
    print("全部校验通过" if not bad else f"校验失败：{bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
