"""bge-reranker-v2-m3 重排：BM25 先取前 N 段，交叉编码器给「问题-段落」打分，留前 k 段。

  py rag/rerank.py "南京条约是哪一年签订的" [-n 20] [-k 5]

- 模型本地目录 data/models/bge-reranker-v2-m3（rag/fetch_reranker.py 下载并核对哈希），只读 safetensors，不执行远程代码。
- 有 CUDA 用 fp16 上显卡，没有就退到 CPU（慢）。
"""
import argparse, sys, threading, time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data/models/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self._lock = threading.Lock()  # 10-27：代理把检索挪进线程池后，重排同一时刻只跑一个，防显存冲高和 tokenizer 并发报错
        self.score("预热", ["预热"])  # 首次调用有 CUDA 初始化开销，别算进时延

    @torch.inference_mode()
    def score(self, query, docs, batch=8):  # 10-23：和 qwen3:8b 同卡时 32 条×512 token 会爆显存
        out = []
        for i in range(0, len(docs), batch):
            part = docs[i:i + batch]
            enc = self.tok([query] * len(part), part, padding=True, truncation="longest_first",  # 10-23：三段式长题的问题本身超过 512，only_second 会报错

                           max_length=self.max_length, return_tensors="pt").to(self.device)
            out.extend(self.model(**enc).logits.view(-1).float().tolist())
        return out

    def rerank(self, query, hits, k):
        with self._lock:
            s = self.score(query, [h["title"] + "：" + h["text"] for h in hits])
        for h, x in zip(hits, s):
            h["rerank"] = round(x, 3)
        return sorted(hits, key=lambda h: -h["rerank"])[:k]


def main():
    sys.path.insert(0, str(ROOT / "rag"))
    from search import Retriever
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    r, rr = Retriever(), Reranker()
    print(f"设备 {rr.device}")
    t = time.time()
    hits = r.search(a.query, a.n)
    t1 = time.time()
    hits = rr.rerank(a.query, hits, a.k)
    print(f"BM25 {(t1 - t) * 1000:.0f}ms  重排 {(time.time() - t1) * 1000:.0f}ms")
    for h in hits:
        print(f"[{h['rerank']} | bm25 {h['score']}] {h['title']}｜{h['text'][:120]}")


if __name__ == "__main__":
    main()
