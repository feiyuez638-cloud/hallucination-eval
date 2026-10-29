"""BM25 检索：加载 build_kb.py 建好的库，查前 k 段。

  py rag/search.py "南京条约是哪一年签订的"
  py rag/search.py --kb wiki-bm25-test "问题"

倒排表用 mmap 读，常驻内存只有各分片词表（几十 MB）。
"""
import argparse, json, sqlite3, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))
import kb_text  # noqa: E402


class Retriever:
    def __init__(self, kb="wiki-bm25"):
        d = ROOT / "data/kb" / kb
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.shards = []
        for s in self.meta["shards"]:
            p = d / s["dir"]
            self.shards.append((s["base"], s["n"], np.load(p / "vocab.npy"), np.load(p / "ptr.npy"),
                                np.load(p / "doc.npy", mmap_mode="r"), np.load(p / "w.npy", mmap_mode="r")))
        self.db = sqlite3.connect(f"file:{d / 'passages.sqlite'}?mode=ro", uri=True, check_same_thread=False)
        self._db_lock = threading.Lock()  # 10-27：代理多线程检索共用这一个连接
        kb_text.tokens("预热分词词典")  # jieba 首次调用要加载词典约 0.4s，别算进检索时延

    def search(self, query, k=5):
        q = np.unique(np.array([kb_text.h64(w) for w in kb_text.tokens(kb_text.t2s(query))], np.uint64))
        if not len(q):
            return []
        cand_id, cand_s = [], []
        for base, n, vocab, ptr, doc, w in self.shards:
            i = np.searchsorted(vocab, q)
            ok = i < len(vocab)
            i = i[ok][vocab[i[ok]] == q[ok]]
            if not len(i):
                continue
            docs = np.concatenate([doc[ptr[j]:ptr[j + 1]] for j in i])
            ws = np.concatenate([w[ptr[j]:ptr[j + 1]] for j in i])
            sc = np.bincount(docs, weights=ws, minlength=n)
            top = np.argpartition(-sc, k)[:k] if n > k else np.arange(n)
            top = top[sc[top] > 0]
            cand_id.append(top + base)
            cand_s.append(sc[top])
        if not cand_id:
            return []
        ids, ss = np.concatenate(cand_id), np.concatenate(cand_s)
        o = np.argsort(-ss)[:k]
        out = []
        for pid, s in zip(ids[o], ss[o]):
            with self._db_lock:
                title, text = self.db.execute("SELECT title, text FROM p WHERE id=?", (int(pid),)).fetchone()
            out.append({"id": int(pid), "score": round(float(s), 3), "title": title, "text": text})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--kb", default="wiki-bm25")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    t = time.time()
    r = Retriever(a.kb)
    print(f"加载 {time.time() - t:.2f}s  {r.meta['passages']} 段")
    t = time.time()
    hits = r.search(a.query, a.k)
    print(f"检索 {(time.time() - t) * 1000:.0f}ms")
    for h in hits:
        print(f"[{h['score']}] {h['title']}｜{h['text'][:150]}")


if __name__ == "__main__":
    main()
