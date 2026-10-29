"""把中文维基 parquet 建成 BM25 检索库（练习用；官方是否允许检索未知，不允许就整个弃用）。

  py rag/build_kb.py                      # 全量
  py rag/build_kb.py --max-batches 20     # 小样试建，输出到 wiki-bm25-test

流程：繁转简 → 截尾部小节、单篇限 8000 字 → 切 ≤350 字的段（段前拼标题参与分词）
→ jieba 分词 → 词用 64 位哈希代替词表 → 每 20 万段一个分片，按词排好倒排表
→ 全部建完后按全库 df/平均段长算 BM25 权重（k1=1.2, b=0.75）写回各分片。
本机可用内存只有 3 GB 左右，所以只开少量进程、分片建、在途任务数有上限。
"""
import argparse, json, shutil, sqlite3, sys, time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))
import kb_text  # noqa: E402

SRC = ROOT / "data/kb/wiki-zh-20231101"
K1, B = 1.2, 0.75


def work(batch):
    titles, texts, dl, nnz, hs, tfs = [], [], [], [], [], []
    for title, text in batch:
        lines = kb_text.clean(text)
        if sum(len(s) for s in lines) < 20:
            continue
        t = kb_text.t2s(title)
        for p in kb_text.chunk(lines):
            c = Counter(kb_text.tokens(t + "\n" + p))
            if not c:
                continue
            titles.append(t)
            texts.append(p)
            dl.append(sum(c.values()))
            nnz.append(len(c))
            hs.extend(kb_text.h64(w) for w in c)
            tfs.extend(min(v, 65535) for v in c.values())
    return (titles, texts, np.array(dl, np.int32), np.array(nnz, np.int32),
            np.array(hs, np.uint64), np.array(tfs, np.uint16))


def batches(files, max_batches):
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            tab = pf.read_row_group(g, columns=["title", "text"])
            yield list(zip(tab.column("title").to_pylist(), tab.column("text").to_pylist()))
            n += 1
            if max_batches and n >= max_batches:
                return


def flush(out, k, buf, base):
    dl = np.concatenate(buf["dl"])
    nnz = np.concatenate(buf["nnz"])
    H = np.concatenate(buf["h"])
    T = np.concatenate(buf["tf"])
    D = np.repeat(np.arange(len(dl), dtype=np.int32), nnz)
    vocab, inv = np.unique(H, return_inverse=True)
    del H
    order = np.argsort(inv, kind="stable")
    ptr = np.zeros(len(vocab) + 1, np.int64)
    np.cumsum(np.bincount(inv, minlength=len(vocab)), out=ptr[1:])
    del inv
    d = out / f"s{k:03d}"
    d.mkdir()
    np.save(d / "vocab.npy", vocab)
    np.save(d / "ptr.npy", ptr)
    np.save(d / "doc.npy", D[order])
    np.save(d / "tf.npy", T[order])
    np.save(d / "dl.npy", dl)
    return {"dir": d.name, "base": base, "n": int(len(dl)), "postings": int(len(T))}


def weigh(out, shards):
    """全库 df、平均段长 → 每条倒排的 BM25 权重。"""
    N = sum(s["n"] for s in shards)
    avgdl = sum(float(np.load(out / s["dir"] / "dl.npy").sum()) for s in shards) / N
    vs = [np.load(out / s["dir"] / "vocab.npy") for s in shards]
    dfs = [np.diff(np.load(out / s["dir"] / "ptr.npy")) for s in shards]
    gv, ginv = np.unique(np.concatenate(vs), return_inverse=True)
    df = np.bincount(ginv, weights=np.concatenate(dfs).astype(np.float64), minlength=len(gv))
    del ginv
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    for s, v in zip(shards, vs):
        d = out / s["dir"]
        ptr = np.load(d / "ptr.npy")
        tf = np.load(d / "tf.npy").astype(np.float32)
        doc = np.load(d / "doc.npy")
        dl = np.load(d / "dl.npy").astype(np.float32)
        term = np.repeat(np.arange(len(v), dtype=np.int32), np.diff(ptr))
        w = idf[np.searchsorted(gv, v)][term]
        w *= tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[doc] / avgdl))
        np.save(d / "w.npy", w.astype(np.float32))
        (d / "tf.npy").unlink()
        del tf, doc, dl, term, w
    return N, avgdl, len(gv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--shard", type=int, default=200_000)
    ap.add_argument("--max-batches", type=int, default=0)
    a = ap.parse_args()
    name = "wiki-bm25-test" if a.max_batches else "wiki-bm25"
    final = ROOT / "data/kb" / name
    out = final.with_name(name + ".tmp")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = sorted(SRC.glob("*.parquet"))
    print(f"源 {len(files)} 个 parquet，进程 {a.workers}，分片 {a.shard} 段", flush=True)

    db = sqlite3.connect(out / "passages.sqlite")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, title TEXT, text TEXT)")
    t0 = time.time()
    shards, buf, nid, nart, base = [], None, 0, 0, 0

    def new_buf():
        return {"dl": [], "nnz": [], "h": [], "tf": [], "n": 0}

    def take(res):
        nonlocal nid, buf, base
        titles, texts, dl, nnz, hs, tfs = res
        db.executemany("INSERT INTO p VALUES (?,?,?)", zip(range(nid, nid + len(texts)), titles, texts))
        nid += len(texts)
        for key, arr in (("dl", dl), ("nnz", nnz), ("h", hs), ("tf", tfs)):
            buf[key].append(arr)
        buf["n"] += len(texts)
        if buf["n"] >= a.shard:
            db.commit()
            shards.append(flush(out, len(shards), buf, base))
            base = nid
            print(f"分片 {len(shards) - 1}：{shards[-1]['n']} 段  累计 {nid} 段 {nart} 篇  {time.time() - t0:.0f}s", flush=True)
            buf = new_buf()

    buf = new_buf()
    with ProcessPoolExecutor(a.workers) as ex:
        q = deque()
        for b in batches(files, a.max_batches):
            nart += len(b)
            q.append(ex.submit(work, b))
            if len(q) >= a.workers * 2:
                take(q.popleft().result())
        while q:
            take(q.popleft().result())
    if buf["n"]:
        shards.append(flush(out, len(shards), buf, base))
    db.commit()
    db.close()
    print(f"分词建倒排完成：{nid} 段 {nart} 篇 {len(shards)} 分片  {time.time() - t0:.0f}s", flush=True)

    N, avgdl, V = weigh(out, shards)
    meta = {"source": "HF wikimedia/wikipedia 20231101.zh", "passages": N, "articles": nart, "avgdl": avgdl,
            "vocab": V, "k1": K1, "b": B, "art_cap": kb_text.ART_CAP, "passage_max": kb_text.PASSAGE_MAX,
            "shards": shards, "build_s": round(time.time() - t0), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    print(f"完成：{final}  词 {V}  平均段长 {avgdl:.1f} 词  共 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
