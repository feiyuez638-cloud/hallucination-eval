"""检索命中率：dev 集 SimpleQA 题的标准答案是否出现在前 k 段里（本地代理指标，非官方成绩）。

  py rag/eval_retrieval.py [--kb wiki-bm25] [--rerank N]

- --rerank N：BM25 先取前 N 段，再用 bge-reranker-v2-m3 重排（rag/rerank.py），命中名次按重排后的顺序算；
  同一批段落的 BM25 原名次也记下（bm25_rank），方便并排看。

- 只算 SimpleQA：答案是短实体/数字，子串匹配有意义。HaluQA 的标准答案是整句解释，只统计时延。
- 按题目来源分两组：出处含 zh.wikipedia 的（库里理应有）和不含的。
- 注意：Chinese SimpleQA 部分题出自维基，所以这里的命中率偏乐观。
- 逐题结果（前 20 段的 id/标题、首次命中名次）写到 eval/runs/retrieval-<kb>-<时间>.jsonl。
"""
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))
import kb_text  # noqa: E402
from search import Retriever  # noqa: E402

KS = (1, 5, 10, 20)


def variants(gold):
    out = set()
    for g in gold:
        g = kb_text.t2s(g).lower().strip()
        if g:
            out.add(g)
            if g.strip("《》“”\"") != g:
                out.add(g.strip("《》“”\""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="wiki-bm25")
    ap.add_argument("--rerank", type=int, default=0, help="BM25 前 N 段送重排，0 = 不重排")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT / "data/dev/dev_v1.jsonl", encoding="utf-8")]
    r = Retriever(a.kb)
    rr = None
    if a.rerank:
        from rerank import Reranker
        rr = Reranker()
        print(f"重排：BM25 前 {a.rerank} 段 → bge-reranker-v2-m3（{rr.device}）", flush=True)
    print(f"库 {a.kb}：{r.meta['passages']} 段  题 {len(rows)}", flush=True)

    def first(hits, gs):
        for i, h in enumerate(hits):
            s = (h["title"] + "\n" + h["text"]).lower()
            if any(g in s for g in gs):
                return i + 1
        return None

    lat, lat_rr, recs = [], [], []
    for n, d in enumerate(rows, 1):
        t = time.time()
        hits = r.search(d["question"], max(max(KS), a.rerank))
        lat.append(time.time() - t)
        bm25_hits = hits
        if rr:
            t = time.time()
            hits = rr.rerank(d["question"], [dict(h) for h in hits[:a.rerank]], max(KS))
            lat_rr.append(time.time() - t)
        rank = bm25_rank = None
        if d["source"] == "chinese_simpleqa":
            gs = variants(d["gold"])
            rank = first(hits, gs)
            bm25_rank = first(bm25_hits[:a.rerank or max(KS)], gs)
        rec = {"id": d["id"], "source": d["source"], "wiki": any("zh.wikipedia" in u for u in d["url"]),
               "rank": rank, "ms": round(lat[-1] * 1000, 1), "top": [[h["id"], h["title"]] for h in hits]}
        if rr:
            rec.update(bm25_rank=bm25_rank, rerank_ms=round(lat_rr[-1] * 1000, 1))
        recs.append(rec)
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    L = np.array(lat) * 1000
    print(f"BM25 时延 ms：P50 {np.percentile(L, 50):.0f}  P95 {np.percentile(L, 95):.0f}  最大 {L.max():.0f}")
    if rr:
        L = np.array(lat_rr) * 1000
        print(f"重排时延 ms：P50 {np.percentile(L, 50):.0f}  P95 {np.percentile(L, 95):.0f}  最大 {L.max():.0f}")
        sq = [x for x in recs if x["source"] == "chinese_simpleqa"]
        print(f"送重排的 {a.rerank} 段里含答案（重排能达到的上限）：{sum(1 for x in sq if x['bm25_rank']) / len(sq):.1%}")
    print("【检索命中率：本地代理指标，非官方成绩】")
    for name, f in (("SimpleQA 全部", lambda x: True), ("出处含维基", lambda x: x["wiki"]),
                    ("出处不含维基", lambda x: not x["wiki"])):
        g = [x for x in recs if x["source"] == "chinese_simpleqa" and f(x)]
        cells = "  ".join(f"@{k} {sum(1 for x in g if x['rank'] and x['rank'] <= k) / len(g):.1%}" for k in KS)
        print(f"{name}: n={len(g)}  {cells}")
    tag = f"-rerank{a.rerank}" if a.rerank else ""
    out = ROOT / "eval/runs" / f"retrieval-{a.kb}{tag}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"逐题记录：{out}")


if __name__ == "__main__":
    main()
