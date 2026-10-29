"""V6 门控单测 + 用已有 V4 答案离线回放（2026-10-22，按和 Codex 定的 V6-abstain 方案）。

  py eval/gate_eval.py data/dev/dev_v1.jsonl --replay eval/runs/answers-20261021-165805-v4-rerank-qwen3-4b.jsonl

- 只把题目送门控（与 serve/proxy.py 同一个提示和解析），不作答、不判分；直接连 Ollama，不经代理。
- 回放：门控判「是」的题把 V4 答案换成「我不知道」，其余保留原文；用 V4 已有 agnes 判分统计
  拦住幻觉 / 误拒答对 / 本来就拒答。类别只用来分组统计，不送进门控。
- 输出 eval/runs/gate-<ts>.jsonl。所有数字都是本地代理指标，非官方成绩。
"""
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
os.environ["STRATEGY"] = "b0"  # 只取门控提示，不加载检索和重排
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from serve.proxy import BACKEND_URL, GATE_TIMEOUT, gate_request, parse_gate  # noqa: E402
from compare_runs import verdicts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("--replay")
a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.inp, encoding="utf-8")]
grp = lambda d: d["category"] if d["source"] == "halluqa" else "SimpleQA"

out = ROOT / "eval/runs" / f"gate-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
cli = httpx.Client(trust_env=False, timeout=GATE_TIMEOUT)
gate = {}
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for n, d in enumerate(rows, 1):
        rec, t0 = {"id": d["id"], "raw": None, "verdict": None, "error": None}, time.time()
        try:
            r = cli.post(f"{BACKEND_URL}/chat/completions", json=gate_request(d["question"]))
            r.raise_for_status()
            rec["raw"] = r.json()["choices"][0]["message"]["content"]
            rec["verdict"] = parse_gate(rec["raw"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:200]
        rec["latency_s"] = round(time.time() - t0, 2)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        gate[d["id"]] = rec
        if n % 50 == 0:
            print(f"门控 {n}/{len(rows)}", flush=True)
print("记录:", out)

lat = sorted(g["latency_s"] for g in gate.values())
print(f"门控时延 P50 {lat[len(lat) // 2]}s  P95 {lat[int(len(lat) * 0.95)]}s  最长 {lat[-1]}s")
bad = [g for g in gate.values() if g["verdict"] is None]
print(f"无效或失败 {len(bad)}：", Counter((g["raw"] or "")[:10] if not g["error"] else "出错" for g in bad).most_common(8))
c = Counter((grp(d), gate[d["id"]]["verdict"]) for d in rows)
print("\n触发（判「是」）")
for k in ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard"):
    tot = sum(v for (g, _), v in c.items() if g == k)
    print(f"  {k:16s} {c[(k, '是')]:3d}/{tot}  否 {c[(k, '否')]:3d}  无效 {c[(k, None)]:2d}")

if a.replay:
    v4 = verdicts(a.replay, "agnes")
    lab = lambda v: "幻觉" if v["hallucinated"] else ("拒答" if v["refused"] else "答对")
    print("\n离线回放（V4 判分；触发的题换成「我不知道」）")
    tot = Counter()
    for k in ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard"):
        hit = Counter(lab(v4[d["id"]]) for d in rows if grp(d) == k and gate[d["id"]]["verdict"] == "是" and d["id"] in v4)
        tot += hit
        print(f"  {k:16s} 拦住幻觉 {hit['幻觉']:3d}  误拒答对 {hit['答对']:3d}  本来就拒答 {hit['拒答']:3d}")
    h4 = sum(v["hallucinated"] for v in v4.values())
    print(f"  合计：拦住幻觉 {tot['幻觉']}，误拒答对 {tot['答对']}；V4 幻觉 {h4} → 回放 {h4 - tot['幻觉']}（有判定 {len(v4)} 题）")
    print("  门槛（事先定）：幻觉减少且答对零损失 →", "通过" if tot["幻觉"] > 0 and tot["答对"] == 0 else "不通过")
    for d in rows:
        g = gate[d["id"]]
        if g["verdict"] == "是" and d["id"] in v4 and lab(v4[d["id"]]) == "答对":
            print(f"    误拒 {d['id']} [{grp(d)}] {d['question'][:50]}")
