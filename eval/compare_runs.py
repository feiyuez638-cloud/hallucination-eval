"""几次作答的裁判结果并排比（本地代理指标，非官方成绩）。不调用任何模型。

  py eval/compare_runs.py <作答文件1> <作答文件2> ... [--judge agnes]

判定 = 严格解析 + report_lenient 宽松补解析；同一条多次记录取有效那次。
每组列：有判定 n、幻觉、拒答、答对（既不幻觉也不拒答）、HR 两种口径。作答失败（超时）不送裁判，单列条数。
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_judge  # noqa: E402
from judge_runs import load  # noqa: E402
from report_lenient import lenient  # noqa: E402


def verdicts(answers_path, judge):
    out = {}
    for r in load(ROOT / "eval/runs" / f"judged-{Path(answers_path).stem}-{judge}.jsonl"):
        v = r["verdict"] if r.get("parsed_ok") else lenient(r)
        if v and v["hallucinated"] is not None:
            out[r["id"]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--judge", default="agnes", choices=list(run_judge.JUDGES))
    a = ap.parse_args()
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    print("【LLM 裁判：本地代理指标，非官方成绩】")
    for p in a.answers:
        ans = load(p)
        v = verdicts(p, a.judge)
        g = defaultdict(list)
        for i, x in v.items():
            d = dev[i]
            for k in ("全部", d["source"], f"{d['source']}/{d['category']}" if d["source"] == "halluqa" else None):
                if k:
                    g[k].append(x)
        print(f"\n== {Path(p).stem}  作答失败 {sum(x['error'] is not None for x in ans)}  有判定 {len(v)}")
        for k in sorted(g, key=lambda k: (k != "全部", "/" in k, k)):
            xs = g[k]
            n, h, r = len(xs), sum(x["hallucinated"] for x in xs), sum(x["refused"] for x in xs)
            ok = sum(not x["hallucinated"] and not x["refused"] for x in xs)
            kept = [x for x in xs if not x["refused"]]
            hk = sum(x["hallucinated"] for x in kept)
            print(f"  {k}: n={n}  幻觉 {h}  拒答 {r} ({r / n:.0%})  答对 {ok} ({ok / n:.0%})  "
                  f"HR(拒答算有效) {h / n:.1%}  HR(拒答剔除) {(hk / len(kept)) if kept else 0:.1%}")


if __name__ == "__main__":
    main()
