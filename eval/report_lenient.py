"""判定汇总：严格解析 vs 宽松补解析（本地代理指标，非官方成绩）。不调用任何模型。

  py eval/report_lenient.py <作答文件> [--judge agnes]

- agnes 常在 reason 里写没转义的英文双引号，整段 JSON 非法，judge.parse_verdict（冻结版）判无结果；
  但两个布尔字段本身清楚。宽松补解析只对「调用成功、严格解析失败」的记录，从原始正文里各找一次
  "hallucinated": true|false 和 "refused": true|false，两个都恰好出现一次才采用，否则仍算无结果。
- 先在验收集上自检：宽松补出来的条目与预期对照。
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_judge  # noqa: E402
from judge_runs import load, report  # noqa: E402

PAT = {k: re.compile(rf'"{k}"\s*:\s*(true|false)') for k in ("hallucinated", "refused")}


def text_of(rec):
    raw = rec.get("raw_response") or {}
    if "choices" in raw:
        return ((raw["choices"] or [{}])[0].get("message") or {}).get("content") or ""
    return raw.get("response") or ""


def lenient(rec):
    if rec.get("parsed_ok") or not rec.get("call_ok"):
        return None
    t = text_of(rec)
    got = {k: p.findall(t) for k, p in PAT.items()}
    if all(len(v) == 1 for v in got.values()):
        return {"hallucinated": got["hallucinated"][0] == "true", "refused": got["refused"][0] == "true",
                "reason": "（宽松补解析）"}
    return None


def selfcheck(model):
    rows = {r["id"]: r for r in load(ROOT / "eval/judge_acceptance.jsonl")}
    for f in sorted(run_judge.RUNS.glob("*.jsonl")):
        if f.name.startswith("judged-"):
            continue
        for r in load(f):
            if r.get("model") == model and r["id"] in rows and (v := lenient(r)):
                e = rows[r["id"]]["expected"]
                ok = (v["hallucinated"], v["refused"]) == (e["hallucinated"], e["refused"])
                print(f"验收自检 {r['id']}（{f.name}）：宽松 {(v['hallucinated'], v['refused'])}  预期 "
                      f"{(e['hallucinated'], e['refused'])}  {'一致' if ok else '不一致'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--judge", default="agnes", choices=list(run_judge.JUDGES))
    a = ap.parse_args()
    model = run_judge.JUDGES[a.judge]
    selfcheck(model)
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    answers = load(a.answers)
    judged = load(ROOT / "eval/runs" / f"judged-{Path(a.answers).stem}-{a.judge}.jsonl")
    last = {}
    for r in judged:  # 同一条多次记录：有效判定优先，其次取最后一次
        if r["id"] not in last or r.get("parsed_ok") or not last[r["id"]].get("parsed_ok"):
            last[r["id"]] = r
    recs = list(last.values())
    print("\n===== 严格解析（judge.parse_verdict 冻结版）=====")
    report(dev, answers, recs)
    fixed, still = [], []
    for r in recs:
        v = lenient(r)
        if v:
            fixed.append({**r, "parsed_ok": True, "verdict": v})
        elif not r.get("parsed_ok"):
            still.append(r["id"])
    print(f"\n===== 严格 + 宽松补解析（补回 {len(fixed)} 条，仍无结果 {len(still)} 条）=====")
    report(dev, answers, [r for r in recs if r.get("parsed_ok")] + fixed)
    if still:
        print("仍无结果:", still)


if __name__ == "__main__":
    main()
