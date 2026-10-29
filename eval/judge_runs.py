"""用 LLM 裁判给作答记录判幻觉（本地代理指标，非官方成绩）。

  py eval/judge_runs.py eval/runs/answers-xxx.jsonl [--judge gemini-pro] [--workers 3] [--limit N]
  py eval/judge_runs.py eval/runs/answers-xxx.jsonl --summary    # 只汇总已有判定，不调用

- 参考材料由 to_judge.reference 拼；judge.PROMPT 用 10-01 冻结版（24/24 验收），调用走 run_judge.call。
- 裁判只按参考材料判：与参考冲突才算幻觉，参考里没提到的多余细节不算（这是该口径的已知盲区）。
- 每条只调 1 次，不重试；连续 2 次失败停止。续跑：输出文件里已有有效判定的条目跳过。
- 输出 eval/runs/judged-<作答文件名>-<裁判>.jsonl（追加写）。作答失败（超时等）的条目不送裁判，汇总里单列。
"""
import argparse, json, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_judge  # noqa: E402
from judge import summarize  # noqa: E402
from to_judge import reference  # noqa: E402


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).exists() else []


def report(dev, answers, judged):
    v = {r["id"]: r for r in judged if r.get("parsed_ok")}
    print("【LLM 裁判：本地代理指标，非官方成绩】")
    print(f"作答 {len(answers)}  作答失败 {sum(a['error'] is not None for a in answers)}  已判 {len(v)}")
    groups = defaultdict(list)
    for i, r in v.items():
        d = dev[i]
        groups["全部"].append(r["verdict"])
        groups[d["source"]].append(r["verdict"])
        groups[f"{d['source']}/{d['category']}"].append(r["verdict"])
    for k in sorted(groups, key=lambda k: (k != "全部", "/" in k, k)):
        s = summarize(groups[k])
        pct = lambda x: "—" if x is None else f"{x:.1%}"  # noqa: E731
        print(f"{k}: n={s['success']}  幻觉 {s['hallucinated']}  拒答 {s['refused']}  "
              f"HR(拒答算有效) {pct(s['hr_refusal_counted'])}  HR(拒答剔除) {pct(s['hr_refusal_excluded'])}")
    # 2026-10-22 起两向分开报：官方对正常题过度拒答也扣分。可答题 = SimpleQA + HalluQA Knowledge；
    # 误导题拒答不算过度拒答（指出前提有误本来就对）。过度拒答 = 可答题上拒答且没编造。
    ans = [r["verdict"] for i, r in v.items()
           if dev[i]["source"] == "chinese_simpleqa" or dev[i]["category"] == "Knowledge"]
    over = sum(x["refused"] and not x["hallucinated"] for x in ans)
    hal = sum(r["verdict"]["hallucinated"] for r in v.values())
    print(f"两向口径：幻觉率 {hal}/{len(v)} = {hal / len(v):.1%}  "
          f"过度拒答率 {over}/{len(ans)} = {over / len(ans):.1%}（分母为可答题）" if v and ans else "两向口径：无已判数据")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--judge", default="gemini-pro", choices=list(run_judge.JUDGES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    model = run_judge.JUDGES[a.judge]
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    answers = load(a.answers)
    out = ROOT / "eval/runs" / f"judged-{Path(a.answers).stem}-{a.judge}.jsonl"
    if not a.summary:
        done = {r["id"] for r in load(out) if r.get("parsed_ok")}
        rows = [{"id": x["id"], "question": dev[x["id"]]["question"], "answer": x["answer"],
                 "reference": reference(dev[x["id"]])}
                for x in answers if x["error"] is None and x["id"] not in done][:a.limit]
        print(f"已判 {len(done)}，本次待判 {len(rows)}，裁判 {model}，并发 {a.workers}", flush=True)
        lock, state = threading.Lock(), {"fails": 0, "stop": False, "n": 0}
        with open(out, "a", encoding="utf-8") as f:
            def one(row):
                if state["stop"]:
                    return
                rec = run_judge.call(row, model)
                v = rec["verdict"]
                rec["parsed_ok"] = bool(v) and v["hallucinated"] is not None
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    state["n"] += 1
                    state["fails"] = 0 if rec["call_ok"] else state["fails"] + 1
                    if rec["error"] or state["n"] % 20 == 0:
                        print(f"{state['n']}/{len(rows)}  {rec['id']}  {rec['latency_s']}s  "
                              f"{(rec['error'] or '')[:160]}", flush=True)
                    if state["fails"] >= 2:
                        state["stop"] = True
                        print("连续 2 次调用失败，停止。", flush=True)
            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(one, rows))
        print("记录:", out, flush=True)
    report(dev, answers, load(out))


if __name__ == "__main__":
    main()
