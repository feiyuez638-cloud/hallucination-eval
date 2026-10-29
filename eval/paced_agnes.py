"""agnes 免费档慢速裁判（本地代理指标，非官方成绩）。越越 2026-10-20 定：
「等 3 分钟后用 agnes 慢速续跑剩下 5 条，每条间隔 8 秒；还是 429 就停下告诉我；验收过了就按同样节奏判 449 题」

  py eval/paced_agnes.py <作答文件> [--wait 180] [--gap 8]

- 判定逻辑不变：每条走 run_judge.call（judge.PROMPT 冻结版），只加等待、间隔和 429 即停。
- 验收：补跑 judge_acceptance 里 agnes 还没有有效结果的条目，记录写 eval/runs/<时间>-paced.jsonl（run_judge.done_ids 认得）。
  调用成功过的条目不再调（解析失败也不重抽）。过关 = 不一致 + 无结果 ≤ 2 条；不过就停。
- 判题：写 eval/runs/judged-<作答文件名>-agnes.jsonl（与 judge_runs.py 同名同格式，追加，已判跳过），串行。
- 任何一次 429 立即停；其他失败连续 2 次停。
"""
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run_judge  # noqa: E402
from judge_runs import load, report  # noqa: E402
from to_judge import reference  # noqa: E402

MODEL = run_judge.MODEL


class Stop(Exception):
    pass


def paced(rows, out, gap, on_rec):
    fails = 0
    with open(out, "a", encoding="utf-8") as f:
        for k, row in enumerate(rows):
            if k:
                time.sleep(gap)
            rec = run_judge.call(row, MODEL)
            v = rec["verdict"]
            rec["parsed_ok"] = bool(v) and v["hallucinated"] is not None
            on_rec(row, rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if rec["error"] and "429" in rec["error"]:
                raise Stop(f"{rec['id']} 撞 429 限流，停止。")
            fails = 0 if rec["call_ok"] else fails + 1
            if fails >= 2:
                raise Stop("连续 2 次调用失败，停止。")


def acceptance(gap):
    rows = {r["id"]: r for r in load(ROOT / "eval/judge_acceptance.jsonl")}
    called = set()  # 调用成功过就不再调（输出解析不了也算一次，不重抽到过关为止）
    for f in sorted(run_judge.RUNS.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("model") == MODEL and r.get("call_ok") and r["id"] in rows and r["prompt_sha256"] == run_judge.prompt_hash(rows[r["id"]]):
                called.add(r["id"])
    todo = [rows[i] for i in rows if i not in called]
    print(f"验收：待补 {len(todo)} 条", flush=True)

    def on_rec(row, rec):
        exp = row["expected"]
        rec["match"] = rec["parsed_ok"] and (rec["verdict"]["hallucinated"], rec["verdict"]["refused"]) == (
            exp["hallucinated"], exp["refused"])
        got = (rec["verdict"]["hallucinated"], rec["verdict"]["refused"]) if rec["parsed_ok"] else None
        print(f"{row['id']}  调用{'成功' if rec['call_ok'] else '失败'}  {rec['latency_s']}s  "
              f"预期{(exp['hallucinated'], exp['refused'])}  裁判{got}  {'一致' if rec['match'] else '不一致/无结果'}"
              + (f"  错误:{rec['error'][:120]}" if rec["error"] else ""), flush=True)

    paced(todo, run_judge.RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-paced.jsonl", gap, on_rec)
    done = run_judge.done_ids(rows, MODEL)
    match = set()
    for f in sorted(run_judge.RUNS.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("model") == MODEL and r["id"] in done and r.get("match") and r["prompt_sha256"] == run_judge.prompt_hash(rows[r["id"]]):
                match.add(r["id"])
    bad = sorted(set(rows) - match)
    print(f"验收汇总：有效 {len(done)}/24  一致 {len(match)}  不一致或无结果 {bad}", flush=True)
    return len(bad) <= 2  # 越越口径「全对或只错一两条」，无结果按错算


def judge_answers(answers_path, gap):
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    answers = load(answers_path)
    out = run_judge.RUNS / f"judged-{Path(answers_path).stem}-agnes.jsonl"
    done = {r["id"] for r in load(out) if r.get("call_ok")}  # 调用成功过就不再调；解析失败的交给 report_lenient 补解析
    rows = [{"id": x["id"], "question": dev[x["id"]]["question"], "answer": x["answer"],
             "reference": reference(dev[x["id"]])}
            for x in answers if x["error"] is None and x["id"] not in done]
    print(f"判题：已判 {len(done)}，本次待判 {len(rows)}，裁判 {MODEL}，串行间隔 {gap}s", flush=True)
    n = [0]

    def on_rec(row, rec):
        n[0] += 1
        if rec["error"] or n[0] % 20 == 0:
            print(f"判题进度 {n[0]}/{len(rows)}  {rec['id']}  {rec['latency_s']}s  {(rec['error'] or '')[:160]}", flush=True)

    try:
        paced(rows, out, gap, on_rec)
        print("判题完成。记录:", out, flush=True)
    finally:
        report(dev, answers, load(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--wait", type=int, default=180)
    ap.add_argument("--gap", type=float, default=8)
    a = ap.parse_args()
    print(f"{datetime.now():%H:%M:%S} 先等 {a.wait} 秒", flush=True)
    time.sleep(a.wait)
    try:
        if not acceptance(a.gap):
            print("验收没过，停止，不判 449 题。", flush=True)
            return
        print("验收通过，开始判题。", flush=True)
        judge_answers(a.answers, a.gap)
    except Stop as e:
        print("停止：", e, flush=True)
    print(f"{datetime.now():%H:%M:%S} 结束", flush=True)


if __name__ == "__main__":
    main()
