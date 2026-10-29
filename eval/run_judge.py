"""小批量真实裁判验证（本地裁判验收，不是官方比赛成绩）。

  py eval/run_judge.py stage1          # A1 B1 C1 D1 E1 F1 串行
  py eval/run_judge.py stage2          # 其余 18 条（仅在 stage1 六条全成功且全一致后运行）
  py eval/run_judge.py resume          # 续跑：跳过 eval/runs/ 里已有有效结果（且提示词哈希一致）的条目
  py eval/run_judge.py all gemini-pro  # 换裁判独立一轮跑 24 条；记录文件名带裁判名，与 agnes 结果分开统计

- 只把 question / answer / reference 填进 judge.PROMPT；预期标签与理由不发送。
- 每条只发 1 次 HTTP 请求（retries=1），不重试；连续 2 次调用失败立即停止。
- 默认裁判 agnes-2.5-flash（用户 2026-09-09 确认不计费）；gemini-pro = gemini-3.1-pro-high 会员道（agy CLI，
  用户 2026-10-01 指定）；gemini-flash-high = gemini-3.8-flash-high 会员道（2026-10-20 用户选）。只用命令行指定的裁判，不自动切换；续跑只认同一模型的历史结果。
- 每次调用落盘：模型、参数、发送的提示词哈希、原始响应、耗时、解析结果、失败信息。
"""
import hashlib, json, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"C:\Users\Honerable\Workspace\agent-desk")
import agnes  # noqa: E402
import gemini  # noqa: E402
from judge import PROMPT, parse_verdict  # noqa: E402

MODEL = agnes.FLASH
PARAMS = {"max_tokens": 6000, "temperature": 0}
JUDGES = {"agnes": MODEL, "gemini-pro": gemini.AGY_PRO,
          "gemini-flash-high": "gemini-3.8-flash-high"}  # 2026-10-20 会员道已无 3.1 Pro，越越选 3.8 Flash 高档，先跑验收
# agy 传不进 temperature/max_tokens；gemini._ask_agy 会在提示词前加「（不要调用任何工具，直接用文字回答。）」
AGY_PARAMS = {"backend": "agy CLI", "temperature": "不可设", "prefix": "（不要调用任何工具，直接用文字回答。）"}
TIMEOUT = 120
RUNS = ROOT / "eval/runs"  # 测试可替换
STAGE1 = ["A1", "B1", "C1", "D1", "E1", "F1"]


def call(row, model=MODEL):
    prompt = PROMPT.format(question=row["question"], reference=row["reference"], answer=row["answer"])
    rec = {"id": row["id"], "model": model, "params": PARAMS if model == MODEL else AGY_PARAMS, "timeout": TIMEOUT,
           "prompt_sha256": prompt_hash(row),
           "started": datetime.now().isoformat(timespec="seconds")}
    t0 = time.time()
    if model != MODEL:  # 会员道：只有正文和用量，没有 OpenAI 格式原始响应
        try:
            text, usage = gemini.ask(prompt, model=model, timeout=TIMEOUT, return_usage=True, purpose="hallucination-judge")
        except Exception as e:  # noqa: BLE001
            rec.update(latency_s=round(time.time() - t0, 2), call_ok=False, error=repr(e)[:500], raw_response=None,
                       verdict=None)
            return rec
        rec.update(latency_s=round(time.time() - t0, 2), call_ok=True, error=None,
                   raw_response={"response": text, "usage": usage}, verdict=parse_verdict(text))
        return rec
    try:
        resp = agnes.raw({"model": MODEL, "messages": [{"role": "user", "content": prompt}], **PARAMS},
                         timeout=TIMEOUT, retries=1)
    except Exception as e:  # noqa: BLE001
        rec.update(latency_s=round(time.time() - t0, 2), call_ok=False, error=repr(e)[:500], raw_response=None,
                   verdict=None)
        return rec
    rec.update(latency_s=round(time.time() - t0, 2), call_ok=True, error=None, raw_response=resp)
    msg = ((resp.get("choices") or [{}])[0].get("message") or {})
    rec["verdict"] = parse_verdict(msg.get("content") or "")  # 只解析正文，不回落到推理内容
    return rec


def prompt_hash(row):
    p = PROMPT.format(question=row["question"], reference=row["reference"], answer=row["answer"])
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


def done_ids(rows, model=MODEL):
    """已有有效结果的条目：同一裁判模型、历史记录里可解析，且当时发送的提示词与现在完全相同（哈希一致）。"""
    done = set()
    for f in sorted(RUNS.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("parsed_ok") and r.get("model") == model and r["id"] in rows and r["prompt_sha256"] == prompt_hash(rows[r["id"]]):
                done.add(r["id"])
    return done


def main():
    stage = sys.argv[1]
    judge = sys.argv[2] if len(sys.argv) > 2 else "agnes"
    model = JUDGES[judge]
    rows = {r["id"]: r for r in map(json.loads, (ROOT / "eval/judge_acceptance.jsonl").read_text(encoding="utf-8").splitlines())}
    if stage == "resume":  # 只调用还没有有效结果的条目，按样例顺序
        done = done_ids(rows, model)
        ids = [i for i in rows if i not in done]
        print(f"已有有效结果 {len(done)} 条，本次待调用 {len(ids)} 条", flush=True)
    else:
        ids = list(rows) if stage == "all" else STAGE1 if stage == "stage1" else [i for i in rows if i not in STAGE1]
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{stage}{'' if judge == 'agnes' else '-' + judge}.jsonl"
    out.parent.mkdir(exist_ok=True)
    fails = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in ids:
            rec = call(rows[i], model)
            exp = rows[i]["expected"]
            v = rec["verdict"]
            rec["parsed_ok"] = bool(v) and v["hallucinated"] is not None
            rec["match"] = rec["parsed_ok"] and (v["hallucinated"], v["refused"]) == (exp["hallucinated"], exp["refused"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            got = (v["hallucinated"], v["refused"]) if rec["parsed_ok"] else None
            print(f"{i}  调用{'成功' if rec['call_ok'] else '失败'}  {rec['latency_s']}s  "
                  f"预期{(exp['hallucinated'], exp['refused'])}  裁判{got}  {'一致' if rec['match'] else '不一致/无结果'}"
                  + (f"  错误:{rec['error'][:120]}" if rec["error"] else ""), flush=True)
            fails = 0 if rec["call_ok"] else fails + 1
            if fails >= 2:
                print("连续 2 次调用失败，停止。", flush=True)
                break
    print("记录:", out)


if __name__ == "__main__":
    main()
