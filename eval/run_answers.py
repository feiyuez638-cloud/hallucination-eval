"""让被测接口逐题作答并落盘（练习阶段跑本机模型，不是官方成绩）。

  py eval/run_answers.py data/dev/dev_v1.jsonl --tag p0-qwen3-4b [--base http://127.0.0.1:8000/v1] [--limit N]

- 只发 question 一条 user 消息，不带系统提示、不带 gold / 参考材料（B0 口径）。
- 解码参数不传，用后端默认，实际值从 /x/info 记进 meta。
- 逐条落盘：发送的消息与哈希、回答、耗时、finish_reason、用量、错误。连续 3 次失败停止。
- 输出 eval/runs/answers-<时间>-<tag>.jsonl，元信息 .meta.json。
"""
import argparse, hashlib, json, time
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 60  # 与官方单次上限一致


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()][:a.limit]
    http = httpx.Client(trust_env=False, timeout=TIMEOUT)
    cli = OpenAI(base_url=a.base, api_key="local", http_client=http, max_retries=0)
    out = ROOT / f"eval/runs/answers-{datetime.now():%Y%m%d-%H%M%S}-{a.tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    meta = {"args": vars(a), "n": len(rows), "started": datetime.now().isoformat(timespec="seconds"),
            "info": http.get(a.base.removesuffix("/v1") + "/x/info").json()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    fails = 0
    with open(out, "w", encoding="utf-8") as f:
        for k, r in enumerate(rows, 1):
            msgs = [{"role": "user", "content": r["question"]}]
            rec = {"id": r["id"], "messages_sha256": hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest(),
                   "question": r["question"]}
            t0 = time.time()
            try:
                resp = cli.chat.completions.create(model="x", messages=msgs)
                c = resp.choices[0]
                rec.update(answer=c.message.content, finish_reason=c.finish_reason, model=resp.model,
                           usage=resp.usage.model_dump() if resp.usage else None, error=None)
                if (resp.model_extra or {}).get("x_retrieved") is not None:  # V3 起记下检索到的段落
                    rec["x_retrieved"] = resp.model_extra["x_retrieved"]
                if (resp.model_extra or {}).get("x_gate") is not None:  # V6 起记下门控结果
                    rec["x_gate"] = resp.model_extra["x_gate"]
                fails = 0
            except Exception as e:  # noqa: BLE001
                rec.update(answer=None, error=repr(e)[:300])
                fails += 1
            rec["latency_s"] = round(time.time() - t0, 2)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0 or rec["error"]:
                print(f"{k}/{len(rows)}  {rec['latency_s']}s  {rec['error'] or ''}", flush=True)
            if fails >= 3:
                print("连续 3 次失败，停止。", flush=True)
                break
    print("记录:", out)


if __name__ == "__main__":
    main()
