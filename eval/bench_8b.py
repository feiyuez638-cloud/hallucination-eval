"""本机 Qwen3-8B 可行性压测（2026-10-22）：显存、单题时延、并发下 P50/P99 与吞吐。

  py eval/bench_8b.py --conc 1 4 16 --n 32 [--model qwen3:8b] [--ctx 8192]

- 直连 Ollama 原生 /api/chat，think=False（关思考），temperature 0.6（手册参考实现的值），num_predict 默认 200（--max-out 可改）。
- 两种题：短题取开发集 SimpleQA 题面；长题按手册的三段式拼到约 8 KB（材料取本项目 README，只在本地用）。
- 结果追加到 eval/runs/bench-<ts>.jsonl，终端打印汇总。只测速度，不判对错。
"""
import argparse, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 16])
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--model", default="qwen3:8b")
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--max-out", type=int, default=200)
a = ap.parse_args()

qs = [json.loads(l)["question"] for l in open(ROOT / "data/dev/dev_v1.jsonl", encoding="utf-8")
      if json.loads(l)["source"] != "halluqa"]
mat = (ROOT / "README.md").read_text(encoding="utf-8")


def long_q(i):
    world = mat[(i * 997) % 4000:][:3000]
    talk = "\n".join(f"用户：第 {k} 轮提问。\n助手：第 {k} 轮回答，订单号 {8800 + k}。" for k in range(60))
    return f"【世界设定】\n{world}\n\n【对话记录】\n{talk}\n\n【最终问题】\n{qs[i % len(qs)]}"


cli = httpx.Client(trust_env=False, timeout=300)


def one(prompt):
    t0 = time.time()
    try:
        r = cli.post("http://127.0.0.1:11434/api/chat", json={
            "model": a.model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.6, "num_predict": a.max_out, "num_ctx": a.ctx}})
        r.raise_for_status()
        d = r.json()
        return {"ok": True, "s": round(time.time() - t0, 2), "in": d.get("prompt_eval_count"), "out": d.get("eval_count")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "s": round(time.time() - t0, 2), "err": repr(e)[:150]}


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


one("你好")  # 预热，加载模型
log = ROOT / "eval/runs" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
print(f"模型 {a.model}  num_ctx {a.ctx}  预热后显存 {vram()} MiB")
for kind, make in (("短题", lambda i: qs[i]), ("长题", long_q)):
    print(f"\n{kind}（字符数约 {len(make(0))}）")
    for c in a.conc:
        prompts = [make(i) for i in range(a.n)]
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, prompts))
        wall = time.time() - t0
        ok = sorted(r["s"] for r in res if r["ok"])
        p = lambda q: ok[min(len(ok) - 1, int(len(ok) * q))] if ok else None
        row = {"kind": kind, "conc": c, "n": a.n, "ok": len(ok), "p50": p(0.5), "p99": p(0.99), "max": ok[-1] if ok else None,
               "wall": round(wall, 1), "qps": round(len(ok) / wall, 3), "vram": vram(),
               "in_tok": res[0].get("in"), "fail": [r["err"] for r in res if not r["ok"]][:2]}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  并发 {c:2d}：成功 {len(ok)}/{a.n}  P50 {row['p50']}s  P99 {row['p99']}s  最长 {row['max']}s  "
              f"吞吐 {row['qps']} 题/s  显存 {row['vram']} MiB  输入 {row['in_tok']} tok  {row['fail'] or ''}")
print("\n记录:", log)
