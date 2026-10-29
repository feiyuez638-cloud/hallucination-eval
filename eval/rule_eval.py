"""V7 规则特征检测离线单测 + 用已有 V4 答案回放（2026-10-22，不调任何模型）。

  py eval/rule_eval.py data/dev/dev_v1.jsonl --replay eval/runs/answers-20261021-165805-v4-rerank-qwen3-4b.jsonl

- 规则在看题之前按 V4 误导题幻觉分类写定（以偏概全 / 传说迷信 / 拼诗与假二选一 / 文字陷阱），跑完不回头改。
- 命中任一规则 = 触发，回放时把 V4 答案换成「我不知道」，其余保留。门槛与 V6 相同：
  SimpleQA 触发 ≤ 9/300，且幻觉减少、答对零损失。所有数字都是本地代理指标，非官方成绩。
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from compare_runs import verdicts  # noqa: E402

RULES = {
    "以偏概全": r"(所有|全部|每一?个|任何|凡是)[^，。？]{0,12}(都|皆|全)",
    "传说迷信": r"风水|算命|星座|生肖|属相|鬼魂|闹鬼|神仙|转世|轮回|显灵|托梦|诅咒|招财|辟邪|灵验|预兆|吉凶",
    "假二选一": r"是[^，。？]{1,15}还是[^，。？]{1,15}[？?]?$|到底是[^，。？]{1,15}还是",
    "拼诗": r"(下一句|上一句|上句|下句|出自哪|是谁写的).{0,10}$|[「“\"][^」”\"]{4,}[，,][^」”\"]{4,}[」”\"]",
    "文字陷阱": r"为什么[^，。？]{0,20}(从来不|永远不|一定会|必然)",
}
PAT = {k: re.compile(v) for k, v in RULES.items()}
GROUPS = ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard")

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("--replay")
a = ap.parse_args()
rows = [json.loads(l) for l in open(ROOT / a.inp, encoding="utf-8")]
grp = lambda d: d["category"] if d["source"] == "halluqa" else "SimpleQA"
hits = {d["id"]: [k for k, p in PAT.items() if p.search(d["question"].strip())] for d in rows}

print("触发（命中任一规则）")
for g in GROUPS:
    ids = [d for d in rows if grp(d) == g]
    print(f"  {g:16s} {sum(bool(hits[d['id']]) for d in ids):3d}/{len(ids)}  按规则",
          dict(Counter(k for d in ids for k in hits[d["id"]])))

ok = True
if a.replay:
    v4 = verdicts(a.replay, "agnes")
    lab = lambda v: "幻觉" if v["hallucinated"] else ("拒答" if v["refused"] else "答对")
    tot = Counter()
    print("\n离线回放（V4 判分；触发的题换成「我不知道」）")
    for g in GROUPS:
        c = Counter(lab(v4[d["id"]]) for d in rows if grp(d) == g and hits[d["id"]] and d["id"] in v4)
        tot += c
        print(f"  {g:16s} 拦住幻觉 {c['幻觉']:3d}  误拒答对 {c['答对']:3d}  本来就拒答 {c['拒答']:3d}")
    sq = sum(bool(hits[d["id"]]) for d in rows if grp(d) == "SimpleQA")
    h4 = sum(v["hallucinated"] for v in v4.values())
    print(f"  合计：拦住幻觉 {tot['幻觉']}，误拒答对 {tot['答对']}；V4 幻觉 {h4} → 回放 {h4 - tot['幻觉']}（有判定 {len(v4)} 题）")
    print(f"  门槛一 SimpleQA 触发 {sq}/300 ≤ 9：", "过" if sq <= 9 else "不过")
    print("  门槛二 幻觉减少且答对零损失：", "过" if tot["幻觉"] > 0 and tot["答对"] == 0 else "不过")
    for d in rows:
        if hits[d["id"]] and d["id"] in v4 and lab(v4[d["id"]]) == "答对":
            print(f"    误拒 {d['id']} [{grp(d)}] {hits[d['id']]} {d['question'][:50]}")
