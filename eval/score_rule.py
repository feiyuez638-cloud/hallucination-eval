"""规则粗评（SimpleQA 部分）+ 时延统计。只是粗代理：措辞不同的正确回答会被记成「未匹配」，要抽查校准。

  py eval/score_rule.py eval/runs/answers-xxx.jsonl [--dev data/dev/dev_v1.jsonl]

- 命中：标准答案归一化后是回答归一化后的子串。
- 拒答：未命中且回答含拒答用语。
- 未匹配：其余，粗算为幻觉。HalluQA 题只统计时延与拒答用语，正误留给裁判。
"""
import argparse, json, re, statistics, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFUSE = ("不知道", "不确定", "无法确定", "无法回答", "无法确认", "无法提供", "不清楚", "没有找到", "没有相关", "缺乏", "查不到",
          "不存在", "没有公开", "建议查阅", "无从得知")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\s\W_]+", "", s)


def label(row, ans):
    a = norm(ans)
    if any(norm(g) and norm(g) in a for g in row["gold"]):
        return "命中"
    if any(w in (ans or "") for w in REFUSE):
        return "拒答"
    return "未匹配"


def pct(x, n):
    return f"{x}/{n} = {x / n:.1%}" if n else "不可计算"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--dev", default=str(ROOT / "data/dev/dev_v1.jsonl"))
    a = ap.parse_args()
    dev = {r["id"]: r for r in map(json.loads, Path(a.dev).read_text(encoding="utf-8").splitlines())}
    recs = [json.loads(l) for l in Path(a.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if r["error"] is None]
    lat = sorted(r["latency_s"] for r in ok)
    print("【规则粗评：本地代理指标，非官方成绩】")
    print(f"作答 {len(recs)}  成功 {len(ok)}  失败 {len(recs) - len(ok)}")
    if lat:
        print(f"时延 P50 {statistics.median(lat):.1f}s  P95 {lat[int(len(lat) * .95) - 1]:.1f}s  最大 {lat[-1]:.1f}s"
              f"  超 45s {sum(x > 45 for x in lat)} 条")
    print("截断(length):", sum(r.get("finish_reason") == "length" for r in ok))
    sqa = [r for r in ok if dev[r["id"]]["source"] == "chinese_simpleqa"]
    c = Counter(label(dev[r["id"]], r["answer"]) for r in sqa)
    n = len(sqa)
    print(f"\nSimpleQA {n} 题：命中 {c['命中']}  拒答 {c['拒答']}  未匹配 {c['未匹配']}")
    print("粗 HR(拒答算有效) =", pct(c["未匹配"], n))
    print("粗 HR(拒答剔除)   =", pct(c["未匹配"], n - c["拒答"]))
    by = Counter((dev[r["id"]]["category"], label(dev[r["id"]], r["answer"])) for r in sqa)
    for cat in sorted({k[0] for k in by}):
        t = sum(v for k, v in by.items() if k[0] == cat)
        print(f"  {cat}: 命中 {by[(cat, '命中')]}/{t}  拒答 {by[(cat, '拒答')]}  未匹配 {by[(cat, '未匹配')]}")
    hq = [r for r in ok if dev[r["id"]]["source"] == "halluqa"]
    ref = Counter(dev[r["id"]]["category"] for r in hq if any(w in (r["answer"] or "") for w in REFUSE))
    tot = Counter(dev[r["id"]]["category"] for r in hq)
    print(f"\nHalluQA {len(hq)} 题（正误待裁判）；含拒答用语：" + "  ".join(f"{k} {ref[k]}/{v}" for k, v in sorted(tot.items())))


if __name__ == "__main__":
    main()
