"""裁判一致性：同一份裁判输入被不同裁判、或同一裁判多次判定时，标签是否一致。

  python scripts/judge_agreement.py            # 打印各组一致率、Cohen's kappa（bootstrap 95% CI）、混淆矩阵
  python scripts/judge_agreement.py --show     # 另外逐条列出全部不一致样本（问题、回答、双方理由）

- 只读 eval/runs/ 里已有的裁判记录，不调用任何模型或 API，只用标准库。
- 「同一份输入」按记录里的 prompt_sha256 判断：它是 judge.PROMPT 填入问题、参考材料、回答后的哈希，
  哈希相同 = 裁判看到的文字完全相同。不同版本、甚至 4B 和 8B 给出一字不差的回答时，会落到同一个哈希上。
  各判定文件都是独立调用（续跑只跳过同一文件里已判的题，不跨文件复用），所以同哈希的多条判定是真实的重复判定。
- 判定选取与 reproduce_metrics.py 相同：严格解析成功用原判定，否则宽松补解析；都解析不出的不计。
- 跨裁判：每个哈希每个裁判取最早一条有效判定。同裁判重测：每个哈希取该裁判最早的两条有效判定。
- bootstrap 按题目 id 整群重抽（同一题的不同回答不独立），2000 次，固定种子。
- 重叠样本不是随机抽样：只有不同版本给出完全相同回答的题才会重叠，偏向短答案和 SimpleQA，结论不能直接外推到全量。
"""
import argparse, hashlib, random, sys
from collections import Counter, defaultdict

from reproduce_metrics import ROOT, RUNS, group_of, lenient, load, wilson, mcnemar_p, pct, text_of

sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from judge import PROMPT  # noqa: E402
from to_judge import reference  # noqa: E402

AGNES, FLASH, PRO = "agnes-2.5-flash", "gemini-3.8-flash-high", "gemini-3.1-pro-high"
LABELS = ("hallucinated", "refused", "三分类")
B, SEED = 2000, 0


def judgments():
    """所有有效判定：prompt_sha256 -> 裁判 -> [记录…]（按开始时间排序）。"""
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(RUNS.glob("*.jsonl")):
        if not (f.name.startswith("judged-") or f.name.startswith("2026")):
            continue
        for r in load(f):
            v = r["verdict"] if r.get("parsed_ok") else lenient(r)
            if v and v.get("hallucinated") is not None:
                by[r["prompt_sha256"]][r["model"]].append({**r, "v": v, "file": f.name})
    for d in by.values():
        for recs in d.values():
            recs.sort(key=lambda r: (r["started"], r["file"]))
    return by


def lab(v, k):
    if k == "三分类":
        return "幻觉" if v["hallucinated"] else ("拒答" if v["refused"] else "答对")
    return v[k]


def kappa(pairs):
    """pairs: [(x, y)]，返回 (一致率, Cohen's kappa)；期望一致率为 1 时 kappa 无定义，返回 None。"""
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[c] * cy[c] for c in cx) / n / n
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def boot_ci(units):
    """units: [(题目 id, x, y)]，按题目 id 整群重抽。"""
    by_q = defaultdict(list)
    for q, x, y in units:
        by_q[q].append((x, y))
    qs, rnd, ks = list(by_q), random.Random(SEED), []
    for _ in range(B):
        s = [p for q in rnd.choices(qs, k=len(qs)) for p in by_q[q]]
        kk = kappa(s)[1]
        if kk is not None:
            ks.append(kk)
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1], B - len(ks)


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def report(title, rows, na, nb):
    """rows: [(题目 id, 判定 a, 判定 b)]。"""
    print(f"\n## {title}（n={len(rows)}，涉及 {len({q for q, _, _ in rows})} 道题）")
    print(f"| 标签 | 一致 | 一致率 [95% Wilson] | Cohen's κ [bootstrap 95% CI] | PABAK | {na} 判真 | {nb} 判真 | 只 {na} 判真 / 只 {nb} 判真（McNemar） |")
    print("|---|---|---|---|---|---|---|---|")
    for k in LABELS:
        units = [(q, lab(a["v"], k), lab(b["v"], k)) for q, a, b in rows]
        po, kp = kappa([(x, y) for _, x, y in units])
        agree = sum(x == y for _, x, y in units)
        lo, hi = wilson(agree, len(units))
        clo, chi, bad = boot_ci(units)
        if k == "三分类":  # PABAK、判真数、McNemar 只对二值标签有意义
            pabak = nx = ny = bias = "—"
        else:
            pabak = fmt(2 * po - 1)
            nx, ny = sum(x for _, x, _ in units), sum(y for _, _, y in units)
            b_, c_ = sum(x and not y for _, x, y in units), sum(y and not x for _, x, y in units)
            bias = f"{b_} / {c_}（p={fmt(mcnemar_p(b_, c_))}）"
        print(f"| {k} | {agree}/{len(units)} | {pct(po)} [{pct(lo)}, {pct(hi)}] | {fmt(kp)} [{fmt(clo)}, {fmt(chi)}]"
              f"{f'（{bad} 次重抽无定义已跳过）' if bad else ''} | {pabak} | {nx} | {ny} | {bias} |")
    for k in LABELS:
        units = [(lab(a["v"], k), lab(b["v"], k)) for _, a, b in rows]
        cats = ("答对", "拒答", "幻觉") if k == "三分类" else (True, False)
        c = Counter(units)
        print(f"\n混淆矩阵 {k}（行 {na}，列 {nb}）\n\n| | " + " | ".join(map(str, cats)) + " |\n|---|" + "---|" * len(cats))
        for x in cats:
            print(f"| {x} | " + " | ".join(str(c[(x, y)]) for y in cats) + " |")


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def retest(by, m, dev):
    """同裁判重测：同一输入该裁判最早的两次有效判定。"""
    return [(d[m][0]["id"], d[m][0], d[m][1]) for d in by.values() if len(d.get(m, [])) >= 2 and d[m][0]["id"] in dev]


def composition(rows, dev):
    g = Counter(group_of(dev[q]) for q, _, _ in rows)
    print(f"- 题型构成：{dict(g)}；回答长度中位数 {median(len(a['answer_text']) for _, a, _ in rows)} 字")


def show(title, rows, dev, na, nb):
    bad = [(q, a, b) for q, a, b in rows if any(a["v"][k] != b["v"][k] for k in ("hallucinated", "refused"))]
    print(f"\n### 不一致样本：{title}（{len(bad)} 条）")
    for q, a, b in bad:
        d = dev.get(q)
        print(f"\n- **{q}**（{group_of(d) if d else '验收集'}）{a['file']} vs {b['file']}")
        if d:
            print(f"  - 问题：{d['question']}\n  - 参考：{' / '.join(d['gold'])[:150]}")
        print(f"  - 回答：{a['answer_text'][:300]}")
        for name, r in ((na, a), (nb, b)):
            reason = r["v"].get("reason") or text_of(r)[-200:]
            print(f"  - {name}：幻觉={r['v']['hallucinated']} 拒答={r['v']['refused']}；{reason[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    acc = {r["id"]: r for r in load(ROOT / "eval/judge_acceptance.jsonl")}
    by = judgments()

    # 每个哈希对应的回答原文（从作答文件和验收集反查），只用于展示和长度统计
    texts = {}
    for f in sorted(RUNS.glob("answers-*.jsonl")):
        for x in load(f):
            if x.get("error") is None and x["id"] in dev:
                p = PROMPT.format(question=dev[x["id"]]["question"], reference=reference(dev[x["id"]]), answer=x["answer"])
                texts[hashlib.sha256(p.encode("utf-8")).hexdigest()] = x["answer"]
    for x in acc.values():
        p = PROMPT.format(question=x["question"], reference=x["reference"], answer=x["answer"])
        texts[hashlib.sha256(p.encode("utf-8")).hexdigest()] = x["answer"]
    for d in by.values():
        for recs in d.values():
            for r in recs:
                r["answer_text"] = texts.get(r["prompt_sha256"], "（未找到回答原文）")

    print("# 裁判一致性（只读 eval/runs/，不调用任何模型）")
    n_hash = sum(1 for d in by.values() for recs in d.values() for _ in recs)
    print(f"有效判定 {n_hash} 条，不同裁判输入（prompt_sha256）{len(by)} 个；"
          f"回答原文反查成功 {sum(h in texts for h in by)}/{len(by)}")
    dev_all = [h for h, d in by.items() if any(r["id"] in dev for recs in d.values() for r in recs)]
    print(f"- 开发集全部被判过的回答：{len(dev_all)} 个，回答长度中位数 {median(len(texts[h]) for h in dev_all if h in texts)} 字"
          "（对照下面各组重叠样本的长度）")

    inter_dev = [(d[AGNES][0]["id"], d[AGNES][0], d[FLASH][0]) for d in by.values()
                 if AGNES in d and FLASH in d and d[AGNES][0]["id"] in dev]
    report(f"跨裁判：{AGNES} × {FLASH}，开发集", inter_dev, "agnes", "flash")
    composition(inter_dev, dev)
    for m in (AGNES, FLASH):
        rt = retest(by, m, dev)
        report(f"同裁判重测：{m}，开发集（同一输入最早两次判定）", rt, "第 1 次", "第 2 次")
        composition(rt, dev)
        print(f"- 来源文件对：{dict(Counter((x[1]['file'][15:30], x[2]['file'][15:30]) for x in rt).most_common(4))} …")

    # 验收集：三个裁判两两比较，以及各自对人工标注
    print("\n## 验收集（24 条人工标注，构造题，偏易，只作参照）")
    print("| 比较 | n | hallucinated 一致 | refused 一致 | 双标签全一致 |\n|---|---|---|---|---|")
    accv = defaultdict(dict)
    for d in by.values():
        for m, recs in d.items():
            if recs[0]["id"] in acc:
                accv[m][recs[0]["id"]] = recs[0]["v"]
    human = {i: r["expected"] for i, r in acc.items()}
    for x, y in ((AGNES, FLASH), (AGNES, PRO), (FLASH, PRO), ("人工", AGNES), ("人工", FLASH), ("人工", PRO)):
        vx = human if x == "人工" else accv[x]
        common = sorted(set(vx) & set(accv[y]))
        h = sum(vx[i]["hallucinated"] == accv[y][i]["hallucinated"] for i in common)
        r = sum(vx[i]["refused"] == accv[y][i]["refused"] for i in common)
        both = sum((vx[i]["hallucinated"], vx[i]["refused"]) == (accv[y][i]["hallucinated"], accv[y][i]["refused"]) for i in common)
        print(f"| {x} × {y} | {len(common)} | {h}/{len(common)} | {r}/{len(common)} | {both}/{len(common)} |")

    if a.show:
        print("\n# 不一致样本（hallucinated 或 refused 任一不同）")
        show(f"跨裁判 {AGNES} × {FLASH}", inter_dev, dev, "agnes", "flash")
        for m in (AGNES, FLASH):
            show(f"同裁判重测 {m}", retest(by, m, dev), dev, "第 1 次", "第 2 次")


if __name__ == "__main__":
    main()
