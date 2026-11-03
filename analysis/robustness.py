"""稳健性分析：报告里的版本对比结论，换一种统计方法、换一种口径、去掉一部分题之后还站不站得住。

  python analysis/robustness.py                    # 重新生成 analysis/稳健性分析.md
  python analysis/robustness.py --out x.md         # 写到别处（scripts/check_site_numbers.py 用它核对已提交的文件是不是脚本输出）
  python analysis/robustness.py --json x.json      # 另存机器可读结果（tests/ 用）

- 只读 eval/runs/ 里已有的作答与裁判判定、data/dev/dev_v1.jsonl；不调用任何模型或 API，只用标准库。
- 判定选取、标签口径、McNemar 与 Holm 与 scripts/reproduce_metrics.py 完全相同（直接复用其函数），
  所以基准结果与报告 §4、§7 逐项一致（tests/test_robustness.py 核对）。
- 做了六件事：
  1. 逐版本 HR / 答对率的 bootstrap 95% 区间（按题重抽），与 Wilson 区间并列；
  2. 相邻版本配对：精确 McNemar + Holm（与报告同一个 14 检验族）+ 配对 bootstrap 差值区间与 Bonferroni 同时区间；
  3. 按题型分层：每个题型内部的配对差值、McNemar 与 Holm（每个指标 28 个检验一族），以及四个题型等权平均的效应；
  4. 留一类别：依次去掉 9 个类别（SimpleQA 的 6 个一级类 + HalluQA 的 3 类）之一，重算 14 个检验与 Holm；
  5. 裁判口径：去掉宽松补解析、双标签只算拒答、「资料未提及」改记拒答、「理由只说材料外细节」不算幻觉、
     缺判定按最不利 / 最有利补齐、只比较回答有变化的题，各自重算 14 个检验与 Holm；
  6. 裁判噪声：按已有重测记录估计的翻转率叠加随机翻转，看 Holm 结论保持的比例；以及结论翻转所需的题数（临界题数）。
- 随机数全部用固定种子（每个任务各自一个字符串种子），重跑结果逐字相同。
- 结论分三档，规则写在 verdict() 里；脚本末尾的 EXPECT 写明了每条预期的档位，数据与之不符时直接报错，不会静默生成错的结论。
"""
import argparse, json, math, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reproduce_metrics import (ROOT, RUNS, RUNS_4B, RUNS_8B, GATE_V6, RULES_V7, CITE_ABSENT,  # noqa: E402
                               load, verdicts, lenient, group_of, mcnemar_p, holm, fmt_p, pct, wilson)

OUT = ROOT / "analysis/稳健性分析.md"
SEED, B, NSIM, ALPHA = 0, 10000, 2000, 0.05
PAIRS = [("p0", "V1"), ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("b0-8B", "V4-8B"), ("V4-8B", "V8-8B")]
METRICS = ("hal", "ok")
MNAME = {"hal": "幻觉", "ok": "答对"}
GRPS = ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard")
# 「理由只说材料外细节」：裁判理由提到「未提及 / 无法确认 / 额外」一类说法，且没有提到冲突、错误。关键词规则，粗筛。
NOT_MENTIONED = re.compile(r"未提及|没有提及|未提到|没有提到|未提供|没有提供|未包含|没有包含|无法(从参考材料中?)?(确认|核实|验证|判断)|额外")
CONFLICT = re.compile(r"冲突|矛盾|错|不符|不一致|相悖|相反|不正确|有误")


# ---------- 数据 ----------

def load_data():
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    V, LEN, A, REASON = {}, {}, {}, {}
    for name, stem in {**RUNS_4B, **RUNS_8B}.items():
        f = RUNS / f"judged-{stem}-{'agnes' if name in RUNS_4B else 'gemini-flash-high'}.jsonl"
        V[name], LEN[name] = verdicts(f)
        A[name] = {r["id"]: r for r in load(RUNS / f"{stem}.jsonl")}
        REASON[name] = {}
        for r in load(f):  # 与 verdicts() 同序覆盖：取最后一条有效判定的理由；宽松补解析没有理由字段
            v = r["verdict"] if r.get("parsed_ok") else lenient(r)
            if v and v.get("hallucinated") is not None:
                REASON[name][r["id"]] = r["verdict"].get("reason") if r.get("parsed_ok") else None
    # V6 / V7：用 V4 判定离线回放，触发的题记为「只拒答」（与 reproduce_metrics.replay 相同）
    gate = {g["id"] for g in load(RUNS / GATE_V6) if g["verdict"] == "是"}
    pats = [re.compile(p) for p in RULES_V7.values()]
    rule = {i for i, d in dev.items() if any(p.search(d["question"].strip()) for p in pats)}
    for name, hit in (("V6", gate), ("V7", rule)):
        V[name] = {i: ({"hallucinated": False, "refused": True} if i in hit else x) for i, x in V["V4"].items()}
    return {"dev": dev, "V": V, "LEN": LEN, "A": A, "REASON": REASON}


def hr(v):
    return int(bool(v["hallucinated"])), int(bool(v["refused"]))


def is_ok(h, r):
    return int(not h and not r)


# ---------- 统计工具 ----------

def perc(xs, level=0.95):
    """百分位区间（与 scripts/judge_agreement.py 的取法相同）。"""
    xs = sorted(xs)
    a = (1 - level) / 2
    return xs[int(a * len(xs))], xs[max(int((1 - a) * len(xs)) - 1, 0)]


def resample_counts(rnd, units):
    return Counter(rnd.choices(units, k=len(units)))


def pp(x, sign=True):
    """比例差 → 百分点字符串，避免出现「-0.0」。"""
    v = round(x * 100, 1)
    v = 0.0 if v == 0 else v
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def ci_pp(lo_hi):
    return f"[{pp(lo_hi[0])}, {pp(lo_hi[1])}]"


def ci_pct(lo_hi):
    return f"[{pct(lo_hi[0])}, {pct(lo_hi[1])}]"


def code4(a, b):
    """配对的一道题 → 整数编码：hA + 2rA + 4hB + 8rB。"""
    (ha, ra), (hb, rb) = hr(a), hr(b)
    return ha + 2 * ra + 4 * hb + 8 * rb


def unpack(c):
    return c & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1


def paired_counts(codes):
    """codes: 可迭代的 (编码, 个数)。返回 n、两版幻觉数、答对数、未拒答数与其中幻觉数、两个指标的不一致对。"""
    s = Counter()
    for c, k in codes:
        ha, ra, hb, rb = unpack(c)
        oa, ob = is_ok(ha, ra), is_ok(hb, rb)
        s["n"] += k
        s["ha"] += ha * k
        s["hb"] += hb * k
        s["oa"] += oa * k
        s["ob"] += ob * k
        s["ka"] += (1 - ra) * k
        s["kb"] += (1 - rb) * k
        s["hka"] += ha * (1 - ra) * k
        s["hkb"] += hb * (1 - rb) * k
        s["hal_b"] += (ha and not hb) * k   # 只前一版幻觉
        s["hal_c"] += (hb and not ha) * k   # 只后一版幻觉
        s["ok_b"] += (oa and not ob) * k    # 只前一版答对
        s["ok_c"] += (ob and not oa) * k    # 只后一版答对
    return s


def diffs(s):
    n = s["n"]
    return {"hal": (s["hb"] - s["ha"]) / n, "ok": (s["ob"] - s["oa"]) / n,
            "hr_ex": (s["hkb"] / s["kb"] if s["kb"] else 0) - (s["hka"] / s["ka"] if s["ka"] else 0)}


def mc_p(s, m):
    return mcnemar_p(s[f"{m}_b"], s[f"{m}_c"])


def holm_family(ps):
    """ps: {(对比, 指标): p} → {(对比, 指标): Holm 校正后 p}，键顺序固定为 PAIRS × METRICS。"""
    keys = list(ps)
    return dict(zip(keys, holm([ps[k] for k in keys])))


def pkey(a, b):
    return f"{a}->{b}"


# ---------- 1. 逐版本 ----------

def per_version(D, nboot):
    out = {}
    for name in [*RUNS_4B, "V6", "V7", *RUNS_8B]:
        v = D["V"][name]
        units = [h + 2 * r for h, r in map(hr, v.values())]
        n, h = len(units), sum(u & 1 for u in units)
        rnd, hrs, hexs, oks = random.Random(f"{SEED}:ver:{name}"), [], [], []
        for _ in range(nboot):
            c = resample_counts(rnd, units)
            hh = c[1] + c[3]
            kept, hk = c[0] + c[1], c[1]
            hrs.append(hh / n)
            hexs.append(hk / kept if kept else 0)
            oks.append(c[0] / n)
        kept = sum(1 for u in units if not u & 2)
        out[name] = {"n": n, "hal": h, "hr": h / n, "wilson": wilson(h, n), "boot": perc(hrs),
                     "hr_ex": sum(1 for u in units if u == 1) / kept, "boot_ex": perc(hexs),
                     "ok": sum(1 for u in units if u == 0), "ok_rate": sum(1 for u in units if u == 0) / n, "boot_ok": perc(oks)}
    return out


# ---------- 2. 相邻版本配对 ----------

def pair_codes(D, a, b, ids=None):
    Va, Vb = D["V"][a], D["V"][b]
    common = [i for i in D["dev"] if i in Va and i in Vb and (ids is None or i in ids)]
    return common, [code4(Va[i], Vb[i]) for i in common]


def paired(D, nboot):
    out, ps = {}, {}
    m_sim = len(PAIRS) * len(METRICS)
    for a, b in PAIRS:
        common, codes = pair_codes(D, a, b)
        s = paired_counts(Counter(codes).items())
        d = diffs(s)
        rnd, bs = random.Random(f"{SEED}:pair:{a}->{b}"), {k: [] for k in ("hal", "ok", "hr_ex")}
        for _ in range(nboot):
            dd = diffs(paired_counts(resample_counts(rnd, codes).items()))
            for k in bs:
                bs[k].append(dd[k])
        k = pkey(a, b)
        out[k] = {"n": s["n"], "counts": dict(s), "diff": d,
                  "ci": {m: perc(bs[m]) for m in bs},
                  "sim_ci": {m: perc(bs[m], 1 - ALPHA / m_sim) for m in METRICS}}
        for m in METRICS:
            ps[(k, m)] = mc_p(s, m)
    adj = holm_family(ps)
    for (k, m), p in ps.items():
        out[k].setdefault("p", {})[m] = p
        out[k].setdefault("holm", {})[m] = adj[(k, m)]
    return out


# ---------- 3. 分题型 ----------

def strata(D, nboot):
    out = {}
    for a, b in PAIRS:
        k = pkey(a, b)
        common, codes = pair_codes(D, a, b)
        by_g = {g: [c for i, c in zip(common, codes) if group_of(D["dev"][i]) == g] for g in GRPS}
        rnd = random.Random(f"{SEED}:strata:{k}")
        bs = {g: {m: [] for m in METRICS} for g in (*GRPS, "等权")}
        for _ in range(nboot):
            acc = {m: 0.0 for m in METRICS}
            for g in GRPS:
                dd = diffs(paired_counts(resample_counts(rnd, by_g[g]).items()))
                for m in METRICS:
                    bs[g][m].append(dd[m])
                    acc[m] += dd[m] / len(GRPS)
            for m in METRICS:
                bs["等权"][m].append(acc[m])
        res = {}
        for g in GRPS:
            s = paired_counts(Counter(by_g[g]).items())
            res[g] = {"n": s["n"], "counts": dict(s), "diff": diffs(s), "ci": {m: perc(bs[g][m]) for m in METRICS},
                      "p": {m: mc_p(s, m) for m in METRICS}}
        res["等权"] = {"diff": {m: sum(res[g]["diff"][m] for g in GRPS) / len(GRPS) for m in METRICS},
                     "ci": {m: perc(bs["等权"][m]) for m in METRICS}}
        out[k] = res
    for m in METRICS:  # 每个指标 7 × 4 = 28 个检验一族
        adj = holm_family({(k, g): out[k][g]["p"][m] for k in out for g in GRPS})
        for (k, g), q in adj.items():
            out[k][g].setdefault("holm", {})[m] = q
    return out


# ---------- 4. 留一类别 ----------

def categories(dev):
    sqa = sorted({d["category"] for d in dev.values() if d["source"] == "chinese_simpleqa"})
    return sqa + ["Knowledge", "Misleading", "Misleading-hard"]


def family(D, ids=None, V=None):
    """在给定题集（或给定判定）下重算 14 个检验：返回 {(对比, 指标): {b, c, diff, p, holm}}。"""
    D2 = D if V is None else {**D, "V": V}
    res, ps = {}, {}
    for a, b in PAIRS:
        _, codes = pair_codes(D2, a, b, ids)
        s = paired_counts(Counter(codes).items())
        d = diffs(s)
        for m in METRICS:
            res[(pkey(a, b), m)] = {"b": s[f"{m}_b"], "c": s[f"{m}_c"], "n": s["n"], "diff": d[m]}
            ps[(pkey(a, b), m)] = mc_p(s, m)
    return finish(res, ps)


def finish(res, ps):
    adj = holm_family(ps)
    for k in res:
        res[k]["p"], res[k]["holm"] = ps[k], adj[k]
    return res


def loco(D):
    out = {}
    for c in categories(D["dev"]):
        ids = {i for i, d in D["dev"].items() if d["category"] != c}
        out[c] = family(D, ids)
    return out


# ---------- 5. 裁判口径 ----------

def transform(D, f):
    return {name: {i: y for i, x in v.items() if (y := f(name, i, x)) is not None} for name, v in D["V"].items()}


def variants(D):
    A, LEN, R = D["A"], D["LEN"], D["REASON"]

    def no_lenient(name, i, v):
        return None if i in LEN.get(name, ()) else v

    def both_refuse(name, i, v):
        return {"hallucinated": False, "refused": True} if v["hallucinated"] and v["refused"] else v

    def cite_absent(name, i, v):
        ans = (A.get(name) or A["V4"]).get(i, {}).get("answer") or ""
        return {"hallucinated": False, "refused": True} if v["hallucinated"] and CITE_ABSENT.search(ans) else v

    def extra_detail(name, i, v):
        rs = (R.get(name) or R["V4"]).get(i) or ""
        if v["hallucinated"] and NOT_MENTIONED.search(rs) and not CONFLICT.search(rs):
            return {"hallucinated": False, "refused": v["refused"]}
        return v

    # V6 / V7 是 V4 的回放，口径变换要先作用在 V4 上再回放；它们不在 14 个检验里，这里只影响逐版本表
    out = {"基准": D["V"]}
    for name, f in (("去掉宽松补解析", no_lenient), ("双标签只算拒答", both_refuse),
                    ("「资料未提及」改记拒答", cite_absent), ("理由只说材料外细节不算幻觉", extra_detail)):
        V = transform(D, f)
        for rp in ("V6", "V7"):
            hit = {i for i, x in D["V"][rp].items() if x != D["V"]["V4"][i]}
            V[rp] = {i: ({"hallucinated": False, "refused": True} if i in hit else x) for i, x in V["V4"].items()}
        out[name] = V
    return out


def variant_families(D, Vs):
    out = {name: family(D, V=V) for name, V in Vs.items()}
    # 缺判定按对后一版最不利 / 最有利补齐：前一版缺判定记「答对」/「幻觉」，后一版缺判定记「幻觉」/「答对」
    OK, HAL = {"hallucinated": False, "refused": False}, {"hallucinated": True, "refused": False}
    for name, (fa, fb) in (("缺判定：对后一版最不利", (OK, HAL)), ("缺判定：对后一版最有利", (HAL, OK))):
        res, ps = {}, {}
        for a, b in PAIRS:
            Va, Vb = D["V"][a], D["V"][b]
            codes = [code4(Va.get(i, fa), Vb.get(i, fb)) for i in D["dev"]]
            s = paired_counts(Counter(codes).items())
            d = diffs(s)
            for m in METRICS:
                res[(pkey(a, b), m)] = {"b": s[f"{m}_b"], "c": s[f"{m}_c"], "n": s["n"], "diff": d[m]}
                ps[(pkey(a, b), m)] = mc_p(s, m)
        out[name] = finish(res, ps)
    # 只比较回答有变化的题：两版回答一字不差时，标签不同只可能来自裁判
    res, ps, same = {}, {}, {}
    for a, b in PAIRS:
        common = [i for i in D["dev"] if i in D["V"][a] and i in D["V"][b]]
        eq = {i for i in common if D["A"][a][i].get("answer") == D["A"][b][i].get("answer")}
        _, codes = pair_codes(D, a, b, set(common) - eq)
        s = paired_counts(Counter(codes).items())
        d = diffs(s)
        s_eq = paired_counts(Counter(pair_codes(D, a, b, eq)[1]).items())
        same[pkey(a, b)] = {"n": len(eq), **{f"{m}_{x}": s_eq[f"{m}_{x}"] for m in METRICS for x in "bc"}}
        for m in METRICS:
            res[(pkey(a, b), m)] = {"b": s[f"{m}_b"], "c": s[f"{m}_c"], "n": s["n"], "diff": d[m]}
            ps[(pkey(a, b), m)] = mc_p(s, m)
    out["只比较回答有变化的题"] = finish(res, ps)
    return out, same


# ---------- 6. 裁判噪声 ----------

def retest_flip_rates(dev):
    """同裁判重测里 hallucinated 前后不一致的比例 d；假设每次判定独立地以概率 q 翻转，d = 2q(1-q)。"""
    import judge_agreement as ja
    by = ja.judgments()
    out = {}
    for key, m in (("4B", ja.AGNES), ("8B", ja.FLASH)):
        rt = ja.retest(by, m, dev)
        k = sum(x["v"]["hallucinated"] != y["v"]["hallucinated"] for _, x, y in rt)
        d = k / len(rt)
        out[key] = {"model": m, "pairs": len(rt), "disagree": k, "d": d, "q": (1 - math.sqrt(1 - 2 * d)) / 2}
    return out


def flips(rnd, n, q):
    """以概率 q 独立选中 0..n-1 中的位置（几何跳跃，结果与逐个掷硬币同分布）。"""
    out, j, lq = [], -1, math.log(1 - q)
    while True:
        j += int(math.log(1 - rnd.random()) / lq) + 1
        if j >= n:
            return out
        out.append(j)


def noise_sim(D, rates, base, nsim):
    """在现有判定上再叠加一次与重测同量级的随机翻转（只翻 hallucinated），统计 Holm 结论（p<0.05 与否）保持不变的比例。
    现有判定本身已含裁判噪声，再叠加一次相当于噪声加倍，所以这个比例偏保守。"""
    rnd = random.Random(f"{SEED}:noise")
    data = []
    for a, b in PAIRS:
        common, codes = pair_codes(D, a, b)
        q = rates["8B" if a.endswith("-8B") else "4B"]["q"]
        data.append((pkey(a, b), [list(unpack(c)) for c in codes], q))
    keep = Counter()
    for _ in range(nsim):
        ps = {}
        for k, items, q in data:
            fa, fb = set(flips(rnd, len(items), q)), set(flips(rnd, len(items), q))
            cnt = Counter()
            for j, (ha, ra, hb, rb) in enumerate(items):
                if j in fa:
                    ha = 1 - ha
                if j in fb:
                    hb = 1 - hb
                cnt[ha + 2 * ra + 4 * hb + 8 * rb] += 1
            s = paired_counts(cnt.items())
            for m in METRICS:
                ps[(k, m)] = mc_p(s, m)
        adj = holm_family(ps)
        for key, q in adj.items():
            keep[key] += (q < ALPHA) == (base[key]["holm"] < ALPHA)
    return {key: keep[key] / nsim for key in base}


def tipping(base):
    """其余 13 个 p 值不变时，让这一条的 Holm 结论翻转所需的题数。
    显著的：从多数方向的不一致对里拿掉 k 题（当作裁判误判）后不再显著；不显著的：多数方向再多 k 题才显著。"""
    keys = list(base)
    out = {}
    for key in keys:
        b, c = base[key]["b"], base[key]["c"]
        sig = base[key]["holm"] < ALPHA
        k = 0
        while True:
            k += 1
            nb, nc = (b, c - k) if c >= b else (b - k, c)
            if not sig:
                nb, nc = (b, c + k) if c >= b else (b + k, c)
            if min(nb, nc) < 0:
                out[key] = None
                break
            ps = [mcnemar_p(nb, nc) if kk == key else base[kk]["p"] for kk in keys]
            q = holm(ps)[keys.index(key)]
            if (q < ALPHA) != sig:
                out[key] = k
                break
    return out


# ---------- 结论 ----------

def verdict(key, base, V, L, noise, sim_ci):
    """三档：站得住 / 不确定 / 无显著差异。返回 (档位, 未通过的检查列表)。"""
    if base[key]["p"] >= ALPHA:
        return "无显著差异", []
    sign = 1 if base[key]["diff"] > 0 else -1
    fails = []
    # Holm 后已不显著的，口径与留一改查原始 p，看连「未校正显著」都保不保得住
    lvl, lab = ("holm", "Holm p") if base[key]["holm"] < ALPHA else ("p", "原始 p")
    if lvl == "p":
        fails.append("Holm 校正后不显著")
    lo, hi = sim_ci
    if lo <= 0 <= hi:
        fails.append("Bonferroni 同时区间含 0")
    bad_v = [name for name, f in V.items() if f[key][lvl] >= ALPHA or f[key]["diff"] * sign <= 0]
    if bad_v:
        fails.append(f"{len(bad_v)}/{len(V)} 种口径下{lab} ≥ 0.05（" + "、".join(bad_v) + "）")
    bad_l = [c for c, f in L.items() if f[key][lvl] >= ALPHA or f[key]["diff"] * sign <= 0]
    if bad_l:
        fails.append(f"{len(bad_l)}/{len(L)} 种留一下{lab} ≥ 0.05（去掉 " + "、".join(bad_l) + "）")
    if noise[key] < 0.95:
        fails.append(f"噪声模拟中只有 {pct(noise[key])} 保持原结论")
    return ("站得住" if not fails else "不确定"), fails


# 每条结论的预期档位；数据变了、档位跟着变时脚本报错，提醒同步改写下面的文字说明
EXPECT = {
    ("p0->V1", "hal"): "站得住", ("p0->V1", "ok"): "站得住",
    ("V1->V2", "hal"): "站得住", ("V1->V2", "ok"): "站得住",
    ("V2->V3", "hal"): "站得住", ("V2->V3", "ok"): "站得住",
    ("V3->V4", "hal"): "无显著差异", ("V3->V4", "ok"): "不确定",
    ("V4->V5", "hal"): "站得住", ("V4->V5", "ok"): "无显著差异",
    ("b0-8B->V4-8B", "hal"): "站得住", ("b0-8B->V4-8B", "ok"): "站得住",
    ("V4-8B->V8-8B", "hal"): "不确定", ("V4-8B->V8-8B", "ok"): "无显著差异",
}
CLAIM = {
    ("p0->V1", "hal"): "提示约束让幻觉减少", ("p0->V1", "ok"): "提示约束让答对减少（代价）",
    ("V1->V2", "hal"): "放宽拒答让幻觉增加", ("V1->V2", "ok"): "放宽拒答让答对增加",
    ("V2->V3", "hal"): "检索让幻觉减少", ("V2->V3", "ok"): "检索让答对增加",
    ("V3->V4", "hal"): "重排对幻觉的影响", ("V3->V4", "ok"): "重排让答对增加",
    ("V4->V5", "hal"): "「先挑错」提示让幻觉增加", ("V4->V5", "ok"): "「先挑错」提示对答对的影响",
    ("b0-8B->V4-8B", "hal"): "8B 上 V4 配置让幻觉减少", ("b0-8B->V4-8B", "ok"): "8B 上 V4 配置让答对增加",
    ("V4-8B->V8-8B", "hal"): "窄题型提示让幻觉减少", ("V4-8B->V8-8B", "ok"): "窄题型提示对答对的影响",
}
# 分层结论的预期：(对比, 题型, 指标) -> Holm(28) 后是否显著
EXPECT_STRATA = {
    ("V4->V5", "SimpleQA", "hal"): True, ("V4->V5", "Knowledge", "hal"): False, ("V4->V5", "Misleading", "hal"): False,
    ("V4-8B->V8-8B", "Misleading", "hal"): True, ("V4-8B->V8-8B", "SimpleQA", "hal"): False,
    ("V3->V4", "SimpleQA", "hal"): False, ("V3->V4", "Knowledge", "hal"): False,
    ("V3->V4", "Misleading", "hal"): False, ("V3->V4", "Misleading-hard", "hal"): False,
    ("V2->V3", "SimpleQA", "hal"): True, ("V2->V3", "Misleading", "hal"): False,
}


def fp(p):
    return fmt_p(p, False)


def need(cond, msg):
    """render() 里手写的定性说法要与数据一致，不一致直接报错，而不是生成一段错的文字。"""
    if not cond:
        raise SystemExit(f"文字说明与数据不符，请改写 render()：{msg}")


def excl0(ci):
    return ci[0] > 0 or ci[1] < 0


def render(R):
    P, S, Lc, Vf, same, rates, noise, tip, ver, verd = (R[k] for k in ("paired", "strata", "loco", "variants", "same_answer",
                                                                        "rates", "noise", "tipping", "versions", "verdicts"))
    base = Vf["基准"]
    K = lambda key: f"{key[0].replace('->', ' → ')} {MNAME[key[1]]}"  # noqa: E731
    L = []
    w = L.append
    w("# 稳健性分析：哪些改进在统计上站得住")
    w("")
    w("> 本文件由 `python analysis/robustness.py` 从 `eval/runs/` 里已有的作答与裁判判定生成，**请勿手改**；"
      "`python scripts/check_site_numbers.py` 会重新生成一份并逐字比对。")
    w(f"> 所有数字仍是本地 LLM 裁判的代理指标（口径同报告 §2）。bootstrap 按题重抽 {B} 次、噪声模拟 {NSIM} 次，固定种子，重跑逐字相同；"
      "只用标准库，不调用任何模型或 API。")
    w("> 4B（裁判 agnes-2.5-flash）与 8B（裁判 gemini-3.8-flash-high）两组裁判不同，下文只在组内比较。")
    w("")
    w("## 结论速览")
    w("")
    w("对象是报告 §7 的 14 个配对检验（7 组相邻版本 × 幻觉 / 答对）。判档规则：")
    w("")
    w("- **站得住**：原始 McNemar p<0.05，且同时满足 Holm 校正后 p<0.05、Bonferroni 同时区间不含 0、"
      f"7 种裁判口径与 9 种留一类别下 Holm 仍 p<0.05 且方向不变、裁判噪声模拟中 ≥ 95% 保持结论；")
    w("- **不确定**：原始 p<0.05，但上面至少一条不满足（列在最后一列）；")
    w("- **无显著差异**：原始 p ≥ 0.05。")
    w("")
    w("| 结论 | 对比 | 配对题数 | 差值（百分点）[95% CI] | 原始 p | Holm p | 档位 | 未通过的检查 |")
    w("|---|---|---|---|---|---|---|---|")
    for key in base:
        k, m = key
        tier, fails = verd[key]
        w(f"| {CLAIM[key]} | {K(key)} | {P[k]['n']} | {pp(P[k]['diff'][m])} {ci_pp(P[k]['ci'][m])} | "
          f"{fp(P[k]['p'][m])} | {fp(P[k]['holm'][m])} | **{tier}** | {'；'.join(fails) if fails else '—'} |")
    w("")
    nv, nl = len(Vf) - 1, len(Lc)
    n_rob = sum(t == "站得住" for t, _ in verd.values())
    w(f"**一句话**：14 条里 {n_rob} 条站得住，全部是大改动（提示约束、放宽拒答、检索、「先挑错」的退步、8B 上的 V4 配置）；"
      "报告里两条「校正后不显著」的倾向（重排的答对增加、V8 的幻觉减少）在这里同样只算**不确定**，具体是哪几项检查没过见上表。")
    w("")
    t = P["V3->V4"]
    t8 = P["V4-8B->V8-8B"]
    sa = same["V3->V4"]
    lm = Lc["Misleading"][("V3->V4", "ok")]
    need(excl0(t["ci"]["ok"]) and not excl0(t["sim_ci"]["ok"]), "V3→V4 答对：95% 区间不含 0、同时区间含 0")
    need(sa["ok_b"] + sa["ok_c"] > tip[("V3->V4", "ok")], "V3→V4 答对：回答未变的翻转多于临界题数")
    need(lm["holm"] < ALPHA, "V3→V4 答对：去掉 Misleading 后 Holm 显著")
    w(f"- **V3 → V4 答对（重排）**：+{t['counts']['ok_c'] - t['counts']['ok_b']} 题（只 V3 答对 {t['counts']['ok_b']}、只 V4 答对 {t['counts']['ok_c']}），"
      f"95% 区间 {ci_pp(t['ci']['ok'])} 不含 0，但 14 个检验的同时区间 {ci_pp(t['sim_ci']['ok'])} 含 0；"
      f"Holm 后只差 {tip[('V3->V4', 'ok')]} 题就显著，而这一对里回答一字未改、只是裁判换了判法的「答对」翻转就有 "
      f"{sa['ok_b'] + sa['ok_c']} 题。去掉 Misleading 后 Holm p={fp(lm['holm'])}，说明增益在知识题上、被误导题稀释。"
      "**方向可信，幅度与裁判噪声同一量级**，要靠重复采样或人工复核定论。")
    w(f"- **V3 → V4 幻觉**：只 V3 幻觉 {t['counts']['hal_b']}、只 V4 幻觉 {t['counts']['hal_c']}，差值 {pp(t['diff']['hal'])} 个百分点"
      f" {ci_pp(t['ci']['hal'])}，所有口径、留一与噪声模拟下都不显著：重排对总幻觉没有可检测的影响。")
    best, worst = Vf["缺判定：对后一版最有利"][("V4-8B->V8-8B", "hal")], Vf["缺判定：对后一版最不利"][("V4-8B->V8-8B", "hal")]
    need(best["holm"] < ALPHA <= worst["holm"], "V8 幻觉：缺判定最有利时显著、最不利时不显著")
    w(f"- **V4-8B → V8-8B 幻觉（窄题型提示）**：{pp(t8['diff']['hal'])} 个百分点 {ci_pp(t8['ci']['hal'])}，原始 p={fp(t8['p']['hal'])}、"
      f"Holm p={fp(t8['holm']['hal'])}。它对缺判定很敏感：V4-8B 有 {len(R['dev_ids']) - ver['V4-8B']['n']} 题、V8-8B 有 {len(R['dev_ids']) - ver['V8-8B']['n']} 题没有判定，按对 V8 最有利的方式补齐是 {best['b']}/{best['c']}、"
      f"Holm p={fp(best['holm'])}，按最不利的方式补齐是 {worst['b']}/{worst['c']}、Holm p={fp(worst['holm'])}。"
      "再加上提示是照着同一开发集的错例写的（报告 §7 第 7 条），属于**待留出集确认**。")
    s5 = S["V4->V5"]["SimpleQA"]
    s8 = S["V4-8B->V8-8B"]["Misleading"]
    w(f"- **分题型**：V5 的退步在 SimpleQA 内部也站得住（{pp(s5['diff']['hal'])} 个百分点，Holm {fmt_p(s5['holm']['hal'])}）；"
      f"V8 在 Misleading 上「{s8['counts']['hal_b']} 题修好、{s8['counts']['hal_c']} 题变坏」，28 个分层检验 Holm 校正后仍是 {fmt_p(s8['holm']['hal'])}，"
      "是 V8 唯一显著的分层效应；但正因为提示点名的就是这些题的问法，这个显著性只说明「对写提示时看过的那类题有效」，不能当作泛化证据。")
    w("")

    # 1
    w("## 1. 逐版本幻觉率：bootstrap 区间与 Wilson 区间")
    w("")
    w(f"按题目有放回重抽 {B} 次（每版只抽有判定的题），取 2.5% / 97.5% 分位。两种区间应当接近，接近说明报告用的 Wilson 区间没有被近似方法带偏。"
      "V6 / V7 是用 V4 判定离线回放的（触发题记为「只拒答」）。区间只含抽题误差，不含裁判误判与作答采样的随机性。")
    w("")
    w("| 版本 | 有判定 | 幻觉 | HR（拒答算有效） | Wilson 95% | bootstrap 95% | HR（拒答剔除）[bootstrap 95%] | 答对率 [bootstrap 95%] |")
    w("|---|---|---|---|---|---|---|---|")
    for name, r in ver.items():
        w(f"| {name} | {r['n']} | {r['hal']} | {pct(r['hr'])} | {ci_pct(r['wilson'])} | {ci_pct(r['boot'])} | "
          f"{pct(r['hr_ex'])} {ci_pct(r['boot_ex'])} | {pct(r['ok_rate'])} {ci_pct(r['boot_ok'])} |")
    gap = max(max(abs(r["boot"][0] - r["wilson"][0]), abs(r["boot"][1] - r["wilson"][1])) for r in ver.values())
    w("")
    w(f"两种区间端点最多相差 {pp(gap, False)} 个百分点。")
    w("")

    # 2
    w("## 2. 相邻版本配对检验与 Holm 校正")
    w("")
    w("只用两版都有判定的题。b = 只前一版为真、c = 只后一版为真（「答对」行同理）；p 为精确 McNemar（双侧），"
      "Holm 校正族与报告 §7 相同（这 14 个检验）。差值 = 后一版减前一版、以全部配对题为分母，区间来自配对 bootstrap（按题重抽）；"
      f"同时区间按 Bonferroni 取 {pct(1 - ALPHA / 14)} 分位，14 条同时成立的概率不低于 95%。"
      "「HR（拒答剔除）差值」两边分母不同、含选择效应，只作参照，不做检验。")
    w("")
    w("| 对比 | 指标 | n | b / c | 差值（百分点） | 95% CI | 同时区间 | 原始 p | Holm p | HR（拒答剔除）差值 [95% CI] |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for key in base:
        k, m = key
        r = P[k]
        c = r["counts"]
        ex = f"{pp(r['diff']['hr_ex'])} {ci_pp(r['ci']['hr_ex'])}" if m == "hal" else "—"
        w(f"| {k.replace('->', ' → ')} | {MNAME[m]} | {r['n']} | {c[m + '_b']} / {c[m + '_c']} | {pp(r['diff'][m])} | {ci_pp(r['ci'][m])} | "
          f"{ci_pp(r['sim_ci'][m])} | {fp(r['p'][m])} | {fp(r['holm'][m])} | {ex} |")
    w("")
    need(all((P[k]["p"][m] < ALPHA) == excl0(P[k]["ci"][m]) for k, m in base)
         and all((P[k]["holm"][m] < ALPHA) == excl0(P[k]["sim_ci"][m]) for k, m in base), "bootstrap 区间与 McNemar / Holm 结论一致")
    n_raw = sum(P[k]["p"][m] < ALPHA for k, m in base)
    n_holm = sum(P[k]["holm"][m] < ALPHA for k, m in base)
    w(f"bootstrap 与 McNemar 的结论逐条一致：原始 p<0.05 的 {n_raw} 条，95% 区间都不含 0，其余的都含 0；"
      f"同时区间不含 0 的 {n_holm} 条，正好是 Holm 后仍 p<0.05 的那几条。")
    w("")

    # 3
    w("## 3. 按题型分层的效应")
    w("")
    w("每个题型内部单独配对：差值 [题型内 bootstrap 95% CI]、McNemar 原始 p、Holm p（每个指标 7 组 × 4 题型 = 28 个检验一族）。"
      "「等权平均」把四个题型的差值等权平均，相当于把开发集换成四类题一样多，用来检验「全部」一栏是否被 SimpleQA 300 题的配比主导（报告 §7 第 8 条）；"
      "它的区间来自分层 bootstrap（各题型内部分别重抽）。✓ = Holm 后 p<0.05。")
    for m in METRICS:
        w("")
        w(f"**{MNAME[m]}**（差值单位：百分点；后一版减前一版）")
        w("")
        w("| 对比 | " + " | ".join(f"{g}" for g in GRPS) + " | 等权平均 | 全部（按配比） |")
        w("|---|" + "---|" * (len(GRPS) + 2))
        for k in S:
            cells = []
            for g in GRPS:
                r = S[k][g]
                c = r["counts"]
                mark = " ✓" if r["holm"][m] < ALPHA else ""
                cells.append(f"{pp(r['diff'][m])} {ci_pp(r['ci'][m])}，{c[m + '_b']}/{c[m + '_c']}，{fmt_p(r['p'][m])}，Holm {fp(r['holm'][m])}{mark}（n={r['n']}）")
            e = S[k]["等权"]
            cells.append(f"{pp(e['diff'][m])} {ci_pp(e['ci'][m])}")
            cells.append(f"{pp(P[k]['diff'][m])} {ci_pp(P[k]['ci'][m])}")
            w(f"| {k.replace('->', ' → ')} | " + " | ".join(cells) + " |")
    w("")
    eq_flip = [(k, m) for k in S for m in METRICS if excl0(S[k]["等权"]["ci"][m]) != excl0(P[k]["ci"][m])]
    need(eq_flip == [("V3->V4", "ok"), ("V4-8B->V8-8B", "ok")], "等权与按配比在区间是否含 0 上不同的恰好是 V3→V4 答对、V8 答对")
    need(all(abs(S[k]["等权"]["diff"]["hal"]) < abs(P[k]["diff"]["hal"]) for k in ("V2->V3", "V4->V5", "b0-8B->V4-8B")),
         "等权后检索与 V4-8B 的收益、V5 的退步都变小")
    need(all(excl0(S[k]["等权"]["ci"][m]) == excl0(P[k]["ci"][m]) and S[k]["等权"]["diff"][m] * P[k]["diff"][m] > 0
             for k in S for m in METRICS if P[k]["holm"][m] < ALPHA), "Holm 显著的结论在等权下方向与是否含 0 都不变")
    e34, e8 = S["V3->V4"]["等权"]["ci"]["ok"], S["V4-8B->V8-8B"]["等权"]["ci"]["ok"]
    w("读法：")
    w("")
    w("- Holm 后显著的结论，换成等权平均后方向不变、区间仍不含 0：大结论不是 SimpleQA 的配比造出来的。"
      "幅度上，误导题权重一大，检索与 8B 上 V4 配置的收益、V5 的退步都变小，因为这些效应主要发生在知识题上。")
    w(f"- 对配比敏感的是两条小效应：V3 → V4 答对按配比 {ci_pp(P['V3->V4']['ci']['ok'])}、等权 {ci_pp(e34)}，增益主要在知识题；"
      f"V4-8B → V8-8B 答对按配比 {ci_pp(P['V4-8B->V8-8B']['ci']['ok'])}、等权 {ci_pp(e8)}，增益主要在误导题、被 SimpleQA 配比稀释。"
      "哪一种加权「对」取决于真实流量里误导题占多少，开发集回答不了（报告 §7 第 8 条）。")
    v23 = S["V2->V3"]
    need(v23["SimpleQA"]["holm"]["hal"] < ALPHA <= v23["Misleading"]["holm"]["hal"], "V2→V3：SimpleQA 显著、Misleading 不显著")
    w(f"- 检索（V2 → V3）的幻觉减少主要来自 SimpleQA（{pp(v23['SimpleQA']['diff']['hal'])} 个百分点，Holm 后显著）；"
      f"Misleading 内 {pp(v23['Misleading']['diff']['hal'])} 个百分点、Holm p={fp(v23['Misleading']['holm']['hal'])}，看不出检索对误导题有帮助。")
    v45 = S["V4->V5"]
    w(f"- V5 的退步：SimpleQA {pp(v45['SimpleQA']['diff']['hal'])} 个百分点（Holm 后显著）；Knowledge {pp(v45['Knowledge']['diff']['hal'])}"
      f"（原始 {fmt_p(v45['Knowledge']['p']['hal'])}，Holm p={fp(v45['Knowledge']['holm']['hal'])}，分层校正后不再显著）。")
    need(all(not excl0(S["V3->V4"][g]["ci"]["hal"]) for g in GRPS), "V3→V4 四个题型幻觉区间都含 0")
    w(f"- V3 → V4 四个题型的幻觉差值区间全部含 0，报告「分题型变化都在噪声内」成立。")
    w("")

    # 4
    w("## 4. 留一类别敏感性")
    w("")
    cats = list(Lc)
    w(f"依次去掉 {len(cats)} 个类别之一（SimpleQA 的 6 个一级类各 50 题，HalluQA 三类各 60 / 60 / 30 题），在剩下的题上重算 14 个检验与 Holm。"
      "看两件事：有没有某一类题单独撑起了一个结论；有没有某一类题单独压住了一个结论。")
    w("")
    w("| 对比 | 指标 | 全量 Holm p | 留一后 Holm p 最小 ～ 最大 | 留一后差值范围（百分点） | 改变显著性的类别 |")
    w("|---|---|---|---|---|---|")
    for key in base:
        hs = [Lc[c][key]["holm"] for c in cats]
        ds = [Lc[c][key]["diff"] for c in cats]
        flip = [c for c in cats if (Lc[c][key]["holm"] < ALPHA) != (base[key]["holm"] < ALPHA)]
        w(f"| {key[0].replace('->', ' → ')} | {MNAME[key[1]]} | {fp(base[key]['holm'])} | {fp(min(hs))} ～ {fp(max(hs))} | "
          f"{pp(min(ds))} ～ {pp(max(ds))} | {'、'.join(flip) if flip else '—'} |")
    w("")
    w("<details><summary>完整矩阵：去掉每个类别后的 Holm p</summary>")
    w("")
    w("| 对比 | 指标 | " + " | ".join(cats) + " |")
    w("|---|---|" + "---|" * len(cats))
    for key in base:
        w(f"| {key[0].replace('->', ' → ')} | {MNAME[key[1]]} | " + " | ".join(fp(Lc[c][key]["holm"]) for c in cats) + " |")
    w("")
    w("</details>")
    w("")

    # 5
    w("## 5. 裁判口径变化")
    w("")
    w("裁判只输出两个布尔值，没有可调的分数阈值；能变的是「怎么把裁判输出变成标签」的口径。下面每种口径都对全部版本同时生效，再重算 14 个检验与 Holm：")
    w("")
    w("| 口径 | 做法 | 动机 |")
    w("|---|---|---|")
    w("| 去掉宽松补解析 | 裁判输出不是合法 JSON、靠正则补解析的判定不用 | 补解析可能取错字段（报告 §2） |")
    w("| 双标签只算拒答 | 「拒答 + 幻觉」双标签改为只拒答 | 「我不知道」算不算幻觉，提示词没说清（裁判一致性分析） |")
    w("| 「资料未提及」改记拒答 | 回答命中「资料未提及」句式且被判幻觉的，改为只拒答 | 检索版本的已知伪影（报告 §3.4 第 3 条） |")
    w("| 理由只说材料外细节不算幻觉 | 裁判理由只提到「未提及 / 无法确认 / 额外」、没提冲突或错误的幻觉，改为不幻觉（关键词粗筛） | 更宽的裁判：材料没写的细节不算错（报告 §3.4 第 2 条） |")
    w("| 缺判定：对后一版最不利 / 最有利 | 前一版缺判定记答对、后一版缺判定记幻觉（或反过来），用满 450 题 | 缺判定的题不是随机缺失时的最坏 / 最好情形 |")
    w("| 只比较回答有变化的题 | 两版回答一字不差的题不进配对 | 这些题上的标签差异只可能来自裁判自身不稳 |")
    w("")
    vnames = [n for n in Vf if n != "基准"]
    w("**各口径下的 Holm p**（括号里是 b / c；✗ = 与基准相比显著性改变）")
    w("")
    w("| 对比 | 指标 | 基准 | " + " | ".join(vnames) + " |")
    w("|---|---|---|" + "---|" * len(vnames))
    for key in base:
        cells = []
        for n in vnames:
            f = Vf[n][key]
            mark = " ✗" if (f["holm"] < ALPHA) != (base[key]["holm"] < ALPHA) else ""
            cells.append(f"{fp(f['holm'])}（{f['b']}/{f['c']}）{mark}")
        w(f"| {key[0].replace('->', ' → ')} | {MNAME[key[1]]} | {fp(base[key]['holm'])}（{base[key]['b']}/{base[key]['c']}） | " + " | ".join(cells) + " |")
    w("")
    n_change = sum((Vf[n][key]["holm"] < ALPHA) != (base[key]["holm"] < ALPHA) for n in vnames for key in base)
    w(f"{len(vnames)} 种口径 × 14 个检验中，显著性改变的共 {n_change} 处。")
    w("")
    w("**各口径下的逐版本 HR（拒答算有效）/ 答对数**")
    w("")
    tv = [n for n in R["variant_versions"]]
    w("| 版本 | " + " | ".join(tv) + " |")
    w("|---|" + "---|" * len(tv))
    for name in ver:
        w(f"| {name} | " + " | ".join(f"{pct(R['variant_versions'][n][name]['hr'])} / {R['variant_versions'][n][name]['ok']}" for n in tv) + " |")
    w("")
    rk = R["rank_changes"]
    need(all(x["only_v3v4"] for x in rk), "口径变化只会让 V3 / V4 互换")
    w(f"4B 六个真实作答版本（p0～V5）按 HR（拒答算有效）的排序，在 {len(tv) - 1} 种口径下"
      + ("都不变。" if not rk else "基本不变：只有" + "、".join(f"「{x['variant']}」" for x in rk) + "下 V3 / V4 互换（两者本来就持平）。"))
    w("")

    # 6
    w("## 6. 裁判噪声：结论要翻多少题")
    w("")
    w("已有记录里同一份裁判输入被同一裁判判过两次的样本（`scripts/judge_agreement.py` 的同裁判重测）给出了裁判自身的不稳定程度。"
      "假设每次判定独立地以概率 q 翻转 `hallucinated`，两次不一致的比例 d = 2q(1-q)：")
    w("")
    w("| 组 | 裁判 | 重测对数 | hallucinated 不一致 | 单次翻转率 q |")
    w("|---|---|---|---|---|")
    for g, r in rates.items():
        w(f"| {g} | {r['model']} | {r['pairs']} | {r['disagree']}（{pct(r['d'])}） | {pct(r['q'])} |")
    w("")
    w(f"在现有判定上按 q 再随机翻转一次（两版各自独立），重算 14 个检验与 Holm，重复 {NSIM} 次。现有判定本身已含一次裁判噪声，再叠加一次相当于噪声加倍，所以「保持比例」偏保守。"
      "重测样本偏短答案和 SimpleQA，长回答与误导题上的真实翻转率可能更高。")
    w("")
    w("「临界题数」：其余 13 个 p 值不变，显著的结论要从多数方向的不一致对里拿掉几题（当作裁判误判）才不再显著；不显著的要在多数方向再多几题才显著。"
      "「回答未变的翻转」= 两版回答一字不差、标签却不同的题数（b / c），是这一对里可以直接确认的裁判噪声。")
    w("")
    w("| 对比 | 指标 | b / c | Holm p | 噪声模拟中结论保持 | 临界题数 | 回答未变的翻转 b / c |")
    w("|---|---|---|---|---|---|---|")
    for key in base:
        k, m = key
        sm = same[k]
        tk = tip[key]
        tk_s = "—" if tk is None else (f"拿掉 {tk} 题" if base[key]["holm"] < ALPHA else f"再多 {tk} 题")
        w(f"| {k.replace('->', ' → ')} | {MNAME[m]} | {base[key]['b']} / {base[key]['c']} | {fp(base[key]['holm'])} | {pct(noise[key])} | {tk_s} | "
          f"{sm[m + '_b']} / {sm[m + '_c']}（回答相同 {sm['n']} 题） |")
    w("")

    # 7
    w("## 7. 方法说明与局限")
    w("")
    w("- 这里所有分析都建立在同一批裁判判定上：bootstrap 和留一只反映「抽到哪些题」，口径变化和噪声模拟只覆盖已知的几种裁判问题；"
      "裁判的系统性偏差（比如对某类回答一贯偏严）不会被这些分析发现，需要人工金标准（报告 §8 第 2 条）。")
    w("- 每题只作答一次，作答采样的随机性没有计入；V3 / V4 这类小差异要靠重复采样（报告 §8 第 3 条）。")
    w("- 14 个检验沿用报告的校正族；分层的 28 个检验另成一族。两族都是事后分析，没有事先指定主假设，所以「站得住」的意思是"
      "「在这些检查下都没有翻转」，不是确认性结论；确认要在留出集上事先写定主对比后做一次（报告 §8 第 6 条）。")
    w("- 「理由只说材料外细节」用关键词粗筛裁判理由，宽松补解析的判定没有理由字段、不参与这一口径；"
      "噪声模型假设翻转独立且对称，实际裁判不一致集中在几类边界题上（裁判一致性分析 §3）。")
    w("- 所有结论只在组内成立：4B 与 8B 裁判不同，不能横比。")
    return "\n".join(L) + "\n"


def run(nboot=B, nsim=NSIM):
    D = load_data()
    P = paired(D, nboot)
    Vs = variants(D)
    Vf, same = variant_families(D, Vs)
    base = Vf["基准"]
    for key in base:  # 自检：基准族与 paired() 的 p 值、Holm 完全相同
        assert base[key]["p"] == P[key[0]]["p"][key[1]] and base[key]["holm"] == P[key[0]]["holm"][key[1]], key
    Lc = loco(D)
    rates = retest_flip_rates(D["dev"])
    noise = noise_sim(D, rates, base, nsim)
    tip = tipping(base)
    S = strata(D, nboot)
    ver = per_version(D, nboot)
    vv = {}
    for n, V in Vs.items():
        vv[n] = {}
        for name in ver:
            vs = list(V[name].values())
            vv[n][name] = {"hr": sum(x["hallucinated"] for x in vs) / len(vs),
                           "ok": sum(not x["hallucinated"] and not x["refused"] for x in vs)}
    order = lambda n: sorted(RUNS_4B, key=lambda v: vv[n][v]["hr"])  # noqa: E731
    def swap34(o):
        o = list(o)
        i, j = o.index("V3"), o.index("V4")
        o[i], o[j] = o[j], o[i]
        return o
    rank_changes = [{"variant": n, "order": order(n), "only_v3v4": order(n) == swap34(order("基准"))}
                    for n in vv if n != "基准" and order(n) != order("基准")]
    verd = {key: verdict(key, base, {n: f for n, f in Vf.items() if n != "基准"}, Lc, noise, P[key[0]]["sim_ci"][key[1]])
            for key in base}
    return {"paired": P, "strata": S, "loco": Lc, "variants": Vf, "same_answer": same, "rates": rates, "noise": noise,
            "tipping": tip, "versions": ver, "dev_ids": sorted(D["dev"]), "variant_versions": vv, "rank_changes": rank_changes, "verdicts": verd}


def check_expect(R):
    bad = [f"{k[0]} {MNAME[k[1]]}：预期「{e}」，数据给出「{R['verdicts'][k][0]}」"
           for k, e in EXPECT.items() if R["verdicts"][k][0] != e]
    for (k, g, m), e in EXPECT_STRATA.items():
        got = R["strata"][k][g]["holm"][m] < ALPHA
        if got != e:
            bad.append(f"{k} {g} {MNAME[m]}：预期 Holm 后{'显著' if e else '不显著'}，数据相反")
    return bad


def jsonable(R):
    def conv(x):
        if isinstance(x, dict):
            return {(" ".join(k) if isinstance(k, tuple) else str(k)): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    return conv(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--json")
    a = ap.parse_args()
    R = run()
    bad = check_expect(R)
    if bad:
        sys.exit("结论档位与 EXPECT 不符，请核对数据后同步更新 EXPECT 与 render() 里的文字：\n  " + "\n  ".join(bad))
    Path(a.out).write_text(render(R), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(jsonable(R), ensure_ascii=False, indent=1), encoding="utf-8")
    n = Counter(t for t, _ in R["verdicts"].values())
    out = Path(a.out).resolve()
    print(f"已生成 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else a.out}：14 条配对结论中 "
          + "、".join(f"{k} {v} 条" for k, v in n.items()))


if __name__ == "__main__":
    main()
