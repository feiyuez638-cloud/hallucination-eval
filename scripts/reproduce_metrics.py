"""一条命令重算报告里的全部表格数字（本地代理指标，非官方成绩）。

  python scripts/reproduce_metrics.py            # 打印所有表
  python scripts/reproduce_metrics.py --json out.json   # 另存机器可读结果（画图脚本用）

- 只读 eval/runs/ 里已有的作答与裁判判定、data/dev/ 开发集、eval/judge_acceptance.jsonl；
  不调用任何模型或付费 API，不需要 GPU，只用标准库。
- 判定选取与 eval/compare_runs.py 相同：严格解析成功用原判定；严格解析失败但调用成功的，
  从裁判原文里各找一次 "hallucinated"/"refused" 布尔值补解析（两个都恰好出现一次才采用）；
  同一题多条记录时，后出现的有效判定覆盖前面的。
- 指标：幻觉率 HR(拒答算有效) = 幻觉 / 有判定；HR(拒答剔除) = 未拒答中的幻觉 / 未拒答；
  答对 = 既不幻觉也不拒答（裁判不单独判正误，「答对」是「作答且未判幻觉」的简称）；拒答率 = 拒答 / 有判定。
  拒答同时编造（双标签）计入幻觉，也计入拒答，所以 幻觉 + 拒答 + 答对 = 有判定 + 双标签。
- 置信区间为 Wilson 95% 区间，只反映抽题误差，不含裁判误差与作答采样随机性。
"""
import argparse, json, math, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval/runs"
GROUPS = ("全部", "SimpleQA", "HalluQA", "Knowledge", "Misleading", "Misleading-hard")

# 版本 -> 作答文件 stem。4B 系列全部用 agnes-2.5-flash 裁判；8B 系列用 gemini-3.8-flash-high 裁判，两组不可横比。
RUNS_4B = {
    "p0": "answers-20261020-131247-p0-qwen3-4b",
    "V1": "answers-20261020-200400-v1-prompt-qwen3-4b",
    "V2": "answers-20261020-215418-v2-prompt-qwen3-4b",
    "V3": "answers-20261021-134415-v3-retrieval-qwen3-4b",
    "V4": "answers-20261021-165805-v4-rerank-qwen3-4b",
    "V5": "answers-20261021-195757-v5-premise-qwen3-4b",
}
RUNS_8B = {
    "b0-8B": "answers-20261022-220103-b0-v1-qwen3-8b",
    "V4-8B": "answers-20261022-225940-v4-v1-qwen3-8b",
    "V8-8B": "answers-20261023-213722-v8-v1-qwen3-8b",
}
RETRIEVAL = {
    "BM25": "retrieval-wiki-bm25-20261021-133722.jsonl",
    "BM25+重排(N=20)": "retrieval-wiki-bm25-rerank20-20261021-163221.jsonl",
    "BM25+重排(N=50)": "retrieval-wiki-bm25-rerank50-20261021-163610.jsonl",
}
GATE_V6 = "gate-20261022-140430.jsonl"
# V7 规则，与 eval/rule_eval.py 逐字相同（离线、确定性）
RULES_V7 = {
    "以偏概全": r"(所有|全部|每一?个|任何|凡是)[^，。？]{0,12}(都|皆|全)",
    "传说迷信": r"风水|算命|星座|生肖|属相|鬼魂|闹鬼|神仙|转世|轮回|显灵|托梦|诅咒|招财|辟邪|灵验|预兆|吉凶",
    "假二选一": r"是[^，。？]{1,15}还是[^，。？]{1,15}[？?]?$|到底是[^，。？]{1,15}还是",
    "拼诗": r"(下一句|上一句|上句|下句|出自哪|是谁写的).{0,10}$|[「“\"][^」”\"]{4,}[，,][^」”\"]{4,}[」”\"]",
    "文字陷阱": r"为什么[^，。？]{0,20}(从来不|永远不|一定会|必然)",
}
PAT = {k: re.compile(rf'"{k}"\s*:\s*(true|false)') for k in ("hallucinated", "refused")}


def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def text_of(rec):
    raw = rec.get("raw_response") or {}
    if "choices" in raw:
        return ((raw["choices"] or [{}])[0].get("message") or {}).get("content") or ""
    return raw.get("response") or ""


def lenient(rec):
    if rec.get("parsed_ok") or not rec.get("call_ok"):
        return None
    got = {k: p.findall(text_of(rec)) for k, p in PAT.items()}
    if all(len(v) == 1 for v in got.values()):
        return {"hallucinated": got["hallucinated"][0] == "true", "refused": got["refused"][0] == "true"}
    return None


def verdicts(judged_path):
    out, n_lenient = {}, set()
    for r in load(judged_path):
        v = r["verdict"] if r.get("parsed_ok") else lenient(r)
        if v and v.get("hallucinated") is not None:
            out[r["id"]] = {"hallucinated": v["hallucinated"], "refused": v["refused"]}
            (n_lenient.discard if r.get("parsed_ok") else n_lenient.add)(r["id"])
    return out, n_lenient


def group_of(d):
    return "SimpleQA" if d["source"] == "chinese_simpleqa" else d["category"]


def in_group(d, g):
    return g == "全部" or g == group_of(d) or (g == "HalluQA" and d["source"] == "halluqa")


def label(v):
    return "幻觉" if v["hallucinated"] else ("拒答" if v["refused"] else "答对")


def stats(vs):
    n = len(vs)
    h = sum(v["hallucinated"] for v in vs)
    r = sum(v["refused"] for v in vs)
    ok = sum(not v["hallucinated"] and not v["refused"] for v in vs)
    kept = [v for v in vs if not v["refused"]]
    hk = sum(v["hallucinated"] for v in kept)
    return {"n": n, "hal": h, "ref": r, "ok": ok, "both": sum(v["hallucinated"] and v["refused"] for v in vs),
            "hr": h / n if n else None, "hr_ex": hk / len(kept) if kept else None,
            "ref_rate": r / n if n else None, "n_kept": len(kept), "hal_kept": hk}


def short_answer(g):
    g = g.strip()
    return bool(re.fullmatch(r"[\d.,]+\s*\S{0,3}", g)) or len(g) <= 2


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return (m - c, m + c)


def mcnemar_p(b, c):
    """精确双侧 McNemar（二项检验），b/c 为两种不一致方向的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fmt_p(p, inline=True):
    """inline：「p<0.001」/「p=0.018」；否则只给数值部分（放表格列里）。"""
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    return ("p" if s[0] == "<" else "p=") + s if inline else s


def holm(ps):
    """Holm 逐步校正，返回与输入同序的校正后 p 值。"""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def run_block(dev, runs):
    res = {}
    for name, stem in runs.items():
        judge = "agnes" if name in RUNS_4B else "gemini-flash-high"
        ans = load(RUNS / f"{stem}.jsonl")
        v, nl = verdicts(RUNS / f"judged-{stem}-{judge}.jsonl")
        res[name] = {"stem": stem, "judge": judge, "answered": len(ans),
                     "answer_fail": sum(a.get("error") is not None for a in ans), "lenient": len(nl),
                     "lenient_ids": sorted(nl), "v": v,
                     "groups": {g: stats([x for i, x in v.items() if in_group(dev[i], g)]) for g in GROUPS}}
        lat = sorted(a["latency_s"] for a in ans if a.get("latency_s") is not None and a.get("error") is None)
        res[name]["lat_p50"] = lat[len(lat) // 2] if lat else None
        # 过度拒答：可答题（SimpleQA + Knowledge）上拒答且没编造
        ansable = [x for i, x in v.items() if group_of(dev[i]) in ("SimpleQA", "Knowledge")]
        res[name]["over_refusal"] = (sum(x["refused"] and not x["hallucinated"] for x in ansable), len(ansable))
    return res


def print_main(title, res):
    print(f"\n## {title}")
    print("| 版本 | 作答 | 有判定 | 宽松补解析 | 幻觉 | 拒答 | 其中双标签 | 答对 | HR(拒答算有效) [95% CI] | HR(拒答剔除) | 拒答率 | 可答题拒答率 | 作答时延 P50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        s = r["groups"]["全部"]
        lo, hi = wilson(s["hal"], s["n"])
        o, on = r["over_refusal"]
        print(f"| {name} | {r['answered'] - r['answer_fail']}/{r['answered']} | {s['n']} | {r['lenient']} | {s['hal']} | "
              f"{s['ref']} | {s['both']} | {s['ok']} | {pct(s['hr'])} [{pct(lo)}, {pct(hi)}] | {pct(s['hr_ex'])} | {pct(s['ref_rate'])} | "
              f"{o}/{on} = {pct(o / on if on else None)} | {r['lat_p50']}s |")


def print_groups(title, res):
    print(f"\n## {title}：分题型（HR 拒答算有效 / HR 拒答剔除 / 答对 / 拒答，n=有判定）")
    print("| 版本 | " + " | ".join(GROUPS[1:]) + " |")
    print("|---|" + "---|" * (len(GROUPS) - 1))
    for name, r in res.items():
        cells = []
        for g in GROUPS[1:]:
            s = r["groups"][g]
            cells.append(f"{pct(s['hr'])} / {pct(s['hr_ex'])} / 对{s['ok']} / 拒{s['ref']} (n={s['n']})")
        print(f"| {name} | " + " | ".join(cells) + " |")


def paired(dev, a, b, na, nb):
    common = sorted(set(a["v"]) & set(b["v"]))
    tr = Counter((label(a["v"][i]), label(b["v"][i])) for i in common)
    ha = sum(a["v"][i]["hallucinated"] for i in common)
    hb = sum(b["v"][i]["hallucinated"] for i in common)
    oa = sum(label(a["v"][i]) == "答对" for i in common)
    ob = sum(label(b["v"][i]) == "答对" for i in common)
    hb_only = sum(not a["v"][i]["hallucinated"] and b["v"][i]["hallucinated"] for i in common)
    ha_only = sum(a["v"][i]["hallucinated"] and not b["v"][i]["hallucinated"] for i in common)
    ob_only = sum(label(a["v"][i]) != "答对" and label(b["v"][i]) == "答对" for i in common)
    oa_only = sum(label(a["v"][i]) == "答对" and label(b["v"][i]) != "答对" for i in common)
    print(f"\n### 配对 {na} → {nb}（两边都有判定 {len(common)} 题）")
    print(f"- 幻觉 {ha} → {hb}（只 {na} 幻觉 {ha_only}，只 {nb} 幻觉 {hb_only}，McNemar {fmt_p(mcnemar_p(ha_only, hb_only))}）")
    print(f"- 答对 {oa} → {ob}（只 {na} 答对 {oa_only}，只 {nb} 答对 {ob_only}，McNemar {fmt_p(mcnemar_p(oa_only, ob_only))}）")
    print("- 转移（行 " + na + "，列 " + nb + "）：")
    labs = ("答对", "拒答", "幻觉")
    print("  | | " + " | ".join(labs) + " |\n  |---|---|---|---|")
    for x in labs:
        print(f"  | {x} | " + " | ".join(str(tr[(x, y)]) for y in labs) + " |")
    return {"common": len(common), "hal": (ha, hb), "ok": (oa, ob), "trans": {f"{x}->{y}": tr[(x, y)] for x in labs for y in labs},
            "p_hal": mcnemar_p(ha_only, hb_only), "p_ok": mcnemar_p(oa_only, ob_only)}


def replay(dev, v4, hit_ids, title):
    """触发的题换成「我不知道」（记为拒答、非幻觉），其余保留 V4 判定。"""
    print(f"\n## {title}")
    print("| 题型 | 触发 | 拦住幻觉 | 误拒答对 | 本来就拒答 |\n|---|---|---|---|---|")
    tot = Counter()
    for g in GROUPS[1:]:
        if g == "HalluQA":
            continue
        ids = [i for i, d in dev.items() if group_of(d) == g]
        trig = [i for i in ids if i in hit_ids]
        c = Counter(label(v4[i]) for i in trig if i in v4)
        tot += c
        print(f"| {g} | {len(trig)}/{len(ids)} | {c['幻觉']} | {c['答对']} | {c['拒答']} |")
    new = {i: ({"hallucinated": False, "refused": True} if i in hit_ids else x) for i, x in v4.items()}
    s0, s1 = stats(list(v4.values())), stats(list(new.values()))
    print(f"- 合计拦住幻觉 {tot['幻觉']}，误拒答对 {tot['答对']}；回放后 幻觉 {s0['hal']} → {s1['hal']}，答对 {s0['ok']} → {s1['ok']}，"
          f"拒答 {s0['ref']} → {s1['ref']}（{pct(s1['ref_rate'])}），HR(拒答算有效) {pct(s0['hr'])} → {pct(s1['hr'])}，HR(拒答剔除) {pct(s0['hr_ex'])} → {pct(s1['hr_ex'])}")
    sq = sum(1 for i, d in dev.items() if group_of(d) == "SimpleQA" and i in hit_ids)
    print(f"- 门槛一 SimpleQA 触发 {sq}/300 ≤ 9：{'过' if sq <= 9 else '不过'}；门槛二 拦住幻觉>0 且误拒答对=0："
          f"{'过' if tot['幻觉'] > 0 and tot['答对'] == 0 else '不过'}")
    return {"blocked": tot["幻觉"], "false_refuse": tot["答对"], "sqa_trig": sq, "after": s1, "before": s0}


REFUSE_WORDS = ("不知道", "不确定", "无法确定", "无法回答", "无法确认", "无法提供", "不清楚", "没有找到", "没有相关", "缺乏",
                "查不到", "不存在", "没有公开", "建议查阅", "无从得知")  # 与 eval/score_rule.py 相同


def norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\s\W_]+", "", s)


def rule_label(d, ans):
    a = norm(ans)
    if any(norm(g) and norm(g) in a for g in d["gold"]):
        return "命中"
    return "拒答用语" if any(w in (ans or "") for w in REFUSE_WORDS) else "未匹配"


def rule_vs_judge(dev, res):
    """SimpleQA 上「标准答案子串匹配」规则与 LLM 裁判的交叉表：两个独立的粗代理互相校验。"""
    print("\n## SimpleQA：规则（标准答案子串匹配）× LLM 裁判")
    print("| 版本 | 规则命中 n | 其中裁判判幻觉 | 规则未匹配 n | 其中裁判判答对 | 规则粗 HR(拒答算有效) | 裁判 HR(拒答算有效) |")
    print("|---|---|---|---|---|---|---|")
    out = {}
    for name, r in res.items():
        ans = {a["id"]: a for a in load(RUNS / f"{r['stem']}.jsonl")}
        sq = [i for i, d in dev.items() if d["source"] == "chinese_simpleqa" and ans.get(i) and ans[i].get("error") is None]
        rl = {i: rule_label(dev[i], ans[i]["answer"]) for i in sq}
        both = [i for i in sq if i in r["v"]]
        hit = [i for i in both if rl[i] == "命中"]
        miss = [i for i in both if rl[i] == "未匹配"]
        h_in_hit = sum(r["v"][i]["hallucinated"] for i in hit)
        ok_in_miss = sum(label(r["v"][i]) == "答对" for i in miss)
        c = Counter(rl.values())
        judge_hr = r["groups"]["SimpleQA"]["hr"]
        out[name] = {"rule": dict(c), "n_sq": len(sq), "hit": len(hit), "hal_in_hit": h_in_hit, "miss": len(miss),
                     "ok_in_miss": ok_in_miss}
        print(f"| {name} | {len(hit)} | {h_in_hit}（{pct(h_in_hit / len(hit))}） | {len(miss)} | {ok_in_miss}（{pct(ok_in_miss / len(miss))}） | "
              f"{c['未匹配']}/{len(sq)} = {pct(c['未匹配'] / len(sq))}（命中 {c['命中']} / 拒答用语 {c['拒答用语']}） | {pct(judge_hr)} |")
    return out


CITE_ABSENT = re.compile(r"(参考)?资料[^。]{0,12}(未|没有|并未|没)[^。]{0,6}(提及|提到|明确|给出|说明|记载|出现|提供|标明)")


def judge_artifact(dev, res):
    """检索版本里模型常说「（参考）资料未提及」——指的是它自己检索到的段落，裁判却按自己的参考材料判成「声称来源没写」= 幻觉。
    敏感性分析：把命中该句式且被判幻觉的题改记为「只拒答」，看结论是否改变。"""
    print("\n## 裁判口径敏感性：「资料未提及」类回答")
    print("| 版本 | 命中句式 | 其中被判幻觉 | 幻觉（原） | 幻觉（改记拒答后） | HR(拒答算有效) 原 → 改 |\n|---|---|---|---|---|---|")
    out = {}
    for name in ("V3", "V4", "V5"):
        r = res[name]
        ans = {a["id"]: a for a in load(RUNS / f"{r['stem']}.jsonl")}
        m = [i for i in r["v"] if CITE_ABSENT.search(ans[i]["answer"] or "")]
        hm = [i for i in m if r["v"][i]["hallucinated"]]
        alt = {i: ({"hallucinated": False, "refused": True} if i in hm else x) for i, x in r["v"].items()}
        s0, s1 = stats(list(r["v"].values())), stats(list(alt.values()))
        out[name] = {"match": len(m), "hal": len(hm), "before": s0["hal"], "after": s1["hal"], "hr_after": s1["hr"]}
        print(f"| {name} | {len(m)} | {len(hm)} | {s0['hal']} | {s1['hal']} | {pct(s0['hr'])} → {pct(s1['hr'])} |")
    return out


def smoke20(dev, v4):
    ids = [r["id"] for r in load(ROOT / "data/dev/smoke20-seed22.jsonl")]
    v5, _ = verdicts(RUNS / "judged-answers-20261021-194409-v5-smoke20-qwen3-4b-agnes.jsonl")
    print("\n## V5 事前 20 题小样（smoke20-seed22，误导 10 + SimpleQA 10）")
    out = {}
    for name, v in (("V4", v4), ("V5", v5)):
        c = Counter(label(v[i]) for i in ids if i in v)
        out[name] = dict(c)
        print(f"- {name}：有判定 {sum(i in v for i in ids)}  幻觉 {c['幻觉']}  拒答 {c['拒答']}  答对 {c['答对']}")
    return out


def acceptance():
    acc = {r["id"]: r for r in load(ROOT / "eval/judge_acceptance.jsonl")}
    by_model = {}
    for f in sorted(RUNS.glob("2026*.jsonl")):
        for r in load(f):
            if r.get("id") not in acc:
                continue
            m = by_model.setdefault(r["model"], {"calls": 0, "call_ok": 0, "strict": {}, "lenient": {}})
            m["calls"] += 1
            m["call_ok"] += bool(r.get("call_ok"))
            v = r.get("verdict")
            if v and v.get("hallucinated") is not None:
                m["strict"][r["id"]] = v
            elif (lv := lenient(r)):
                m["lenient"][r["id"]] = lv
    print("\n## 裁判验收集（24 条，六类各 4，预期双标签人工标注）")
    print("| 裁判 | 调用次数 | 调用成功 | 严格解析有结果 | 严格一致 | 宽松补解析 | 补解析一致 | 无结果 |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    for model, m in by_model.items():
        key = lambda v: (v["hallucinated"], v["refused"])  # noqa: E731
        s_ok = [i for i, v in m["strict"].items() if key(v) == key(acc[i]["expected"])]
        l_only = {i: v for i, v in m["lenient"].items() if i not in m["strict"]}
        l_ok = [i for i, v in l_only.items() if key(v) == key(acc[i]["expected"])]
        missing = sorted(set(acc) - set(m["strict"]) - set(l_only))
        bad = sorted(set(m["strict"]) - set(s_ok)) + sorted(set(l_only) - set(l_ok))
        print(f"| {model} | {m['calls']} | {m['call_ok']} | {len(m['strict'])} | {len(s_ok)}/{len(m['strict'])} | {len(l_only)} | "
              f"{len(l_ok)}/{len(l_only)} | {len(missing)} {missing if missing else ''} |" + (f" 不一致 {bad}" if bad else ""))
        out[model] = {"strict": len(m["strict"]), "strict_ok": len(s_ok), "lenient": len(l_only), "lenient_ok": len(l_ok),
                      "missing": missing, "bad": bad}
    for model, o in out.items():
        k, n = o["strict_ok"] + o["lenient_ok"], o["strict"] + o["lenient"]
        print(f"- {model}：一致 {k}/{n}，一致率 95% Wilson 下界 {pct(wilson(k, n)[0])}")
    cats = Counter(r["category"] for r in acc.values())
    print("- 类别分布：", dict(cats))
    return out


def retrieval(dev):
    print("\n## 检索命中率（SimpleQA 300 题，标准答案字符串出现在前 k 段）")
    print("| 检索 | @1 | @5 | @20 | 送重排的 50 段内含答案 | 出处含维基 @5 | 出处不含维基 @5 | 「其他答案」@5 |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    hit = lambda r, k: r["rank"] is not None and r["rank"] <= k  # noqa: E731
    rate = lambda rs, k: sum(hit(r, k) for r in rs) / len(rs)  # noqa: E731
    for name, f in RETRIEVAL.items():
        rs = [r for r in load(RUNS / f) if r["source"] == "chinese_simpleqa"]
        other = [r for r in rs if not short_answer(dev[r["id"]]["gold"][0])]
        row = {k: rate(rs, k) for k in (1, 5, 20)}
        row.update(wiki5=rate([r for r in rs if r["wiki"]], 5), nonwiki5=rate([r for r in rs if not r["wiki"]], 5),
                   other5=rate(other, 5), n_other=len(other))
        ub = sum(bool(r.get("bm25_rank")) for r in rs) / len(rs) if "bm25_rank" in rs[0] else None
        row["upper"] = ub
        out[name] = row
        print(f"| {name} | {pct(row[1])} | {pct(row[5])} | {pct(row[20])} | {pct(ub) if 'N=50' in name else '—'} | "
              f"{pct(row['wiki5'])} | {pct(row['nonwiki5'])} | {pct(row['other5'])}（n={len(other)}） |")
    print("- 「其他答案」= 去掉以数字为主（正则 ^[\\d.,]+\\s*\\S{0,3}$）或 ≤2 字的短答案，这类答案容易在段落里偶然命中")
    return out


def retrieval_vs_answer(dev, v3):
    """V3 答案的判定按「检索前 5 段是否含标准答案」分组（SimpleQA）。"""
    rs = {r["id"]: r for r in load(RUNS / RETRIEVAL["BM25"]) if r["source"] == "chinese_simpleqa"}
    print("\n## V3：SimpleQA 按检索前 5 段是否命中分组")
    out = {}
    for name, cond in (("命中", True), ("未命中", False)):
        ids = [i for i, r in rs.items() if (r["rank"] is not None and r["rank"] <= 5) == cond and i in v3]
        c = Counter(label(v3[i]) for i in ids)
        out[name] = {"n": len(ids), **c}
        print(f"- {name} {len(ids)} 题：答对 {c['答对']}（{pct(c['答对'] / len(ids))}）  拒答 {c['拒答']}  幻觉 {c['幻觉']}")
    return out


def wiki_split(dev, res):
    """SimpleQA 题目出处是否为维基（retrieval 文件里的 wiki 字段）分开看，检验检索是否偏乐观。"""
    rs = {r["id"]: r["wiki"] for r in load(RUNS / RETRIEVAL["BM25"]) if r["source"] == "chinese_simpleqa"}
    print("\n## SimpleQA 按出处是否含维基链接分组（检索知识库为维基快照）")
    print("| 版本 | 出处含维基：HR / 答对率 (n) | 出处不含维基：HR / 答对率 (n) |\n|---|---|---|")
    out = {}
    for name, r in res.items():
        cells = []
        for w in (True, False):
            vs = [x for i, x in r["v"].items() if i in rs and rs[i] == w]
            s = stats(vs)
            cells.append(f"{pct(s['hr'])} / {pct(s['ok'] / s['n'] if s['n'] else None)} (n={s['n']})")
            out.setdefault(name, {})["wiki" if w else "nonwiki"] = s
        print(f"| {name} | " + " | ".join(cells) + " |")
    n_w = sum(rs.values())
    print(f"- SimpleQA 300 题中出处含维基链接 {n_w} 题，不含 {300 - n_w} 题")
    return out


def smoke_runs(dev):
    """V8～V12 小样：裁判中途失败，覆盖不全，只报覆盖情况。"""
    print("\n## 8B 小样运行覆盖（gemini-3.8-flash-high；多数中途因裁判调用失败停止，不进入对照结论）")
    print("| 运行 | 作答 | 有判定 |\n|---|---|---|")
    for f in sorted(RUNS.glob("judged-answers-*smoke*-gemini-flash-high.jsonl")):
        stem = f.name[len("judged-"):-len("-gemini-flash-high.jsonl")]
        v, _ = verdicts(f)
        print(f"| {stem} | {len(load(RUNS / (stem + '.jsonl')))} | {len(v)} |")


def extra_stats(dev, r4, r8, pairs):
    """外审补充：分题型区间与配对检验、多重比较校正、宽松补解析与缺判定的敏感性。"""
    grps = ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard")
    print("\n## 补充统计（外审新增）")
    print("\n### 分题型 HR(拒答算有效) 的 95% Wilson 区间（幻觉/有判定）")
    print("| 版本 | " + " | ".join(grps) + " |\n|---|" + "---|" * len(grps))
    for name, r in (("V4", r4["V4"]), ("V4-8B", r8["V4-8B"]), ("V8-8B", r8["V8-8B"])):
        cells = []
        for g in grps:
            s = r["groups"][g]
            lo, hi = wilson(s["hal"], s["n"])
            cells.append(f"{pct(s['hr'])} [{pct(lo)}, {pct(hi)}]（{s['hal']}/{s['n']}）")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n### 分题型配对：幻觉的不一致对（只前一版幻觉 / 只后一版幻觉）与 McNemar p")
    print("| 对比 | " + " | ".join(grps) + " |\n|---|" + "---|" * len(grps))
    for na, nb, a, b in (("V3", "V4", r4["V3"], r4["V4"]), ("V4", "V5", r4["V4"], r4["V5"]),
                         ("V4-8B", "V8-8B", r8["V4-8B"], r8["V8-8B"])):
        cells = []
        for g in grps:
            c = [i for i in set(a["v"]) & set(b["v"]) if group_of(dev[i]) == g]
            x = sum(a["v"][i]["hallucinated"] and not b["v"][i]["hallucinated"] for i in c)
            y = sum(not a["v"][i]["hallucinated"] and b["v"][i]["hallucinated"] for i in c)
            cells.append(f"{x} / {y}，{fmt_p(mcnemar_p(x, y))}（n={len(c)}）")
        print(f"| {na} → {nb} | " + " | ".join(cells) + " |")
    keys = [(k, m) for k in pairs for m in ("p_hal", "p_ok")]
    adj = holm([pairs[k][m] for k, m in keys])
    print(f"\n### 多重比较：前文 {len(pairs)} 组配对 × 幻觉/答对，共 {len(keys)} 个 p 值的 Holm 校正")
    print("| 对比 | 指标 | 原始 p | Holm 校正后 p |\n|---|---|---|---|")
    for (k, m), q in zip(keys, adj):
        print(f"| {k} | {'幻觉' if m == 'p_hal' else '答对'} | {fmt_p(pairs[k][m], False)} | {fmt_p(q, False)} |")
    print(f"\n### 敏感性：去掉宽松补解析的判定；缺判定（共 {len(dev)} 题）按最好 / 最坏情形")
    print("| 版本 | HR(拒答算有效) 原 | 去掉宽松补解析 | 缺判定 | 缺判定全算不幻觉 | 缺判定全算幻觉 |\n|---|---|---|---|---|---|")
    for name, r in {**r4, **r8}.items():
        s = r["groups"]["全部"]
        strict = stats([x for i, x in r["v"].items() if i not in set(r["lenient_ids"])])
        miss = len(dev) - s["n"]
        print(f"| {name} | {pct(s['hr'])} | {pct(strict['hr'])}（n={strict['n']}） | {miss} | {pct(s['hal'] / len(dev))} | "
              f"{pct((s['hal'] + miss) / len(dev))} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    print("# 重算结果（只读 eval/runs/，本地 LLM 裁判代理指标，非官方成绩）")
    c = Counter(group_of(d) for d in dev.values())
    print(f"开发集 dev_v1：{len(dev)} 题，", dict(c))

    r4 = run_block(dev, RUNS_4B)
    r8 = run_block(dev, RUNS_8B)
    print_main("Qwen3-4B 各版本（裁判 agnes-2.5-flash）", r4)
    print_groups("Qwen3-4B", r4)
    pairs = {f"{x}->{y}": paired(dev, r4[x], r4[y], x, y) for x, y in (("p0", "V1"), ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"))}
    print_main("Qwen3-8B 全量（裁判 gemini-3.8-flash-high，与上表裁判不同，不可横比）", r8)
    print_groups("Qwen3-8B", r8)
    pairs.update({f"{x}->{y}": paired(dev, r8[x], r8[y], x, y) for x, y in (("b0-8B", "V4-8B"), ("V4-8B", "V8-8B"))})

    gate = {g["id"] for g in load(RUNS / GATE_V6) if g["verdict"] == "是"}
    v6 = replay(dev, r4["V4"]["v"], gate, "V6 门控离线回放（门控判「是」→ 答「我不知道」，用 V4 判定回放）")
    pat = {k: re.compile(v) for k, v in RULES_V7.items()}
    rule_hit = {i for i, d in dev.items() if any(p.search(d["question"].strip()) for p in pat.values())}
    v7 = replay(dev, r4["V4"]["v"], rule_hit, "V7 规则特征离线回放（命中任一规则 → 答「我不知道」，用 V4 判定回放）")
    fr = Counter(k for i, d in dev.items() if i in rule_hit and i in r4["V4"]["v"] and label(r4["V4"]["v"][i]) == "答对"
                 for k, p in pat.items() if p.search(d["question"].strip()))
    print(f"- V7 误拒答对的题按命中规则：{dict(fr)}")
    v7["false_refuse_by_rule"] = dict(fr)
    sm = smoke20(dev, r4["V4"]["v"])
    rvj = rule_vs_judge(dev, r4)
    art = judge_artifact(dev, r4)
    acc = acceptance()
    ret = retrieval(dev)
    rva = retrieval_vs_answer(dev, r4["V3"]["v"])
    ws = wiki_split(dev, {k: r4[k] for k in ("V2", "V3", "V4")})
    smoke_runs(dev)
    extra_stats(dev, r4, r8, pairs)

    if a.json:
        slim = lambda res: {k: {kk: vv for kk, vv in r.items() if kk != "v"} for k, r in res.items()}  # noqa: E731
        Path(a.json).write_text(json.dumps({"4b": slim(r4), "8b": slim(r8), "pairs": pairs, "v6": v6, "v7": v7,
                                            "acceptance": acc, "retrieval": ret, "retrieval_vs_answer": rva,
                                            "wiki_split": ws, "smoke20": sm, "rule_vs_judge": rvj, "judge_artifact": art}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
