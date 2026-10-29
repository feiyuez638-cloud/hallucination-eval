"""本地幻觉率评分器（模型无关，官方细则下发前的替身）。

输出的 HR 是**本地模型裁判的代理指标**，不等于主办方评分；官方判定方式与拒答口径均未公开。

输入 jsonl：每行 {"id", "question", "answer", "reference"(可空)}
输出：逐条判定 jsonl + 汇总（成功/失败/覆盖率/拒答数/两种口径 HR）。

  py judge.py answers.jsonl [--judge agnes|deepseek] [--out judged.jsonl]
"""
import argparse, json, re, sys

sys.path.insert(0, r"C:\Users\Honerable\Workspace\agent-desk")

PROMPT = """你是事实核查员，只依据【参考材料】判断，不要用你自己的记忆补充、确认或否定任何事实。
幻觉（hallucinated=true）：回答中有与参考材料冲突的内容，或声称某来源写了某内容、而材料显示该来源（已完整给出的部分）并没有写。
拒答（refused=true）：回答明确表示无法回答、不确定或材料不足。拒答的同时若仍有上述幻觉内容，两个标签都为 true。
【问题】{question}
【参考材料】
{reference}
【回答】{answer}
只输出一行 JSON：{{"hallucinated": true/false, "refused": true/false, "reason": "一句话"}}"""


def _fail(reason):
    return {"hallucinated": None, "refused": None, "reason": reason}


def parse_verdict(out):
    """裁判原始输出 -> 判定；字段缺失、类型不是布尔记为失败。幻觉与拒答可同时为真（拒答但编造理由）。"""
    m = re.search(r"\{.*\}", out or "", re.S)
    try:
        v = json.loads(m.group(0))
    except Exception:
        return _fail("裁判输出无法解析: " + str(out)[:200])
    if not isinstance(v, dict):
        return _fail("裁判输出不是对象: " + str(out)[:200])
    for k in ("hallucinated", "refused"):
        if type(v.get(k)) is not bool:  # 拒绝 "true"/"false" 字符串、0/1、null、缺失
            return _fail(f"字段 {k} 缺失或不是布尔: {v.get(k)!r}")
    return {"hallucinated": v["hallucinated"], "refused": v["refused"], "reason": str(v.get("reason", ""))}


def judge_one(ask, row):
    try:
        out = ask(PROMPT.format(question=row["question"], reference=row.get("reference") or "（未提供参考材料）",
                                answer=row["answer"]), max_tokens=6000, temperature=0)
    except Exception as e:  # 网络断连等：记为失败，不让整批崩
        return _fail("裁判调用失败: " + repr(e)[:200])
    return parse_verdict(out)


def summarize(res):
    """分母为 0 时 HR 返回 None（不可计算），不伪装成 0%。
    拒答剔除口径：分子分母都去掉拒答样本，拒答夹带编造的那条幻觉也随之剔除。"""
    ok = [v for v in res if v["hallucinated"] is not None]
    hal = sum(v["hallucinated"] for v in ok)
    ref = sum(v["refused"] for v in ok)
    n = len(ok)
    kept = [v for v in ok if not v["refused"]]
    hal_kept = sum(v["hallucinated"] for v in kept)
    return {
        "total": len(res), "success": n, "failed": len(res) - n,
        "coverage": n / len(res) if res else None,
        "hallucinated": hal, "refused": ref,
        "hr_refusal_counted": hal / n if n else None,
        "hallucinated_non_refused": hal_kept,
        "hr_refusal_excluded": hal_kept / len(kept) if kept else None,
    }


def agreement(res):
    """行内带 expected（裁判验收集）时，比对双标签；裁判失败的条目不计入。"""
    cmp = [v for v in res if "expected" in v and v["hallucinated"] is not None]
    bad = [str(v.get("id")) for v in cmp
           if (v["hallucinated"], v["refused"]) != (v["expected"]["hallucinated"], v["expected"]["refused"])]
    return {"compared": len(cmp), "matched": len(cmp) - len(bad), "mismatched": bad}


def _pct(x):
    return "不可计算" if x is None else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--judge", default="agnes")
    ap.add_argument("--out", default="judged.jsonl")
    a = ap.parse_args()
    ask = __import__(a.judge).ask
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    res = []
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            v = {**r, **judge_one(ask, r)}
            res.append(v)
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    s = summarize(res)
    print("【代理指标：本地模型裁判，非官方评分】")
    print(f"总数 {s['total']}  成功 {s['success']}  失败 {s['failed']}  覆盖率 {_pct(s['coverage'])}")
    print(f"幻觉 {s['hallucinated']}  拒答 {s['refused']}")
    print(f"HR(拒答算有效) = {_pct(s['hr_refusal_counted'])}")
    print(f"HR(拒答剔除)   = {_pct(s['hr_refusal_excluded'])}")
    a = agreement(res)
    if a["compared"]:
        print(f"与预期标签一致 {a['matched']}/{a['compared']}  不一致: {', '.join(a['mismatched']) or '无'}")


if __name__ == "__main__":
    main()
