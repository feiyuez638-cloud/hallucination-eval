"""跑一遍 sql/metrics.sql，与 reproduce_metrics.py 的输出逐项比对；不一致时列出并以非零状态退出。

  python scripts/check_sql.py            # 重建 SQLite → 执行 SQL → 比对
  python scripts/check_sql.py --print    # 另外把每段 SQL 的结果打印成表

- 先调用 build_sqlite.py 重建库（写到临时目录，不动 sql/hallucination.sqlite），再用 reproduce_metrics.py --json 取 Python 结果。
- 比对项：表 1/表 3 每个版本的各列、表 2 每个版本 × 题型、V6/V7 回放、V3→V4 转移表、相邻版本配对计数与 McNemar p；
  另外把 a6_cross_judge 的混淆矩阵与 judge_agreement.py 比对。计数要求完全相等，比率允许 1e-9 的浮点误差。
- 只用标准库，不调用任何模型。
"""
import argparse, json, re, sqlite3, subprocess, sys, tempfile
from pathlib import Path

import judge_agreement as ja
from reproduce_metrics import ROOT, mcnemar_p

SCRIPTS = ROOT / "scripts"


def blocks(sql):
    """按「-- name: 名字」切段，返回 [(名字, SQL)]。"""
    parts = re.split(r"^-- name: (\S+)\s*$", sql, flags=re.M)
    return list(zip(parts[1::2], parts[2::2]))


def show(name, cur):
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    f = lambda c, x: "—" if x is None else (f"{x:.1f}" if c.endswith(("_pct", "_pp")) else str(x))  # noqa: E731
    print(f"\n### {name}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f(c, x) for c, x in zip(cols, r)) + " |")
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    db, js = tmp / "check.sqlite", tmp / "py.json"
    subprocess.run([sys.executable, str(SCRIPTS / "build_sqlite.py"), "--db", str(db)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(SCRIPTS / "reproduce_metrics.py"), "--json", str(js)], check=True, stdout=subprocess.DEVNULL)
    py = json.loads(js.read_text(encoding="utf-8"))
    runs = {**py["4b"], **py["8b"]}

    con = sqlite3.connect(db)
    res = {}
    for name, sql in blocks((ROOT / "sql/metrics.sql").read_text(encoding="utf-8")):
        if name == "setup":
            con.executescript(sql)
            continue
        cur = con.execute(sql)
        if a.print:
            cols, rows = show(name, cur)
        else:
            cols, rows = [c[0] for c in cur.description], cur.fetchall()
        res[name] = [dict(zip(cols, r)) for r in rows]

    n, bad = 0, []

    def eq(what, s, p, ratio=False):
        nonlocal n
        n += 1
        same = (s is None and p is None) or (s is not None and p is not None and abs(s - p) < 1e-9) if ratio else s == p
        if not same:
            bad.append(f"{what}: SQL={s!r} Python={p!r}")

    for r in res["main_table"]:
        v, g = runs[r["version"]], runs[r["version"]]["groups"]["全部"]
        w = f"main/{r['version']}"
        for sk, pv in (("answered", v["answered"]), ("answer_fail", v["answer_fail"]), ("n", g["n"]), ("lenient", v["lenient"]),
                       ("hal", g["hal"]), ("ref", g["ref"]), ("both_labels", g["both"]), ("ok", g["ok"]),
                       ("over_ref", v["over_refusal"][0]), ("answerable", v["over_refusal"][1]), ("lat_p50", v["lat_p50"])):
            eq(f"{w}/{sk}", r[sk], pv)
        for sk, pk in (("hr_pct", "hr"), ("hr_ex_pct", "hr_ex"), ("ref_rate_pct", "ref_rate")):
            eq(f"{w}/{sk}", r[sk] / 100, g[pk], ratio=True)
    eq("main/版本数", len(res["main_table"]), len(runs))

    for r in res["by_type"]:
        g = runs[r["version"]]["groups"][r["grp"]]
        w = f"by_type/{r['version']}/{r['grp']}"
        for k in ("n", "hal", "ref", "ok"):
            eq(f"{w}/{k}", r[k], g[k])
        eq(f"{w}/hr", r["hr_pct"] / 100, g["hr"], ratio=True)
        eq(f"{w}/hr_ex", None if r["hr_ex_pct"] is None else r["hr_ex_pct"] / 100, g["hr_ex"], ratio=True)
    eq("by_type/行数", len(res["by_type"]), sum(len(v["groups"]) for v in runs.values()))

    for r in res["replay_v6_v7"]:
        p = py["v6" if r["version"] == "V6*" else "v7"]
        w = f"replay/{r['version']}"
        eq(f"{w}/blocked", r["blocked_hal"], p["blocked"])
        eq(f"{w}/false_refuse", r["false_refuse"], p["false_refuse"])
        for k in ("n", "hal", "ref", "ok"):
            eq(f"{w}/{k}", r[k], p["after"][k])
        for sk, pk in (("hr_pct", "hr"), ("hr_ex_pct", "hr_ex"), ("ref_rate_pct", "ref_rate")):
            eq(f"{w}/{sk}", r[sk] / 100, p["after"][pk], ratio=True)

    tr = py["pairs"]["V3->V4"]["trans"]
    for r in res["transition_v3_v4"]:
        for y in ("答对", "拒答", "幻觉"):
            eq(f"transition/{r['v3']}->{y}", r[f"v4_{y}"], tr[f"{r['v3']}->{y}"])
    eq("transition/格数", 3 * len(res["transition_v3_v4"]), len(tr))

    for r in res["paired_adjacent"]:
        p = py["pairs"][f"{r['a']}->{r['b']}"]
        w = f"paired/{r['a']}->{r['b']}"
        eq(f"{w}/common", r["common"], p["common"])
        eq(f"{w}/hal", [r["hal_a"], r["hal_b"]], p["hal"])
        eq(f"{w}/ok", [r["ok_a"], r["ok_b"]], p["ok"])
        eq(f"{w}/p_hal", mcnemar_p(r["only_a_hal"], r["only_b_hal"]), p["p_hal"], ratio=True)
        eq(f"{w}/p_ok", mcnemar_p(r["only_a_ok"], r["only_b_ok"]), p["p_ok"], ratio=True)
    eq("paired/对数", len(res["paired_adjacent"]), len(py["pairs"]))

    by = ja.judgments()
    dev = {q for (q,) in con.execute("SELECT qid FROM questions")}
    inter = [(d[ja.AGNES][0]["v"]["hallucinated"], d[ja.FLASH][0]["v"]["hallucinated"]) for d in by.values()
             if ja.AGNES in d and ja.FLASH in d and d[ja.AGNES][0]["id"] in dev]
    for r in res["a6_cross_judge"]:
        eq(f"a6_cross_judge/agnes={r['agnes_hal']},flash={r['flash_hal']}", r["n"],
           sum((x, y) == (bool(r["agnes_hal"]), bool(r["flash_hal"])) for x, y in inter))
    eq("a6_cross_judge/总数", sum(r["n"] for r in res["a6_cross_judge"]), len(inter))

    print(f"\nSQL 与 Python 比对：共 {n} 项，一致 {n - len(bad)} 项，不一致 {len(bad)} 项")
    for b in bad:
        print("  不一致：", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
