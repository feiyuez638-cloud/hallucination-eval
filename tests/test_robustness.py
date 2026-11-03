"""analysis/robustness.py：统计工具的单元测试、与 reproduce_metrics.py 的一致性、结论档位的回归测试。"""
import random

import pytest

import reproduce_metrics as rm
import robustness as rb
from conftest import ROOT


# ---------- 单元测试 ----------

def test_perc_and_pp():
    xs = list(range(1, 101))
    assert rb.perc(xs) == (3, 97)
    assert rb.pp(-0.00001) == "+0.0" and rb.pp(0.0423) == "+4.2" and rb.pp(-0.0283) == "-2.8"
    assert rb.excl0((0.01, 0.2)) and rb.excl0((-0.2, -0.01)) and not rb.excl0((-0.01, 0.2))


def test_code_roundtrip_and_counts():
    vs = [{"hallucinated": h, "refused": r} for h in (False, True) for r in (False, True)]
    for a in vs:
        for b in vs:
            assert rb.unpack(rb.code4(a, b)) == (*rb.hr(a), *rb.hr(b))
    # 一题 只前一版幻觉、一题 只后一版答对
    s = rb.paired_counts([(rb.code4(vs[2], vs[0]), 1), (rb.code4(vs[1], vs[0]), 1)])
    assert (s["n"], s["hal_b"], s["hal_c"], s["ok_b"], s["ok_c"]) == (2, 1, 0, 0, 2)


def test_flips_rate_and_determinism():
    a = rb.flips(random.Random(1), 100000, 0.03)
    assert a == rb.flips(random.Random(1), 100000, 0.03)
    assert 2700 < len(a) < 3300 and a == sorted(set(a)) and a[-1] < 100000


def test_tipping_toy():
    base = {("x", "hal"): {"b": 0, "c": 10, "p": rm.mcnemar_p(0, 10), "holm": rm.mcnemar_p(0, 10)}}
    # 单个检验：0/10 显著（p≈0.002），拿掉 k 题直到 0/(10-k) 不显著：0/5 时 p=0.0625
    assert rb.tipping(base) == {("x", "hal"): 5}


def test_deterministic_small_run():
    a, b = rb.run(nboot=200, nsim=20), rb.run(nboot=200, nsim=20)
    assert rb.jsonable(a) == rb.jsonable(b)


# ---------- 与 reproduce_metrics.py 一致 ----------

def test_base_tests_match_reproduce_metrics(metrics, robust):
    pairs, P = metrics["json"]["pairs"], robust["json"]["paired"]
    keys = [(k, m) for k in pairs for m in ("p_hal", "p_ok")]
    adj = dict(zip(keys, rm.holm([pairs[k][m] for k, m in keys])))
    assert set(P) == set(pairs)
    for k, m in keys:
        mm = "hal" if m == "p_hal" else "ok"
        assert P[k]["p"][mm] == pairs[k][m]
        assert P[k]["holm"][mm] == adj[(k, m)]
        assert P[k]["n"] == pairs[k]["common"]


def test_versions_match_table(metrics, robust):
    tab = {**metrics["json"]["4b"], **metrics["json"]["8b"]}
    for v, s in tab.items():
        r = robust["json"]["versions"][v]
        g = s["groups"]["全部"]
        assert (r["n"], r["hal"], r["ok"]) == (g["n"], g["hal"], g["ok"])
        assert tuple(r["wilson"]) == rm.wilson(g["hal"], g["n"])
        lo, hi = r["boot"]
        assert lo <= r["hr"] <= hi and abs(lo - r["wilson"][0]) < 0.01 and abs(hi - r["wilson"][1]) < 0.01
    assert (robust["json"]["versions"]["V6"]["hal"], robust["json"]["versions"]["V7"]["hal"]) == (64, 69)


def test_strata_partition_overall(robust):
    P, S = robust["json"]["paired"], robust["json"]["strata"]
    for k in P:
        for key in ("hal_b", "hal_c", "ok_b", "ok_c", "n"):
            assert sum(S[k][g]["counts"][key] for g in rb.GRPS) == P[k]["counts"][key]


def test_loco_drops_one_category(robust):
    full = robust["json"]["variants"]["基准"]
    for c, fam in robust["json"]["loco"].items():
        for key, r in fam.items():
            assert r["n"] < full[key]["n"]
    assert list(robust["json"]["loco"])[-3:] == ["Knowledge", "Misleading", "Misleading-hard"]


# ---------- 回归测试：结论 ----------

def test_verdicts_match_expect(robust):
    got = {k: v[0] for k, v in robust["json"]["verdicts"].items()}
    assert got == {" ".join(k): v for k, v in rb.EXPECT.items()}
    assert sum(v == "站得住" for v in got.values()) == 9


def test_key_numbers(robust):
    P = robust["json"]["paired"]
    assert rb.ci_pp(P["V3->V4"]["ci"]["ok"]) == "[+0.9, +7.6]"
    assert rb.ci_pp(P["V3->V4"]["sim_ci"]["ok"]) == "[-0.7, +9.4]"
    assert robust["json"]["tipping"]["V3->V4 ok"] == 2
    assert robust["json"]["tipping"]["V2->V3 hal"] == 96
    sa = robust["json"]["same_answer"]["V3->V4"]
    assert sa["ok_b"] + sa["ok_c"] == 12 and sa["hal_b"] + sa["hal_c"] == 13
    r = robust["json"]["rates"]
    assert (r["4B"]["pairs"], r["4B"]["disagree"], r["8B"]["pairs"], r["8B"]["disagree"]) == (423, 24, 417, 2)
    assert rm.pct(robust["json"]["noise"]["V3->V4 ok"]) == "83.2%"
    best = robust["json"]["variants"]["缺判定：对后一版最有利"]["V4-8B->V8-8B hal"]
    assert (best["b"], best["c"], rm.fmt_p(best["holm"])) == (30, 8, "p=0.002")
    assert rm.fmt_p(robust["json"]["strata"]["V4-8B->V8-8B"]["Misleading"]["holm"]["hal"]) == "p=0.033"


def test_committed_markdown_is_script_output(robust):
    assert robust["md"] == (ROOT / "analysis/稳健性分析.md").read_text(encoding="utf-8")
    assert "-0.0" not in robust["md"]


def test_expect_guard(monkeypatch):
    R = {"verdicts": {k: ("站得住", []) for k in rb.EXPECT}, "strata": {}}
    monkeypatch.setattr(rb, "EXPECT_STRATA", {})
    bad = rb.check_expect(R)
    assert len(bad) == sum(v != "站得住" for v in rb.EXPECT.values()) and bad
