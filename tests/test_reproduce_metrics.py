"""scripts/reproduce_metrics.py：统计函数的单元测试 + 报告数字的回归测试（期望值抄自报告表 1～3、§3～§4）。"""
import math

import pytest

import reproduce_metrics as rm

# 报告表 1 / 表 3：版本 -> (有判定, 幻觉, 拒答, 答对, HR(拒答算有效), HR(拒答剔除), 拒答率)
TABLE = {
    "p0": (448, 325, 12, 121, "72.5%", "72.2%", "2.7%"),
    "V1": (449, 87, 295, 72, "19.4%", "53.2%", "65.7%"),
    "V2": (449, 206, 109, 138, "45.9%", "59.4%", "24.3%"),
    "V3": (450, 83, 126, 250, "18.4%", "22.8%", "28.0%"),
    "V4": (448, 83, 106, 267, "18.5%", "21.9%", "23.7%"),
    "V5": (440, 151, 68, 255, "34.3%", "31.5%", "15.5%"),
    "b0-8B": (445, 302, 14, 139, "67.9%", "67.7%", "3.1%"),
    "V4-8B": (438, 129, 31, 283, "29.5%", "30.5%", "7.1%"),
    "V8-8B": (439, 119, 35, 290, "27.1%", "28.2%", "8.0%"),
}
# 报告表 1：HR(拒答算有效) 的 Wilson 95% 区间
CI = {"p0": ("68.2%", "76.5%"), "V4": ("15.2%", "22.4%"), "b0-8B": ("63.4%", "72.0%"), "V8-8B": ("23.2%", "31.5%")}
# 报告 §4.2 / §4.3：配对 -> (幻觉 前→后, 答对 前→后, 幻觉 p, 答对 p)
PAIRS = {
    "V2->V3": ((206, 83), (138, 249), "p<0.001", "p<0.001"),
    "V3->V4": ((83, 83), (248, 267), "p=1.000", "p=0.018"),
    "V4->V5": ((82, 151), (263, 253), "p<0.001", "p=0.212"),
    "b0-8B->V4-8B": ((296, 127), (133, 280), "p<0.001", "p<0.001"),
    "V4-8B->V8-8B": ((129, 117), (277, 286), "p=0.036", "p=0.108"),
}


def table_of(js):
    return {**js["4b"], **js["8b"]}


# ---------- 单元测试 ----------

def test_wilson_matches_report_lower_bounds():
    assert rm.pct(rm.wilson(24, 24)[0]) == "86.2%"
    assert rm.pct(rm.wilson(23, 23)[0]) == "85.7%"
    assert rm.wilson(0, 0) == (None, None)
    lo, hi = rm.wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1 and math.isclose(0.5 - lo, hi - 0.5)


def test_mcnemar_exact():
    assert rm.mcnemar_p(0, 0) == 1.0
    assert rm.mcnemar_p(38, 38) == 1.0
    assert rm.mcnemar_p(10, 0) == pytest.approx(2 / 2 ** 10)
    assert rm.mcnemar_p(3, 7) == rm.mcnemar_p(7, 3)
    assert rm.mcnemar_p(1, 2) == 1.0  # 双侧 p 截断到 1


def test_holm():
    assert rm.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert rm.holm([0.5, 0.9]) == [1.0, 1.0]
    ps = [0.001, 0.2, 0.03, 0.04]
    adj = rm.holm(ps)
    assert all(a >= p for a, p in zip(adj, ps))
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert [adj[i] for i in order] == sorted(adj)  # 按原始 p 排序后单调不减


def test_formatting():
    assert rm.fmt_p(0.0004) == "p<0.001"
    assert rm.fmt_p(0.0183) == "p=0.018"
    assert rm.fmt_p(0.0183, inline=False) == "0.018"
    assert rm.fmt_p(1.0) == "p=1.000"
    assert rm.pct(None) == "—" and rm.pct(0.185) == "18.5%"


def test_stats_double_label_identity():
    vs = [{"hallucinated": True, "refused": True}, {"hallucinated": True, "refused": False},
          {"hallucinated": False, "refused": True}, {"hallucinated": False, "refused": False}]
    s = rm.stats(vs)
    assert (s["n"], s["hal"], s["ref"], s["ok"], s["both"]) == (4, 2, 2, 1, 1)
    assert s["hal"] + s["ref"] + s["ok"] == s["n"] + s["both"]
    assert s["hr_ex"] == 1 / 2  # 拒答（含双标签）从分子分母一起剔除


def test_lenient_parse():
    rec = lambda txt: {"parsed_ok": False, "call_ok": True,  # noqa: E731
                       "raw_response": {"choices": [{"message": {"content": txt}}]}}
    assert rm.lenient(rec('xx "hallucinated": true, "refused": false yy')) == {"hallucinated": True, "refused": False}
    assert rm.lenient(rec('"hallucinated": true "hallucinated": false "refused": false')) is None  # 出现两次不采用
    assert rm.lenient(rec('"hallucinated": true')) is None
    assert rm.lenient({"parsed_ok": True, "call_ok": True}) is None
    assert rm.lenient({"parsed_ok": False, "call_ok": False}) is None


# ---------- 回归测试：报告数字 ----------

def test_dev_set(metrics):
    assert "开发集 dev_v1：450 题， {'SimpleQA': 300, 'Knowledge': 60, 'Misleading': 60, 'Misleading-hard': 30}" in metrics["stdout"]


@pytest.mark.parametrize("ver", TABLE)
def test_table_rows(metrics, ver):
    s = table_of(metrics["json"])[ver]["groups"]["全部"]
    n, h, r, ok, hr, hr_ex, ref = TABLE[ver]
    assert (s["n"], s["hal"], s["ref"], s["ok"]) == (n, h, r, ok)
    assert (rm.pct(s["hr"]), rm.pct(s["hr_ex"]), rm.pct(s["ref_rate"])) == (hr, hr_ex, ref)
    assert s["hal"] + s["ref"] + s["ok"] == s["n"] + s["both"]


@pytest.mark.parametrize("ver", CI)
def test_wilson_ci_rows(metrics, ver):
    s = table_of(metrics["json"])[ver]["groups"]["全部"]
    assert tuple(rm.pct(x) for x in rm.wilson(s["hal"], s["n"])) == CI[ver]


def test_by_type(metrics):
    g = metrics["json"]["4b"]["V4"]["groups"]
    assert [rm.pct(g[k]["hr"]) for k in ("SimpleQA", "Knowledge", "Misleading", "Misleading-hard")] == ["10.0%", "27.1%", "36.7%", "50.0%"]
    g = metrics["json"]["8b"]["V8-8B"]["groups"]
    assert rm.pct(g["Misleading"]["hr"]) == "42.1%"


@pytest.mark.parametrize("pair", PAIRS)
def test_pairs(metrics, pair):
    p = metrics["json"]["pairs"][pair]
    hal, ok, ph, po = PAIRS[pair]
    assert tuple(p["hal"]) == hal and tuple(p["ok"]) == ok
    assert (rm.fmt_p(p["p_hal"]), rm.fmt_p(p["p_ok"])) == (ph, po)


def test_holm_family_of_14(metrics):
    pairs = metrics["json"]["pairs"]
    keys = [(k, m) for k in pairs for m in ("p_hal", "p_ok")]
    adj = dict(zip(keys, rm.holm([pairs[k][m] for k, m in keys])))
    assert len(keys) == 14
    assert rm.fmt_p(adj[("V3->V4", "p_ok")]) == "p=0.092"
    assert rm.fmt_p(adj[("V4-8B->V8-8B", "p_hal")]) == "p=0.143"
    assert rm.fmt_p(adj[("V2->V3", "p_hal")]) == "p<0.001"


def test_replays(metrics):
    v6, v7 = metrics["json"]["v6"], metrics["json"]["v7"]
    assert (v6["blocked"], v6["false_refuse"], v6["sqa_trig"]) == (19, 67, 73)
    assert (v7["blocked"], v7["false_refuse"], v7["sqa_trig"]) == (14, 7, 7)
    assert (rm.pct(v6["after"]["hr"]), rm.pct(v7["after"]["hr"])) == ("14.3%", "15.4%")
    assert v7["false_refuse_by_rule"] == {"拼诗": 6, "传说迷信": 1}


def test_judge_artifact_and_acceptance(metrics):
    art = metrics["json"]["judge_artifact"]
    assert [(art[v]["before"], art[v]["after"]) for v in ("V3", "V4", "V5")] == [(83, 74), (83, 77), (151, 125)]
    a = metrics["json"]["acceptance"]["agnes-2.5-flash"]
    assert a["strict_ok"] + a["lenient_ok"] == a["strict"] + a["lenient"] == 24


def test_retrieval(metrics):
    r = metrics["json"]["retrieval"]
    assert [rm.pct(r["BM25"][k]) for k in ("1", "5", "20")] == ["54.0%", "68.3%", "73.7%"]
    assert rm.pct(r["BM25+重排(N=50)"]["1"]) == "64.0%"
    assert rm.pct(r["BM25+重排(N=50)"]["upper"]) == "76.3%"
    hit = metrics["json"]["retrieval_vs_answer"]["命中"]
    assert (hit["n"], rm.pct(hit["答对"] / hit["n"])) == (205, "85.4%")
