"""analysis/badcase_gallery.py：已提交的画廊就是脚本输出；预期转移与数据不符时脚本拒绝生成。"""
import copy, sys

import pytest

import badcase_gallery as bg
from conftest import ROOT, run


def test_regenerates_committed_file(tmp_path):
    out = tmp_path / "g.md"
    p = run("analysis/badcase_gallery.py", "--out", str(out))
    assert out.read_text(encoding="utf-8") == (ROOT / "analysis/badcase画廊.md").read_text(encoding="utf-8")
    total = sum(len(c["cases"]) for c in bg.CATS)
    assert f"{len(bg.CATS)} 类、{total} 条" in p.stdout
    assert (len(bg.CATS), total) == (9, 39)


def test_every_case_has_cause_and_valid_labels():
    for cat in bg.CATS:
        assert 3 <= len(cat["cases"]) <= 5, cat["key"]
        for c in cat["cases"]:
            la, lb, why = (c[1], c[2], c[3]) if len(c) == 4 else (c[3], c[4], c[5])
            assert {la, lb} <= {"答对", "拒答", "幻觉"} and why.strip()


def test_wrong_expectation_aborts(tmp_path, monkeypatch):
    cats = copy.deepcopy(bg.CATS)
    qid, la, lb, why = cats[0]["cases"][0]
    cats[0]["cases"][0] = (qid, la, "拒答" if lb != "拒答" else "答对", why)  # 故意写错预期的「改进后」标签
    monkeypatch.setattr(bg, "CATS", cats)
    monkeypatch.setattr(sys, "argv", ["badcase_gallery.py", "--out", str(tmp_path / "x.md")])
    with pytest.raises(SystemExit) as e:
        bg.main()
    assert qid in str(e.value) and not (tmp_path / "x.md").exists()


def test_escape_truncates_and_escapes_pipes():
    s = bg.esc("a|b " + "字" * 300, n=10)
    assert "\\|" in s and "全文" in s
