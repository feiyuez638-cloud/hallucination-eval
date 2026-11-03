"""scripts/check_site_numbers.py：整体核对通过；各项检查能抓到造出来的错误。"""
import re

import check_site_numbers as cs
from conftest import ROOT, run


def test_full_check_passes():
    p = run("scripts/check_site_numbers.py", check=False)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "全部一致。" in p.stdout
    for name in ("analysis/稳健性分析.md", "docs/en/deep-dive-qa.md", "slides/project-talk.en.md", "docs/复现指南.md"):
        assert name in p.stdout.replace("\\", "/")


def test_facts_keys():
    F = cs.facts()
    assert (F["dev.n"], F["V4.hal"], F["V4.hr"], F["pair.V3->V4.holm_ok"]) == ("450", "83", "18.5%", "p=0.092")
    assert F["sql.n"] == F["sql.ok"]


def test_strict_doc_catches_unsourced_numbers(tmp_path):
    rep = cs.REPORT.read_text(encoding="utf-8")
    nums, flat = set(cs.NUM.findall(rep)), re.sub(r"\s+", "", rep)
    good = tmp_path / "good.md"
    good.write_text("V4 HR 18.5%, retrieval p<0.001; Q7. 3-minute version\n", encoding="utf-8")
    assert cs.strict_doc_issues(good, nums, flat, cs.STRUCT_EN)[0] == []
    bad = tmp_path / "bad.md"
    bad.write_text("HR 18.6% with 98765 items, p=0.0421, see [x](missing.md)\n", encoding="utf-8")
    issues = cs.strict_doc_issues(bad, nums, flat, cs.STRUCT_EN)[0]
    joined = "\n".join(issues)
    assert "98765" in joined and "'18.6%'" in joined and "p=0.0421" in joined and "missing.md" in joined


def test_guide_line_refs(tmp_path):
    src = (ROOT / "scripts/reproduce_metrics.py").read_text(encoding="utf-8").splitlines()
    ln = next(i for i, l in enumerate(src, 1) if l.startswith("def mcnemar_p"))
    doc = tmp_path / "guide.md"
    doc.write_text(f"| a | `scripts/reproduce_metrics.py:{ln}` | `def mcnemar_p` |\n", encoding="utf-8")
    assert cs.guide_issues(doc, set())[0] == []
    doc.write_text(f"| a | `scripts/reproduce_metrics.py:{ln + 1}` | `def mcnemar_p` |\n"
                   f"引用 `scripts/reproduce_metrics.py:{ln}` 没有片段\n", encoding="utf-8")
    issues = cs.guide_issues(doc, set())[0]
    assert len(issues) == 2 and any("不含片段" in x for x in issues) and any("缺少" in x for x in issues)


def test_page_parser_flags_untagged_numbers():
    pg = cs.Page()
    pg.feed('<p>HR <span data-k="V4.hr">18.5%</span>，另有 42 题</p><img alt="图 7" src="x.png"><a href="#nope">a</a>')
    assert pg.tagged[0]["k"] == "V4.hr" and pg.tagged[0]["text"] == "18.5%"
    free = [m.group() for _, s in pg.free for m in cs.NUM.finditer(cs.IDENT.sub("", s))]
    assert "42" in free and "7" in free
    assert ("#nope" in [r for _, r in pg.refs]) and "nope" not in pg.ids


def test_contact_pattern():
    assert cs.CONTACT.search("联系 someone@example.com")
    assert cs.CONTACT.search("电话 13800138000")
    assert not cs.CONTACT.search("答对 267 题，p=0.018")
