"""judge.py 离线测试：用模拟裁判输出，不调用任何外部模型。  py -m unittest test_judge -v"""
import html, json, re, unittest
from collections import Counter
from pathlib import Path
from judge import agreement, judge_one, parse_verdict, summarize

HERE = Path(__file__).parent

ROW = {"question": "q", "answer": "a", "reference": "r"}


def fake(*outs):
    it = iter(outs)
    def ask(prompt, **kw):
        o = next(it)
        if isinstance(o, Exception):
            raise o
        return o
    return ask


def V(h, r):
    return json.dumps({"hallucinated": h, "refused": r, "reason": "x"})


def run(*outs):
    ask = fake(*outs)
    return summarize([judge_one(ask, ROW) for _ in outs])


class T(unittest.TestCase):
    def test_all_failed(self):
        s = run(OSError("ssl eof"), OSError("ssl eof"))
        self.assertEqual((s["success"], s["failed"], s["coverage"]), (0, 2, 0))
        self.assertIsNone(s["hr_refusal_counted"])
        self.assertIsNone(s["hr_refusal_excluded"])

    def test_all_refused(self):
        s = run(V(False, True), V(False, True))
        self.assertEqual(s["refused"], 2)
        self.assertEqual(s["hr_refusal_counted"], 0)
        self.assertIsNone(s["hr_refusal_excluded"])  # 分母 0，不是 0%

    def test_missing_field(self):
        v = parse_verdict('{"hallucinated": true, "reason": "x"}')
        self.assertIsNone(v["hallucinated"])
        self.assertIn("refused", v["reason"])

    def test_string_bool(self):
        for bad in ('{"hallucinated": "false", "refused": false}', '{"hallucinated": 1, "refused": false}',
                    '{"hallucinated": null, "refused": false}'):
            self.assertIsNone(parse_verdict(bad)["hallucinated"], bad)

    def test_garbage(self):
        self.assertIsNone(parse_verdict("不是 JSON")["hallucinated"])
        self.assertIsNone(parse_verdict("[1,2]")["hallucinated"])

    def test_mixed(self):
        s = run(V(False, False), V(True, False), V(False, True), "```json\n" + V(True, False) + "\n```", OSError("x"))
        self.assertEqual((s["total"], s["success"], s["failed"]), (5, 4, 1))
        self.assertEqual((s["hallucinated"], s["refused"]), (2, 1))
        self.assertAlmostEqual(s["coverage"], 0.8)
        self.assertAlmostEqual(s["hr_refusal_counted"], 2 / 4)
        self.assertAlmostEqual(s["hr_refusal_excluded"], 2 / 3)

    def test_refusal_with_fabrication(self):
        v = parse_verdict(V(True, True))  # 拒答但编造理由：双标签合法
        self.assertEqual((v["hallucinated"], v["refused"]), (True, True))
        # 正常 1 + 幻觉 1 + 纯拒答 1 + 拒答夹带编造 1
        s = run(V(False, False), V(True, False), V(False, True), V(True, True))
        self.assertEqual((s["hallucinated"], s["refused"], s["hallucinated_non_refused"]), (2, 2, 1))
        self.assertAlmostEqual(s["hr_refusal_counted"], 2 / 4)   # 拒答保留：两条幻觉都算
        self.assertAlmostEqual(s["hr_refusal_excluded"], 1 / 2)  # 拒答剔除：夹带编造那条随样本一起剔除

    def test_regression_a4_invalid_json(self):
        """2026-10-01 真实裁判响应：reason 内引号未转义，JSON 无效。必须仍判为解析失败，不自动修补。"""
        rec = next(json.loads(l) for f in sorted((HERE / "eval/runs").glob("*-stage2.jsonl"))
                   for l in f.read_text(encoding="utf-8").splitlines() if json.loads(l)["id"] == "A4")
        content = rec["raw_response"]["choices"][0]["message"]["content"]
        self.assertIn('"hallucinated": false', content)  # 标签文字本身是对的
        v = parse_verdict(content)
        self.assertIsNone(v["hallucinated"])
        self.assertIsNone(v["refused"])
        self.assertIn("无法解析", v["reason"])

    def test_agreement(self):
        rows = [{"id": "a", "expected": {"hallucinated": True, "refused": True}, "hallucinated": True, "refused": True},
                {"id": "b", "expected": {"hallucinated": True, "refused": True}, "hallucinated": True, "refused": False},
                {"id": "c", "expected": {"hallucinated": False, "refused": False}, "hallucinated": None, "refused": None}]
        self.assertEqual(agreement(rows), {"compared": 2, "matched": 1, "mismatched": ["b"]})


import sys
sys.path.insert(0, str(HERE / "eval"))
from build_acceptance import ROWS, text as source_text


def _norm(s):
    return re.sub(r"\s+", "", s)


class AcceptanceSet(unittest.TestCase):
    """裁判验收集：验结构、依据可追溯、依据范围声明属实；不证明参赛模型提分。"""
    rows = [json.loads(l) for l in (HERE / "eval/judge_acceptance.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_file_in_sync_with_builder(self):
        self.assertEqual(self.rows, json.loads(json.dumps(ROWS, ensure_ascii=False)))

    def test_size_and_balance(self):
        cats = Counter(r["category"] for r in self.rows)
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(len({r["id"] for r in self.rows}), 24)
        self.assertEqual((len(cats), set(cats.values())), (6, {4}))

    def test_fields(self):
        for r in self.rows:
            for k in ("id", "category", "question", "answer", "reference", "rationale"):
                self.assertTrue(isinstance(r.get(k), str) and r[k].strip(), (r["id"], k))
            for k in ("hallucinated", "refused"):
                self.assertIs(type(r["expected"][k]), bool, (r["id"], k))
            self.assertTrue(r["evidence"], r["id"])

    def test_labels_match_category(self):
        want = {"正确回答": (False, False), "事实错误": (True, False), "捏造引用": (True, False),
                "错误前提纠正": (False, False), "证据不足": (False, True), "拒答夹带编造": (True, True)}
        for r in self.rows:
            self.assertEqual((r["expected"]["hallucinated"], r["expected"]["refused"]), want[r["category"]], r["id"])

    def test_evidence_traceable(self):
        """引用可追溯：每段依据原样出现在来源文件里，且原样出现在裁判收到的 reference 里。"""
        for r in self.rows:
            for e in r["evidence"]:
                for piece in e.get("parts") or [e["text"]]:
                    self.assertIn(_norm(piece), _norm(source_text(e["source"])), (r["id"], e["scope"]))
                self.assertIn(_norm(e["text"]), _norm(r["reference"]), (r["id"], e["scope"]))

    def test_scope_claims_true(self):
        """范围声明属实：标「全文」的确实是整份来源；absent_* 列出的字串确实不在对应范围里。"""
        for r in self.rows:
            for e in r["evidence"]:
                src = _norm(source_text(e["source"]))
                if e["scope"] == "全文":
                    self.assertEqual(_norm(e["text"]), src, (r["id"], e["source"]))
                for s in e.get("absent_in_source", []):
                    self.assertNotIn(_norm(s), src, (r["id"], s))
                for s in e.get("absent_in_excerpt", []):
                    self.assertNotIn(_norm(s), _norm(e["text"]), (r["id"], s))


class Resume(unittest.TestCase):
    """续跑：不调真实模型，agnes.raw 被替换；记录写进临时目录，不碰 eval/runs/。"""

    def setUp(self):
        import shutil, tempfile, run_judge
        self.rj, self.tmp = run_judge, Path(tempfile.mkdtemp())
        for f in (HERE / "eval/runs").glob("*.jsonl"):
            shutil.copy(f, self.tmp)
        self.old = (run_judge.RUNS, run_judge.agnes.raw, sys.argv)
        run_judge.RUNS = self.tmp
        self.calls = []

    def tearDown(self):
        self.rj.RUNS, self.rj.agnes.raw, sys.argv = self.old

    def _run(self, raw):
        def fake(payload, **kw):
            self.calls.append(kw)
            return raw()
        self.rj.agnes.raw = fake
        sys.argv = ["run_judge.py", "resume"]
        before = set(self.tmp.glob("*.jsonl"))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            self.rj.main()
        new = (set(self.tmp.glob("*.jsonl")) - before).pop()
        return [json.loads(l) for l in new.read_text(encoding="utf-8").splitlines()]

    def test_skips_valid_and_stops_after_two_failures(self):
        def boom():
            raise RuntimeError("HTTP 429")
        recs = self._run(boom)
        self.assertEqual([r["id"] for r in recs], ["A4", "B3"])  # 已有效的 9 条被跳过；连续 2 次失败即停
        self.assertTrue(all(k["retries"] == 1 for k in self.calls))  # 每条只发一次，不重试

    def test_resume_covers_exactly_missing(self):
        ok = {"choices": [{"message": {"content": '{"hallucinated": false, "refused": false, "reason": "x"}'}}]}
        recs = self._run(lambda: ok)
        self.assertEqual(len(recs), 15)
        self.assertNotIn("A1", {r["id"] for r in recs})


if __name__ == "__main__":
    unittest.main()
