"""把开发集、作答与裁判判定导入 SQLite，供 sql/metrics.sql 查询。只用标准库，不调用任何模型。

  python scripts/build_sqlite.py                  # 生成 sql/hallucination.sqlite（已在 .gitignore 里）
  python scripts/build_sqlite.py --db other.sqlite

表：
- questions  开发集 dev_v1 450 题：题型 qtype（SimpleQA / Knowledge / Misleading / Misleading-hard）、
             v7_rules（V7 关键词规则命中的规则名，正则在 Python 里算好，因为 SQLite 默认没有正则）
- runs       每个作答文件一行：version（报告里的版本名，非正式运行为 NULL）、family（4B/8B）、ord（版本顺序）、
             judge（报告主表用的裁判；4B 是 agnes-2.5-flash，8B 是 gemini-3.8-flash-high）
- answers    作答：每个作答文件 × 题目一行
- judgments  裁判逐条记录（不去重，含失败和重复调用）：hallucinated / refused 为 0/1，解析不出为 NULL；
             parse = strict（严格解析）/ lenient（宽松补解析，规则同 reproduce_metrics.py）/ NULL；
             line_no 是记录在文件里的行号，「同一题后出现的有效判定覆盖前面的」在 SQL 里按它取最后一条
- gate_v6    V6 门控对每道题的判断（是 / 否），用于离线回放
只导入题目全在 dev_v1 里的作答文件（dev_v2 的小样跳过）；验收集记录的 run 为 NULL、qid 是验收集编号。
"""
import argparse, json, re, sqlite3
from pathlib import Path

from reproduce_metrics import GATE_V6, ROOT, RULES_V7, RUNS, RUNS_4B, RUNS_8B, group_of, lenient, load

SCHEMA = """
CREATE TABLE questions (qid TEXT PRIMARY KEY, source TEXT, category TEXT, qtype TEXT, question TEXT, gold TEXT, v7_rules TEXT);
CREATE TABLE runs (run TEXT PRIMARY KEY, version TEXT, family TEXT, ord INTEGER, judge TEXT);
CREATE TABLE answers (run TEXT REFERENCES runs, qid TEXT REFERENCES questions, answer TEXT, error TEXT, latency_s REAL,
                      PRIMARY KEY (run, qid));
CREATE TABLE judgments (jid INTEGER PRIMARY KEY, file TEXT, line_no INTEGER, run TEXT, judge TEXT, qid TEXT,
                        prompt_sha256 TEXT, started TEXT, call_ok INTEGER, parse TEXT, hallucinated INTEGER,
                        refused INTEGER, reason TEXT);
CREATE TABLE gate_v6 (qid TEXT PRIMARY KEY, verdict TEXT);
CREATE INDEX j_run ON judgments (run, judge, qid);
CREATE INDEX j_prompt ON judgments (prompt_sha256);
"""
JUDGE = {"agnes": "agnes-2.5-flash", "gemini-flash-high": "gemini-3.8-flash-high", "gemini-pro": "gemini-3.1-pro-high"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "sql/hallucination.sqlite"))
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    Path(a.db).unlink(missing_ok=True)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)

    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    pat = {k: re.compile(v) for k, v in RULES_V7.items()}
    con.executemany("INSERT INTO questions VALUES (?,?,?,?,?,?,?)", [
        (i, d["source"], d["category"], group_of(d), d["question"], json.dumps(d["gold"], ensure_ascii=False),
         ",".join(k for k, p in pat.items() if p.search(d["question"].strip())) or None) for i, d in dev.items()])

    version = {stem: (v, fam, k, judge) for fam, judge, runs in (("4B", JUDGE["agnes"], RUNS_4B), ("8B", JUDGE["gemini-flash-high"], RUNS_8B))
               for k, (v, stem) in enumerate(runs.items())}
    skipped = []
    for f in sorted(RUNS.glob("answers-*.jsonl")):
        rows = load(f)
        if not all(r["id"] in dev for r in rows):
            skipped.append(f.name)
            continue
        v, fam, k, judge = version.get(f.stem, (None, "8B" if "qwen3-8b" in f.stem else "4B", None, None))
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (f.stem, v, fam, k, judge))
        con.executemany("INSERT INTO answers VALUES (?,?,?,?,?)",
                        [(f.stem, r["id"], r["answer"], r.get("error"), r.get("latency_s")) for r in rows])

    n = 0
    for f in sorted(RUNS.glob("*.jsonl")):
        m = re.fullmatch(r"judged-(answers-.+)-(agnes|gemini-flash-high|gemini-pro)\.jsonl", f.name)
        if not (m or f.name.startswith("2026")):
            continue
        for line_no, r in enumerate(load(f)):
            v = r["verdict"] if r.get("parsed_ok") else lenient(r)
            ok = bool(v) and v.get("hallucinated") is not None
            con.execute("INSERT INTO judgments (file, line_no, run, judge, qid, prompt_sha256, started, call_ok, parse, "
                        "hallucinated, refused, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f.name, line_no, m and m.group(1), r["model"], r["id"], r.get("prompt_sha256"), r.get("started"),
                         bool(r.get("call_ok")), ("strict" if r.get("parsed_ok") else "lenient") if ok else None,
                         v["hallucinated"] if ok else None, v["refused"] if ok else None,
                         (v.get("reason") if ok else None)))
            n += 1
    con.executemany("INSERT INTO gate_v6 VALUES (?,?)", [(g["id"], g["verdict"]) for g in load(RUNS / GATE_V6)])
    con.commit()
    cnt = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("questions", "runs", "answers", "judgments", "gate_v6")}
    print(f"写入 {a.db}：{cnt}")
    print(f"跳过题目不在 dev_v1 里的作答文件 {len(skipped)} 个（dev_v2 小样）")


if __name__ == "__main__":
    main()
