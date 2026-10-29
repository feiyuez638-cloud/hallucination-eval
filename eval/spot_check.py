"""抽出规则判「未匹配」的 SimpleQA 题，供人工逐条复核规则误判率（固定种子，可复现）。

  py eval/spot_check.py eval/runs/answers-xxx.jsonl [--n 40] > 抽查.md
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_rule import ROOT, label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    dev = {r["id"]: r for r in map(json.loads, (ROOT / "data/dev/dev_v1.jsonl").read_text(encoding="utf-8").splitlines())}
    recs = [json.loads(l) for l in Path(a.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [r for r in recs if r["error"] is None and dev[r["id"]]["source"] == "chinese_simpleqa"
            and label(dev[r["id"]], r["answer"]) == "未匹配"]
    random.Random(20261020).shuffle(pool)
    print(f"未匹配共 {len(pool)} 条，抽 {min(a.n, len(pool))} 条\n")
    for r in pool[:a.n]:
        d = dev[r["id"]]
        print(f"### {r['id']}｜{d['category']}\n- 问：{d['question']}\n- 标准答案：{d['gold'][0]}\n"
              f"- 回答：{r['answer'][:300].replace(chr(10), ' ')}\n- 复核：\n")


if __name__ == "__main__":
    main()
