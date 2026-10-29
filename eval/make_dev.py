"""练习用开发集 / 留出集（公开数据，不是官方题；官方题型公布后再对齐配比）。

  py eval/make_dev.py

来源 data/external/：Chinese-SimpleQA（3000 条事实短答，ModelScope AI-ModelScope/Chinese-SimpleQA）、
HalluQA（450 条，误导前提 + 知识题，GitHub OpenMOSS/HalluQA）。
固定种子分层抽样：SimpleQA 每个一级类 50 条 = 300；HalluQA Knowledge 60 / Misleading 60 / Misleading-hard 30 = 150。
其余进留出集，只在最后确认时用一次。gold / wrong / url 只给评分和裁判，不进被测模型输入。
"""
import json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data/external"
OUT = ROOT / "data/dev"
SEED = 20261020
HALLU_N = {"Knowledge": 60, "Misleading": 60, "Misleading-hard": 30}
SQA_N = 50


def load():
    rows = []
    for i, l in enumerate((EXT / "chinese_simpleqa/chinese_simpleqa.jsonl").read_text(encoding="utf-8").splitlines()):
        r = json.loads(l)  # 原始 id 有重复，按行号编号
        rows.append({"id": f"sqa-{i:04d}", "source": "chinese_simpleqa", "category": r["primary_category"],
                     "question": r["question"], "gold": [r["answer"]], "wrong": [], "url": r["urls"][:2]})
    for r in json.loads((EXT / "halluqa/HalluQA.json").read_text(encoding="utf-8")):
        pick = lambda p: [v for k, v in r.items() if k.startswith(p) and v]  # noqa: E731
        rows.append({"id": f"hqa-{r['question_id']}", "source": "halluqa", "category": r["Category"],
                     "question": r["Question"], "gold": pick("Best Answer"), "wrong": pick("Wrong_Answer"),
                     "url": [r.get("Source", "")]})
    return rows


def split(rows):
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["category"])].append(r)
    dev, hold = [], []
    for (src, cat), g in sorted(groups.items()):
        n = SQA_N if src == "chinese_simpleqa" else HALLU_N[cat]
        rng.shuffle(g)
        dev += g[:n]
        hold += g[n:]
    return dev, hold


def main():
    rows = load()
    assert len({r["id"] for r in rows}) == len(rows), "id 重复"
    dev, hold = split(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("dev_v1", dev), ("holdout_v1", hold)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part),
                                          encoding="utf-8")
        print(name, len(part))


if __name__ == "__main__":
    main()
