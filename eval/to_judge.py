"""把作答记录转成 judge.py 的输入（question / answer / reference），参考材料只给裁判。

  py eval/to_judge.py eval/runs/answers-xxx.jsonl [--source halluqa] > eval/runs/judge-in-xxx.jsonl
  py judge.py eval/runs/judge-in-xxx.jsonl --judge <裁判> --out eval/runs/judged-xxx.jsonl   # 调用前须越越同意裁判与额度

SimpleQA：参考 = 标准答案 + 出处链接。HalluQA：参考 = 数据集给的正确回答 + 出处；错误回答样例不给，免得裁判照抄。
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def reference(d):
    if d["source"] == "chinese_simpleqa":
        return f"标准答案：{d['gold'][0]}\n出处：{'；'.join(d['url'])}"
    best = "\n".join(f"- {g}" for g in d["gold"])
    return f"正确回答（任一即可，措辞不必一致）：\n{best}\n出处：{d['url'][0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--source", choices=["chinese_simpleqa", "halluqa"])
    a = ap.parse_args()
    dev = {r["id"]: r for r in map(json.loads, (ROOT / "data/dev/dev_v1.jsonl").read_text(encoding="utf-8").splitlines())}
    for l in Path(a.runs).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        d = dev[r["id"]]
        if r["error"] is None and (a.source is None or d["source"] == a.source):
            sys.stdout.write(json.dumps({"id": r["id"], "question": d["question"], "answer": r["answer"],
                                         "reference": reference(d)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
