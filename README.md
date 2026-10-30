# 中文大模型幻觉评测：Qwen3-4B / 8B 的提示约束、检索与拦截

在 450 题中文开发集（Chinese SimpleQA + HalluQA）上，用 LLM 裁判逐版评测 Qwen3-4B / 8B 的幻觉率：从原样作答的 p0 出发，逐版对比提示约束、维基检索、重排、前提核查、门控和规则拦截（V1～V7），并分析哪些改动有效、为什么。

📄 **完整报告：[report/大模型幻觉评测报告.md](report/大模型幻觉评测报告.md)**

## 关键结果

Qwen3-4B，dev_v1 450 题，裁判 agnes-2.5-flash。所有数字都是本地 LLM 裁判的代理指标（单裁判、单次采样、开发集），由脚本从已有判定结果重算。「答对」= 作答且未判幻觉（裁判不单独判正误）；幻觉与拒答有少量重叠（拒答同时编造，两项都计）。

| 版本 | 改动 | 幻觉 | 拒答 | 答对 | HR（拒答算有效） | HR（拒答剔除） |
|---|---|---|---|---|---|---|
| p0 | 原样作答 | 325 | 12 | 121 | 72.5% | 72.2% |
| V1 | 系统提示约束 | 87 | 295 | 72 | 19.4% | 53.2% |
| V2 | 放宽拒答 | 206 | 109 | 138 | 45.9% | 59.4% |
| V3 | + 维基 BM25 检索 | 83 | 126 | 250 | 18.4% | 22.8% |
| **V4** | **+ bge 重排（基线）** | **83** | **106** | **267** | **18.5%** | **21.9%** |
| V5 | 「先查问题有没有错」提示 | 151 | 68 | 255 | 34.3% | 31.5% |
| V6 | 模型自判门控（离线回放） | 64 | 189 | 200 | 14.3% | 22.8% |
| V7 | 关键词规则拦截（离线回放） | 69 | 127 | 260 | 15.4% | 19.0% |

- 检索（V3）是相邻版本里唯一让幻觉减少、答对增加且都显著的一步（配对 p<0.001）；重排（V4）幻觉持平、答对 +19（未校正 p=0.018，多重比较校正后不显著）。只靠提示词只是在拒答和幻觉之间挪动。
- 误导题是短板：V5 泛化挑错把正常题也带坏了（配对 438 题幻觉 82 → 151），V6 门控拦 19 个幻觉误拒 67 道答对题，V7 规则拦 14 误拒 7。
- 迁移到 Qwen3-8B（另一裁判，不可横比）：原样 67.9% → V4 29.5% → V8（两条窄题型提示）27.1%；V8 的改善集中在误导题，但提示是照着同一开发集的错例写的，属于样本内结果。
- 检索收益偏乐观：知识库是维基，而 SimpleQA 300 题里 258 题的出处就是维基。

![各版本幻觉率与结果构成](report/figures/fig1_versions.png)

## 怎么复现

```bash
pip install -r requirements.txt
python scripts/reproduce_metrics.py
```

```bash
python scripts/make_figures.py
```

第一条打印报告里的全部表格数字（只用标准库），第二条重画 `report/figures/` 下的 4 张图。两者只读 `eval/runs/` 里已有的作答与裁判判定，**不调用任何模型或付费 API，不需要 GPU**。

## 用 SQL 复现

把开发集、作答和裁判逐条判定导入 SQLite（只用标准库），再用纯 SQL 重算报告表 1～3 的核心数字，包括各版本两种口径的 HR、答对、拒答率、分题型结果、V6/V7 离线回放和 V3→V4 配对转移表，并与 Python 结果逐项比对：

```bash
python scripts/build_sqlite.py
```

```bash
python scripts/check_sql.py --print
```

第一条生成 `sql/hallucination.sqlite`，包含 `questions`、`runs`、`answers`、`judgments`、`gate_v6` 五张表，字段含版本、题型和标签；这个文件不进仓库，可以用任何 SQLite 客户端打开查询。第二条会自己在临时目录重建一份库，执行 [`sql/metrics.sql`](sql/metrics.sql) 并打印每段查询的结果，并与 `reproduce_metrics.py --json` 的输出逐项比对，当前结果是 521 项全部一致。`metrics.sql` 末尾另有 8 条分析型查询，每条都在注释里写明它回答什么问题，用到了窗口函数（LAG / RANK / NTILE / ROW_NUMBER）、CTE 和自连接。几个例子：

- 每个题型里幻觉率最低的版本，是不是答对最多的版本？
- 回答越长，越容易被判幻觉吗？
- 相邻版本的配对差异里，有多少题回答一字未改、只是裁判换了判法？

装了 sqlite3 命令行工具的话，也可以直接执行：`sqlite3 sql/hallucination.sqlite < sql/metrics.sql`。

## 裁判一致性

同一份裁判输入被不同裁判、或同一裁判判过多次的重叠样本，全部取自已有记录，不需要新增调用：

- agnes × gemini-flash 跨裁判，188 条：`hallucinated` 的 κ = 0.78 [0.59, 0.91]，`refused` 的 κ = 0.97。
- agnes 自己再判一次，423 对：κ = 0.68。gemini-flash 再判一次，417 对：κ = 0.99。

不一致样本的原因归类，以及下一步的人工复核 rubric 和抽样方案，见 [docs/裁判一致性分析.md](docs/裁判一致性分析.md)。

```bash
python scripts/judge_agreement.py
```

重新跑模型推理（作答、检索、裁判）需要本地 Ollama、维基知识库（约 4.5 GB，`rag/fetch_wiki.py` + `rag/build_kb.py` 构建）、重排模型和裁判 API，这些都不在仓库里。

## 目录结构

```
report/                 评测报告与插图
scripts/                reproduce_metrics.py（重算全部数字）、make_figures.py（画图）、
                        judge_agreement.py（裁判一致性）、build_sqlite.py / check_sql.py（建库、SQL 与 Python 比对）
sql/metrics.sql         用 SQL 重算表 1～3 + 8 条分析型查询
data/external/          公开数据集原文：Chinese SimpleQA、HalluQA
data/dev/               开发集 dev_v1（450 题）、留出集 holdout_v1（3000 题，未使用）、各小样
eval/runs/              每版作答记录、裁判逐条判定（含原始响应）、检索命中记录
eval/FROZEN-*.txt       每版冻结记录：改动、文件哈希、运行情况
eval/judge_acceptance.jsonl   裁判验收集（24 条，六类各 4）
eval/*.py               开发集构造、作答、判分、汇总、门控与规则单测脚本
judge.py                裁判提示词与判定解析
rag/                    维基 BM25 建库、检索、bge 重排
serve/proxy.py          OpenAI 格式代理，各版本策略（STRATEGY=v1…）都在这里
docs/实验日志.md         原始逐版实验日志（原样保留）
docs/外审意见.md         对报告与脚本的审稿意见及修改记录
docs/裁判一致性分析.md    跨裁判 / 同裁判重测的一致率与 κ、不一致样本归因、人工复核 rubric 与抽样方案
```

## 指标口径

- **HR（拒答算有效）** = 幻觉题数 / 有判定题数；**HR（拒答剔除）** = 未拒答题中的幻觉数 / 未拒答题数。
- **答对** = 既不幻觉也不拒答。三个数必须一起看，否则全拒答也能得到 0% 幻觉率。
- **可答题拒答率** = 可答题（SimpleQA + Knowledge）上只拒答的比例，是过度拒答的上界。
- 裁判可靠性（24 条验收集全对，一致率 95% 下界约 86%；HalluQA 上未独立核对）、多重比较、检索评估偏乐观等局限见报告 §3.4 与 §7；外审意见见 [docs/外审意见.md](docs/外审意见.md)。
