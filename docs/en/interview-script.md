# Interview Script: Chinese LLM Hallucination Evaluation

> Two spoken versions; pick the one that fits the time you are given. **Every number here comes from the [Chinese report](../../report/大模型幻觉评测报告.md) or the script-generated [robustness analysis](../../analysis/稳健性分析.md)**, and `python scripts/check_site_numbers.py` checks each one. English wording follows the [English report](../../report/Hallucination-Evaluation-Report.md): "HR (all)" counts refusals as non-hallucinated, "HR (answered)" drops refusals, and "correct" means *answered and not judged hallucinated* — the judge never grades correctness on its own. Follow-up questions: [deep-dive Q&A](deep-dive-qa.md).
>
> Caveat to keep in mind while speaking: all numbers are proxy metrics from a local LLM judge — single judge, single sample, dev set — not an official score.

## 1-minute version

> I built a hallucination evaluation for Chinese LLMs. I sampled a 450-item dev set from Chinese SimpleQA and HalluQA with a fixed seed, and used an LLM judge that sees only the reference material and outputs two labels: hallucinated, and refused. Then I tested Qwen3-4B version by version — prompt constraints, Wikipedia retrieval, reranking, premise checking, gating and keyword rules — changing one thing at a time, with pass criteria written down before each run.
>
> Three findings. First, a single hallucination rate is easy to game: a prompt change alone took HR from 72.5% to 19.4%, but only because 65.7% of items were refused, and correct answers actually fell from 121 to 72. So I always report two HR variants plus the correct count. Second, retrieval is the only step that both cut hallucinations and added correct answers significantly — paired McNemar, p<0.001 — and it is still significant after Holm correction, under every judge-labelling variant I tried, and with any one category left out. Third, misleading questions remain the weak spot: three attempts failed their pre-set criteria.
>
> Every number is recomputed by one command from the stored judgments, and independently in SQL.

(About one minute at a normal pace. If cut short, keep only the first finding.)

## 3-minute version

> **Why.** This started as practice for a hallucination competition. The official metric was hallucinated samples over valid samples, with no rule for refusals — so a model that refuses everything scores zero. From day one I reported three numbers together: HR (all), HR (answered), and the correct count.
>
> **Setup.** The dev set is 450 items: 300 short factual questions from Chinese SimpleQA and 150 from HalluQA, half of which carry a false premise. Another 3000 items are held out and were never touched. The model sees only the question; the gold answer goes only to the judge. Before using the judge I validated it on 24 hand-labelled cases and it got every one right — but I always add that 24 out of 24 only bounds agreement below at about 86%.
>
> **Results.** Qwen3-4B as-is: HR 72.5%. A constraining system prompt: 19.4%, but 65.7% refused and correct fell from 121 to 72 — it traded hallucinations for refusals. Relaxing the refusal rule brought hallucinations back. Adding Wikipedia BM25 retrieval: paired, hallucinations 206 → 83 and correct 138 → 249, both p<0.001. Adding bge reranking: hallucinations flat, correct up to 267, and I picked it as the baseline at 18.5%.
>
> **What failed.** For misleading questions I tried a generic "check the question first" prompt, which spread to ordinary questions and pushed paired hallucinations from 82 to 151; a self-judging gate that blocked 19 hallucinations but wrongly refused 67 correct answers; and keyword rules that blocked 14 and wrongly refused 7. None passed, so I stopped instead of tuning on the dev set.
>
> **How sure am I.** I ran a robustness pass on all 14 paired tests: bootstrap intervals, Holm correction, seven judge-labelling variants, leave-one-category-out over nine categories, and a judge-noise simulation calibrated on the judge's own retest flips. 9 of the 14 conclusions hold under every check. Two do not: the reranking gain of 19 correct answers has a 95% interval of [+0.9, +7.6] points but loses significance after correction, and is only 2 items short of significance while 12 items in that pair flipped with a word-for-word identical answer — so its size is within judge noise. And the Qwen3-8B gain from two narrow hints (29.5% → 27.1%) has a Holm-corrected p that swings from 0.002 to 1.000 depending on how the 12 and 11 missing judgments are filled, and is in-sample anyway.
>
> **Next** is one run on the held-out set with a single pre-registered comparison, then two-annotator human labels to measure judge-vs-human agreement.

(A little over three minutes. To shorten, drop the last sentence of "How sure am I".)

## If asked "what would you say in one sentence?"

> The point of an evaluation is not a low number but a conclusion that survives: whether the metric can be gamed, whether the judge is reliable, and whether a difference is just noise.
