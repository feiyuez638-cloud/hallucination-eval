# Deep-Dive Q&A: 20 Follow-up Questions

> The follow-ups an interviewer is most likely to ask, in four groups: metrics, statistics and robustness, the judge, and limits / what next. Each answer is a list of **talking points**, not a script.
>
> - Every number is taken from the [Chinese report](../../report/大模型幻觉评测报告.md) or the script-generated [robustness analysis](../../analysis/稳健性分析.md); `python scripts/check_site_numbers.py` checks each one. The § in brackets is the report section; "R§" is a section of the robustness analysis.
> - Wording as in the [English report](../../report/Hallucination-Evaluation-Report.md): HR (all) counts refusals as non-hallucinated; HR (answered) drops them; "correct" = answered and not judged hallucinated.
> - Fuller Chinese versions: [项目深挖40问](../项目深挖40问.md). If you do not know, say "I did not measure that; here is how I would" — never improvise a number.

## A. Metrics

**Q1. How do you define a hallucination, and what does the judge output?** (§2)
- The judge sees the question, the reference and the answer, and returns two booleans: `hallucinated` (contradicts the reference, or claims a source says what it does not) and `refused` (explicitly cannot answer or lacks material).
- Both can be true — a refusal that invents a reason. V5 has 34 such double-labelled items.
- The judge must rely on the reference only, not its own memory.

**Q2. Why report three numbers instead of one hallucination rate?** (§2)
- HR (all) rewards refusing: V1 has HR 19.4% with a 65.7% refusal rate.
- HR (answered) has a selection effect: V1's 53.2% is computed on only the 154 items it chose to answer.
- Both ratios can be gamed by refusals, so add an absolute count: V1's correct answers fell 121 → 72.

**Q3. Is "correct" really correct?** (§2, §3.4)
- No — the judge does not grade correctness. Off-topic or half answers also land there (hqa-31).
- A rule check on SimpleQA supports it roughly: of 15 sampled "rule mismatch, judge correct" items for V4, 13 really were correct.

**Q4. Why are V6 and V7 offline replays, and are they trustworthy?** (§4.2)
- Both failed their pre-set thresholds, so they never ran for real. The replay assumes triggered items become "I don't know" and everything else keeps V4's judgment — an upper bound on the benefit.
- Even so V6 wrongly refuses 67 correct answers to block 19 hallucinations; its 14.3% is the lowest HR only because it refuses 83 more items.

## B. Statistics and robustness

**Q5. Why McNemar instead of comparing two proportions?** (§2, §4.2)
- Both versions answer the same items, so the data are paired; only discordant items carry information.
- V3 vs V4 looks identical (18.4% vs 18.5%), but paired it is 38 items better and 38 worse — they cancel, they are not the same answers.

**Q6. Why bootstrap on top of Wilson intervals, and do they agree?** (R§1)
- Bootstrap makes no distributional assumption and gives intervals for differences and for HR (answered), which Wilson cannot.
- Per version the two intervals differ by at most 0.4 points, so the report's Wilson intervals are not an artefact of the approximation.
- For every one of the 14 paired tests, "McNemar p<0.05" matches "bootstrap 95% interval excludes 0".

**Q7. You ran many tests. Which conclusions survive multiple-comparison correction?** (§7, R§2)
- 14 tests (7 adjacent pairs × hallucinated / correct), Holm-corrected as one family.
- Still significant: p0→V1, V1→V2, V2→V3 (both metrics), V4→V5 hallucinations, and b0-8B→V4-8B.
- No longer significant: the V3→V4 correct gain (0.018 → 0.092) and the V4-8B→V8-8B hallucination drop (0.036 → 0.143).
- Bonferroni simultaneous intervals give exactly the same split.

**Q8. What does "the conclusion holds" mean in your robustness analysis?** (R§ summary)
- Raw and Holm p<0.05, simultaneous interval excludes 0, Holm still significant in the same direction under all 7 judge-labelling variants and all 9 leave-one-category-out runs, and at least 95% of judge-noise simulations keep the verdict.
- 9 of the 14 pass every check; 2 are "uncertain" (significant only before correction); 3 show no significant difference.
- It is still post-hoc: "holds" means "did not flip under these checks", not a confirmatory result.

**Q9. Is the reranking (V3→V4) gain real?** (R§ summary, R§6)
- Correct +4.2 points [+0.9, +7.6] — the direction is credible, but the simultaneous interval [-0.7, +9.4] includes 0.
- It is only 2 items short of Holm significance, while 12 items in that pair flipped "correct" with a word-for-word identical answer: the effect is the same size as judge noise.
- Under the judge-noise simulation only 83.2% of runs keep the verdict. Settling it needs repeated sampling or human review.

**Q10. Does the overall number depend on the 300/150 sampling mix?** (§7, R§3)
- Re-weighting the four question types equally keeps every Holm-significant result in the same direction with an interval still excluding 0.
- Sizes shrink — retrieval mostly helps knowledge questions — and two small effects flip: the V3→V4 correct gain's interval now includes 0 (its gain is on knowledge items), while V8's correct gain no longer does (it helps misleading items, which the SimpleQA-heavy mix dilutes). Which weighting is "right" depends on real traffic, which a dev set cannot tell.

**Q11. What did stratifying by question type show?** (§4.2, R§3)
- Retrieval's hallucination drop comes from SimpleQA (-31.4 points); on Misleading it is +0.0.
- V5's harm holds inside SimpleQA alone (+17.0 points, Holm p<0.001); its Knowledge harm (raw p=0.008) does not survive the 28-test correction.
- V8 on Misleading, 10 fixed and 0 broken, still has Holm p=0.033 — but the hints were written from those very items.

**Q12. How sensitive is V8 to missing judgments?** (R§ summary)
- V4-8B has 12 and V8-8B 11 items with no judgment. Filling them in V8's favour gives 30/8 discordant pairs and Holm p=0.002; against it, 20/21 and Holm p=1.000.
- So that conclusion hinges on the missing judgments as much as on the data — another reason to confirm it on held-out data.

## C. The judge

**Q13. The judge got 24 of 24 acceptance items right. Is it accurate?** (§3.3)
- Not proven: a 95% Wilson lower bound of 86.2% — a true error rate of 14% is consistent with that.
- The acceptance set is rule-document material; the dev set is open-domain QA.

**Q14. How consistent is the judge with itself and with another judge?** (§3.4)
- Cross-judge on 188 identical inputs: κ 0.78 [0.59, 0.91]. agnes re-judging itself at temperature 0: κ 0.68 over 423 pairs; gemini-flash: κ 0.99.
- In the V3→V4 pair, 13 of the 76 hallucination-discordant items had a word-for-word identical answer.

**Q15. How did you turn judge noise into a robustness check?** (R§6)
- From the retest pairs: 24 of 423 disagree for agnes, which implies a per-judgment flip rate of 2.9%; gemini-flash 0.2%.
- I flip labels at that rate on top of the stored judgments, 2000 times, and count how often each Holm verdict survives. Since the stored labels already contain noise, this doubles it — conservative.
- Plus a tipping count: how many discordant items must be judge errors before a verdict flips. Retrieval needs 96 removed; reranking needs only 2 more.

**Q16. Which judge-labelling variants did you try, and did any change a conclusion?** (R§5)
- Drop lenient re-parses; double labels count as refusals only; "the material does not mention…" answers re-labelled as refusals; hallucinations whose reason only cites extra, unmentioned details ignored; missing judgments filled worst / best case; only items whose answer changed.
- Across 7 variants × 14 tests, significance changed in 2 places — both the V8 rows under best-case filling of missing judgments.
- Version order by HR (all) never changes except V3 / V4, which are tied anyway.

## D. Limits and next steps

**Q17. What is the biggest limitation?** (§1, §7)
- Single judge, single sample, dev set only; the judge is not independently checked on HalluQA; 4B and 8B used different judges and cannot be compared.
- Robustness checks only cover item sampling and the judge problems I know about — a systematic judge bias is invisible to them.

**Q18. Which results are in-sample?** (§7)
- V5, V7 and V8 were designed from errors on the same dev set, and V4 was picked as the best version on it. The held-out set was never used.
- The clearest case is V8: Misleading HR 58.2% → 42.1%, using phrasings distilled from those same items.

**Q19. Is the retrieval gain overestimated?** (§7)
- Yes: the knowledge base is Wikipedia and 258 of the 300 SimpleQA items cite Wikipedia as their source.
- For the other 42 items, top-5 hit rate is only 42.9%. Expect a smaller gain on real, non-Wikipedia questions.

**Q20. What would you do first if you redid it?** (§8)
- Pre-register one primary metric and one primary comparison, then run V4 / V8 once on the 3000-item held-out set.
- Two annotators label 300 items from the two baselines: inter-annotator κ first, then judge vs human.
- Sample each key version 3 times to measure answer variance before judging small differences like V3 vs V4.
