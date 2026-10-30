-- 用 SQL 重算报告核心数字，并做几条分析型查询。
-- 先运行 python scripts/build_sqlite.py 生成 sql/hallucination.sqlite，再：
--   python scripts/check_sql.py --print     （逐条执行并打印，同时与 reproduce_metrics.py 逐项比对）
--   或 sqlite3 sql/hallucination.sqlite < sql/metrics.sql
-- 每段以「-- name: 名字」开头，check_sql.py 按名字取结果做比对。
-- 口径与 reproduce_metrics.py 相同：HR(拒答算有效) = 幻觉 / 有判定；HR(拒答剔除) = 未拒答中的幻觉 / 未拒答；
-- 答对 = 既不幻觉也不拒答；双标签（拒答同时编造）既计幻觉也计拒答。比率输出为未取整的百分数。

-- name: setup
-- 两个视图。v_verdict：每个（作答文件, 裁判, 题）取文件里最后一条有效判定（同 reproduce_metrics.py「后出现的覆盖前面的」）。
DROP VIEW IF EXISTS v_main;
DROP VIEW IF EXISTS v_verdict;
CREATE VIEW v_verdict AS
SELECT run, judge, qid, hallucinated, refused, parse
FROM (SELECT j.*, ROW_NUMBER() OVER (PARTITION BY run, judge, qid ORDER BY line_no DESC) AS rn
      FROM judgments j
      WHERE run IS NOT NULL AND hallucinated IS NOT NULL)
WHERE rn = 1;
-- v_main：报告里的正式版本，每版只用报告指定的裁判（4B 用 agnes，8B 用 gemini-flash-high），附题型与三分类标签。
CREATE VIEW v_main AS
SELECT r.version, r.family, r.ord, v.qid, q.qtype, q.source, v.parse,
       v.hallucinated AS hal, v.refused AS ref,
       CASE WHEN v.hallucinated THEN '幻觉' WHEN v.refused THEN '拒答' ELSE '答对' END AS label
FROM v_verdict v
JOIN runs r ON r.run = v.run AND r.judge = v.judge
JOIN questions q ON q.qid = v.qid
WHERE r.version IS NOT NULL;

-- name: main_table
-- 报告表 1 / 表 3：各版本作答数、有判定、幻觉、拒答、双标签、答对、两种口径 HR、拒答率、可答题拒答率、作答时延 P50。
-- 可答题 = SimpleQA + Knowledge；过度拒答 = 可答题上只拒答（拒答且未编造）。P50 取排序后第 ⌊n/2⌋+1 个（同 Python）。
WITH ans AS (
    SELECT run, COUNT(*) AS answered, SUM(error IS NOT NULL) AS answer_fail FROM answers GROUP BY run),
lat AS (
    SELECT run, latency_s,
           ROW_NUMBER() OVER (PARTITION BY run ORDER BY latency_s) AS rn,
           COUNT(*) OVER (PARTITION BY run) AS cnt
    FROM answers WHERE error IS NULL AND latency_s IS NOT NULL),
agg AS (
    SELECT version, family, ord, COUNT(*) AS n, SUM(parse = 'lenient') AS lenient,
           SUM(hal) AS hal, SUM(ref) AS ref, SUM(hal AND ref) AS both_labels, SUM(label = '答对') AS ok,
           SUM(hal AND NOT ref) AS hal_kept, SUM(NOT ref) AS n_kept,
           SUM(ref AND NOT hal AND qtype IN ('SimpleQA', 'Knowledge')) AS over_ref,
           SUM(qtype IN ('SimpleQA', 'Knowledge')) AS answerable
    FROM v_main GROUP BY version, family, ord)
SELECT g.family, g.version, a.answered, a.answer_fail, g.n, g.lenient, g.hal, g.ref, g.both_labels, g.ok,
       100.0 * g.hal / g.n AS hr_pct,
       100.0 * g.hal_kept / g.n_kept AS hr_ex_pct,
       100.0 * g.ref / g.n AS ref_rate_pct,
       g.over_ref, g.answerable, 100.0 * g.over_ref / g.answerable AS over_ref_pct,
       l.latency_s AS lat_p50
FROM agg g
JOIN runs r ON r.version = g.version
JOIN ans a ON a.run = r.run
LEFT JOIN lat l ON l.run = r.run AND l.rn = l.cnt / 2 + 1
ORDER BY g.family, g.ord;

-- name: by_type
-- 报告表 2 与表 3 的分题型列：各版本 × 题型的有判定、幻觉、拒答、答对、两种口径 HR。HalluQA = 后三类合计。
WITH g AS (
    SELECT family, ord, version, qtype AS grp, hal, ref, label FROM v_main
    UNION ALL
    SELECT family, ord, version, 'HalluQA', hal, ref, label FROM v_main WHERE source = 'halluqa'
    UNION ALL
    SELECT family, ord, version, '全部', hal, ref, label FROM v_main)
SELECT family, version, grp, COUNT(*) AS n, SUM(hal) AS hal, SUM(ref) AS ref, SUM(label = '答对') AS ok,
       100.0 * SUM(hal) / COUNT(*) AS hr_pct,
       100.0 * SUM(hal AND NOT ref) / NULLIF(SUM(NOT ref), 0) AS hr_ex_pct
FROM g
GROUP BY family, ord, version, grp
ORDER BY family, ord,
         CASE grp WHEN '全部' THEN 0 WHEN 'SimpleQA' THEN 1 WHEN 'HalluQA' THEN 2 WHEN 'Knowledge' THEN 3
                  WHEN 'Misleading' THEN 4 ELSE 5 END;

-- name: replay_v6_v7
-- 报告表 1 的 V6* / V7* 行：用 V4 的判定离线回放。触发的题改记「只拒答」（非幻觉），其余保留 V4 判定。
-- V6 触发 = 门控判「是」；V7 触发 = 题目命中任一关键词规则（规则命中在建库时算好，存在 questions.v7_rules）。
WITH rep AS (
    SELECT 'V6*' AS version, m.hal, m.ref, m.label, g.verdict = '是' AS hit
    FROM v_main m JOIN gate_v6 g ON g.qid = m.qid WHERE m.version = 'V4'
    UNION ALL
    SELECT 'V7*', m.hal, m.ref, m.label, q.v7_rules IS NOT NULL
    FROM v_main m JOIN questions q ON q.qid = m.qid WHERE m.version = 'V4')
SELECT version, COUNT(*) AS n,
       SUM(hit AND label = '幻觉') AS blocked_hal, SUM(hit AND label = '答对') AS false_refuse,
       SUM(hal AND NOT hit) AS hal, SUM(ref OR hit) AS ref, SUM(label = '答对' AND NOT hit) AS ok,
       100.0 * SUM(hal AND NOT hit) / COUNT(*) AS hr_pct,
       100.0 * SUM(hal AND NOT hit AND NOT ref) / SUM(NOT ref AND NOT hit) AS hr_ex_pct,
       100.0 * SUM(ref OR hit) / COUNT(*) AS ref_rate_pct
FROM rep GROUP BY version ORDER BY version;

-- name: transition_v3_v4
-- V3 → V4 配对转移表（行 V3，列 V4，只算两版都有判定的题）。自连接 v_main。
SELECT a.label AS v3, SUM(b.label = '答对') AS v4_答对, SUM(b.label = '拒答') AS v4_拒答, SUM(b.label = '幻觉') AS v4_幻觉
FROM v_main a
JOIN v_main b ON b.qid = a.qid AND a.version = 'V3' AND b.version = 'V4'
GROUP BY a.label
ORDER BY CASE a.label WHEN '答对' THEN 0 WHEN '拒答' THEN 1 ELSE 2 END;

-- name: paired_adjacent
-- 相邻版本配对（同一系列内 ord 相差 1）：共同有判定题数、两版幻觉 / 答对数，以及 McNemar 用的两个不一致方向计数。
-- 回答：每一步改动在同一批题上让多少题变好、多少题变坏？（p 值由 check_sql.py 用精确二项检验从计数算出）
SELECT x.family, x.version AS a, y.version AS b, COUNT(*) AS common,
       SUM(x.hal) AS hal_a, SUM(y.hal) AS hal_b,
       SUM(x.hal AND NOT y.hal) AS only_a_hal, SUM(y.hal AND NOT x.hal) AS only_b_hal,
       SUM(x.label = '答对') AS ok_a, SUM(y.label = '答对') AS ok_b,
       SUM(x.label = '答对' AND y.label <> '答对') AS only_a_ok, SUM(y.label = '答对' AND x.label <> '答对') AS only_b_ok
FROM v_main x
JOIN v_main y ON y.family = x.family AND y.ord = x.ord + 1 AND y.qid = x.qid
GROUP BY x.family, x.ord, x.version, y.version
ORDER BY x.family, x.ord;

-- ======================== 分析型查询 ========================

-- name: a1_version_delta
-- 问：每一版相对上一版，幻觉数、答对数、HR 各变了多少？（窗口函数 LAG；未配对的总数差，配对结论见 paired_adjacent）
WITH t AS (
    SELECT family, ord, version, COUNT(*) AS n, SUM(hal) AS hal, SUM(label = '答对') AS ok, 100.0 * SUM(hal) / COUNT(*) AS hr
    FROM v_main GROUP BY family, ord, version)
SELECT family, version, n, hal, ok, ROUND(hr, 1) AS hr_pct,
       hal - LAG(hal) OVER w AS d_hal,
       ok - LAG(ok) OVER w AS d_ok,
       ROUND(hr - LAG(hr) OVER w, 1) AS d_hr_pp
FROM t
WINDOW w AS (PARTITION BY family ORDER BY ord)
ORDER BY family, ord;

-- name: a2_rank_by_type
-- 问：在每个题型里，幻觉率最低的 4B 版本是不是也是答对最多的版本？（窗口函数 RANK）
-- 预期看到：幻觉率最低的往往是拒答最多的版本（如 V1），答对最多的另有其人——只看 HR 会选错版本。
WITH t AS (
    SELECT qtype, version, 100.0 * SUM(hal) / COUNT(*) AS hr, SUM(label = '答对') AS ok, SUM(ref) AS ref
    FROM v_main WHERE family = '4B' GROUP BY qtype, version),
r AS (
    SELECT qtype, version, ROUND(hr, 1) AS hr_pct, ok, ref,
           RANK() OVER (PARTITION BY qtype ORDER BY hr) AS hr_rank,
           RANK() OVER (PARTITION BY qtype ORDER BY ok DESC) AS ok_rank
    FROM t)
SELECT * FROM r WHERE hr_rank = 1 OR ok_rank = 1 ORDER BY qtype, hr_rank;

-- name: a3_length_quartile
-- 问：在没拒答的回答里，回答越长越容易被判幻觉吗？（窗口函数 NTILE 按长度分四档；CTE；只看未拒答的回答）
-- 用于检验报告里「p0 回答很长、在正确答案旁夹带错误细节」的说法。
WITH x AS (
    SELECT m.version, m.ord, m.family, m.hal, LENGTH(a.answer) AS len,
           NTILE(4) OVER (PARTITION BY m.version ORDER BY LENGTH(a.answer), m.qid) AS quartile
    FROM v_main m
    JOIN runs r ON r.version = m.version
    JOIN answers a ON a.run = r.run AND a.qid = m.qid
    WHERE m.version IN ('p0', 'V4', 'V4-8B') AND NOT m.ref)
SELECT version, quartile, COUNT(*) AS n, MIN(len) AS min_len, MAX(len) AS max_len,
       ROUND(100.0 * SUM(hal) / COUNT(*), 1) AS hr_pct
FROM x GROUP BY family, ord, version, quartile ORDER BY family, ord, quartile;

-- name: a4_stubborn
-- 问：有没有「怎么改都幻觉」的顽固题？按题型统计 4B 六个版本里次次幻觉 / 从不幻觉的题数（CTE + 条件聚合）。
-- 顽固题是下一轮改进（或检查标准答案 / 裁判口径）的优先清单。
WITH per_q AS (
    SELECT qid, qtype, COUNT(*) AS judged, SUM(hal) AS n_hal
    FROM v_main WHERE family = '4B' GROUP BY qid, qtype)
SELECT qtype, COUNT(*) AS questions,
       SUM(judged = 6 AND n_hal = 6) AS always_hal,
       SUM(judged = 6 AND n_hal = 0) AS never_hal,
       ROUND(AVG(n_hal), 2) AS avg_hal_versions
FROM per_q GROUP BY qtype ORDER BY qtype;

-- name: a5_4b_vs_8b
-- 问：同一道题，4B 和 8B 在同一配置（V4）下的结果怎么对应？（自连接 v_main）
-- 注意 4B / 8B 的裁判不同，这张交叉表混入了裁判差异，只能看大致方向（如 8B 把哪些 4B 拒答的题改成了作答）。
SELECT a.label AS v4_4b, SUM(b.label = '答对') AS v4_8b_答对, SUM(b.label = '拒答') AS v4_8b_拒答,
       SUM(b.label = '幻觉') AS v4_8b_幻觉, COUNT(*) AS n
FROM v_main a
JOIN v_main b ON b.qid = a.qid AND a.version = 'V4' AND b.version = 'V4-8B'
GROUP BY a.label
ORDER BY CASE a.label WHEN '答对' THEN 0 WHEN '拒答' THEN 1 ELSE 2 END;

-- name: a6_cross_judge
-- 问：裁判输入一字不差（prompt_sha256 相同）时，agnes 和 gemini-flash 的 hallucinated 判定有多一致？
-- （ROW_NUMBER 取每个裁判对同一输入最早的一条有效判定；自连接 judgments；只看开发集题目。与 scripts/judge_agreement.py 同口径）
WITH first AS (
    SELECT prompt_sha256, judge, qid, hallucinated, refused,
           ROW_NUMBER() OVER (PARTITION BY prompt_sha256, judge ORDER BY started, file) AS rn
    FROM judgments WHERE hallucinated IS NOT NULL)
SELECT a.hallucinated AS agnes_hal, b.hallucinated AS flash_hal, COUNT(*) AS n
FROM first a
JOIN first b ON b.prompt_sha256 = a.prompt_sha256 AND b.rn = 1 AND b.judge = 'gemini-3.8-flash-high'
WHERE a.rn = 1 AND a.judge = 'agnes-2.5-flash' AND a.qid IN (SELECT qid FROM questions)
GROUP BY a.hallucinated, b.hallucinated
ORDER BY a.hallucinated DESC, b.hallucinated DESC;

-- name: a7_retest_flip
-- 问：同一裁判对一字不差的输入判了不止一次时，有多少输入出现过前后矛盾的 hallucinated？（GROUP BY + HAVING 思路的条件聚合）
-- 与 judge_agreement.py 只取最早两次不同，这里统计「所有重复判定中出现过翻转」的输入，数字会略大。
WITH rep AS (
    SELECT judge, prompt_sha256, COUNT(*) AS times, COUNT(DISTINCT hallucinated) AS distinct_hal,
           COUNT(DISTINCT refused) AS distinct_ref
    FROM judgments
    WHERE hallucinated IS NOT NULL AND qid IN (SELECT qid FROM questions)
    GROUP BY judge, prompt_sha256
    HAVING COUNT(*) >= 2)
SELECT judge, COUNT(*) AS repeated_inputs, SUM(times) AS judgments,
       SUM(distinct_hal > 1) AS hal_flipped, ROUND(100.0 * SUM(distinct_hal > 1) / COUNT(*), 1) AS hal_flip_pct,
       SUM(distinct_ref > 1) AS ref_flipped
FROM rep GROUP BY judge ORDER BY judge;

-- name: a8_same_answer_flip
-- 问：相邻版本的配对差异里，有多少题其实回答一字未改、只是裁判换了判法？（自连接 v_main，再各自连回 answers 比较原文）
-- 这部分差异纯属裁判噪声，给「差几题的版本差异不可靠」一个下界证据。
SELECT x.family, x.version AS a, y.version AS b, COUNT(*) AS common,
       SUM(ax.answer = ay.answer) AS same_answer,
       SUM(x.hal <> y.hal) AS hal_discordant,
       SUM(ax.answer = ay.answer AND x.hal <> y.hal) AS same_answer_hal_flip,
       SUM(ax.answer = ay.answer AND x.label <> y.label) AS same_answer_label_flip
FROM v_main x
JOIN v_main y ON y.family = x.family AND y.ord = x.ord + 1 AND y.qid = x.qid
JOIN runs rx ON rx.version = x.version
JOIN answers ax ON ax.run = rx.run AND ax.qid = x.qid
JOIN runs ry ON ry.version = y.version
JOIN answers ay ON ay.run = ry.run AND ay.qid = y.qid
GROUP BY x.family, x.ord, x.version, y.version
ORDER BY x.family, x.ord;
