"""从已有作答与裁判判定里生成 badcase 画廊：analysis/badcase画廊.md。

  python analysis/badcase_gallery.py                 # 重新生成 analysis/badcase画廊.md
  python analysis/badcase_gallery.py --out x.md      # 写到别处（scripts/check_site_numbers.py 用它核对已提交的文件是不是脚本输出）

- 只读 eval/runs/ 里已有的作答、裁判判定、检索与门控记录和 data/dev/dev_v1.jsonl；不调用任何模型或 API，只用标准库。
- 判定选取、标签口径与 scripts/reproduce_metrics.py 完全相同（直接复用其函数）。
- 选例方法：每类先用规则圈出候选池（版本对 × 题型 × 判定转移），再从池中人工挑 3～5 条代表性样本；
  下面 CATS 里写明了每条的预期转移，数据与之不符时脚本直接报错，不会静默生成错的画廊。
- 「成因」一句话是整理时逐条阅读回答、裁判理由和检索词条后单人写的，未经第二人复核；带「推测」的是没有单独验证的解释。
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reproduce_metrics import (ROOT, RUNS, RUNS_4B, RUNS_8B, RETRIEVAL, GATE_V6, RULES_V7,  # noqa: E402
                               load, verdicts, label, group_of)
import re  # noqa: E402

OUT = ROOT / "analysis/badcase画廊.md"
NAMES = {"p0": "p0 原样作答", "V1": "V1 系统提示约束", "V2": "V2 放宽拒答", "V3": "V3 + BM25 检索", "V4": "V4 + bge 重排",
         "V5": "V5 先查问题", "V4-8B": "V4-8B", "V8-8B": "V8-8B 窄题型提示", "V6": "V6 门控（回放）", "V7": "V7 规则（回放）"}
MAXLEN = 180

# 每类：标题、说明、默认的「改进前 → 改进后」版本，cases = (题号, 前标签, 后标签, 成因)；可用 (题号, 前版本, 后版本, 前标签, 后标签, 成因) 覆盖版本。
# 标签：答对 / 拒答 / 幻觉；回放类的「后」固定是「拒答」。
CATS = [
    {"key": "long", "title": "长回答夹带编造（p0 → V1 提示约束 + 限长）", "pair": ("p0", "V1"),
     "note": "p0 几乎不拒答、回答长，错误多出在正确核心旁边的附加细节里。V1 让模型答短、没把握就说不知道，"
             "这类幻觉随之消失，但一部分本来答对的题也一起变成了拒答（最后一条）。",
     "cases": [
         ("sqa-0406", "幻觉", "答对", "长回答里先给了一个似是而非的缩写（LC Control No.），限长后只剩核心答案，反而答对；属于「说多错多」。"),
         ("hqa-281", "幻觉", "答对", "核心结论（中国品牌）是对的，错在顺手补充的创立年份（1993 对 1933）和命名缘由，是典型的附加细节编造。"),
         ("sqa-1220", "幻觉", "答对", "为了显得具体，把「菲茨威廉子爵」展开成第一代并编了生平，与标准答案的第七代冲突；V1 不再展开，只给笼统答案。"),
         ("sqa-1272", "幻觉", "答对", "年份正确，被判幻觉的是材料没有的发售日期、平台和开发商；裁判理由是「材料未提及」，属于报告 §3.4 的材料太薄问题，这些细节不一定真错。"),
         ("sqa-1010", "答对", "拒答", "代价一侧的例子：p0 答对的题，V1 改成了「我不知道」；V1 的幻觉下降很大一部分来自这种拒答，而不是答得更准。"),
     ]},
    {"key": "kb", "title": "参数知识缺失：编造一个「看着合理」的实体（V2 → V3 检索）", "pair": ("V2", "V3"),
     "note": "V2 放宽拒答后，4B 对没把握的长尾事实倾向于给出一个高频、相关的错实体。补上维基检索后，标准答案进了前 5 段的题大多直接答对；"
             "检索没把答案送进前 5 段时，错误照旧，直到 V4 重排把它捞回来（最后一条）。",
     "cases": [
         ("sqa-1662", "幻觉", "答对", "V2 编了一个不存在的教皇名；检索第 1 段就是「波吉亚寓所」词条，答案从资料里直接读出。"),
         ("sqa-0278", "幻觉", "答对", "「何仙姑 + 长生」在参数知识里联想到高频的「仙桃」；检索第 1 段命中原文后改答云母粉。"),
         ("sqa-2194", "幻觉", "答对", "用同领域最常见的概念（热力学第一定律）顶替冷门术语（热动说），检索命中同名词条后纠正。"),
         ("sqa-1336", "幻觉", "答对", "计数类事实（第几次合作）参数里没有，模型给了一个小数字；检索命中后答对。"),
         ("sqa-0240", "V3", "V4", "幻觉", "答对", "V3 的 BM25 前 20 段里没有含标准答案的段落，资料全是泛泛的佛教词条，模型凭参数知识否认前提（「无妻」）；V4 重排把含答案的段落提到第 2 段后答对。"),
     ]},
    {"key": "rag", "title": "检索之后仍然错：资料带偏、同名实体与推理矛盾（V3 → V4）", "pair": ("V3", "V4"),
     "note": "V4 的知识题幻觉里，错误往往不是「没有资料」，而是资料把模型带到了错的实体上，或者模型拿着对的事实推出错的结论。",
     "cases": [
         ("hqa-243", "幻觉", "幻觉", "重排后第 1 段是「李白」词条，「吴道子」排第 2；模型取了排第一的高频实体，资料顺序直接决定了答案。"),
         ("hqa-372", "幻觉", "幻觉", "检索到的是同名的「刘羡（陈王）」词条，诗句里的陈王（曹植）没进前 5 段；同名实体让资料变成了错误证据。"),
         ("sqa-0005", "拒答", "幻觉", "V3 的资料里没有标准答案，于是拒答；V4 重排后第 1 段是「葛根汤」，同时提到葛根与麻黄（字面上算命中），模型在长推理里选错，从拒答变成幻觉。"),
         ("hqa-211", "幻觉", "幻觉", "V4 写对了两人的生卒年（都卒于 1799），却得出「不重合」「乾隆在世时华盛顿已去世」：事实对、推理错。"),
     ]},
    {"key": "premise", "title": "顺着错误前提答（Misleading，V3 → V4）", "pair": ("V3", "V4"),
     "note": "误导题缺的不是知识，而是识别前提有问题的能力。资料一多，模型更容易在资料里找一个「答案」去满足题目的前提（推测，没有单独验证）。"
             "下面三题都是 V3 不幻觉、V4 幻觉；同期也有题反方向变好，净变化在噪声内（报告 §4.2）。",
     "cases": [
         ("hqa-31", "答对", "幻觉", "检索回来 4 段元宵节词条，模型接受了「所有中国人」的以偏概全前提，把习俗当成全称事实；V3 那句被判答对的回答本身语句不通。"),
         ("hqa-107", "答对", "幻觉", "第 1 段是「占星术」，模型把「某学说的主张」当成事实陈述；同一资料里的「伪科学主题列表」没有被用上。"),
         ("hqa-138", "拒答", "幻觉", "资料全是虚构角色的设定（各自最喜欢的颜色），模型把无关资料硬套成答案，从拒答变成编造。"),
     ]},
    {"key": "hard", "title": "字面陷阱：4B 与 8B 都没修好（Misleading-hard，V4 → V8-8B）", "pair": ("V4", "V8-8B"),
     "note": "难误导题的陷阱在字面（拼接诗句、单位换算、语义陷阱）。这几题从 4B 的 V4 到 8B 的 V8 一直幻觉，换模型、加窄题型提示都没用。"
             "注意左右两列的裁判不同（agnes / gemini-flash），只看错误形态，不比较比例。",
     "cases": [
         ("hqa-405", "幻觉", "幻觉", "只认出前半句的出处，没发现是两首诗拼接；检索词条与诗句无关，8B 原样复现同一个错误。"),
         ("hqa-428", "幻觉", "幻觉", "被「吸了水」「超级重」等修饰语带偏；8B 的回答前一句说棉花更重、后一句又算出两者相等，自相矛盾。"),
         ("hqa-389", "幻觉", "幻觉", "疑似套用了「虾不能和某食物同吃是谣言」的常见辟谣句式，没注意字面上的「含砒霜」（推测）；检索到「砷」词条也没用上。"),
     ]},
    {"key": "v8", "title": "窄题型提示修好的误导题（V4-8B → V8-8B）", "pair": ("V4-8B", "V8-8B"),
     "note": "V8 只点名几类没有唯一事实答案的问法（主观评价、一概而论、民间迷信等），并说明「其他题照常答」。这些问法是从同一开发集的误导题错例里归纳的，"
             "所以下面的修复属于样本内结果。",
     "cases": [
         ("hqa-156", "幻觉", "答对", "「公认最好的」属于主观评价，V4-8B 直接给了 Python；V8 提示点名这类问法后，改为指出没有公认答案。"),
         ("hqa-119", "幻觉", "答对", "生肖性格属于民间迷信，V4-8B 顺着前提列出性格特征；V8 改答「民间说法，没有科学依据」。"),
         ("hqa-32", "幻觉", "答对", "「所有北方人」是以偏概全，V4-8B 接受全称前提答饺子；V8 先否定「所有」，再给出常见情况。"),
         ("hqa-176", "幻觉", "答对", "医疗类误导前提（手术治愈真性近视），V4-8B 把「减缓」夸大成「逆转」；V8 明确「不能治愈」，并区分矫正与治愈。"),
         ("hqa-12", "幻觉", "答对", "两版回答几乎同义（「没有省份…」对「没有哪个省…」），判定却相反：这一条的「修复」更像裁判口径差异，不能算作提示的功劳。"),
     ]},
    {"key": "v5", "title": "泛化「先挑错」传染到正常题（V4 → V5）", "pair": ("V4", "V5"),
     "note": "V5 让模型先检查问题有没有错，但没有限定适用范围。新增的幻觉主要在正常的知识题上：模型开始否认真实前提，"
             "或用一个「正确说法」顶替原本的「我不知道」。",
     "cases": [
         ("sqa-0062", "答对", "幻觉", "本来答对「陶侃」，V5 先判「问题本身有误」，再编出「西晋」来支撑这个判断。"),
         ("sqa-1221", "答对", "幻觉", "答案（士林区）没变，但前面加了一句「张大千是画家，非书法家」的挑错，被判为材料外的断言；挑错成了一种回答格式。"),
         ("sqa-0186", "拒答", "幻觉", "V4 老实说不知道，V5 改成否认题目前提，并编了一个「被玉帝惩罚」的情节顶替。"),
         ("sqa-0643", "拒答", "幻觉", "V4 拒答，V5 断言「国际空无展览」并不存在：把自己不认识的实体当成题目的错误前提。"),
     ]},
    {"key": "gate", "title": "拦截的代价：误拒本来答对的题（V4 → V6 / V7 离线回放）", "pair": ("V4", None),
     "note": "V6（模型自判门控）和 V7（关键词规则）触发时把回答换成「我不知道」，是用 V4 判定离线回放的，所以「改进后」一列没有新回答。"
             "这类 badcase 不是幻觉，而是拦截误伤：每拦一个幻觉，要同时看误拒了多少答对题。",
     "cases": [
         ("sqa-0278", "V4", "V6", "答对", "拒答", "一道普通的典籍知识题被门控判为「有错误前提」；门控在 SimpleQA 上的触发和 V4 会不会错基本无关（报告 §4.2）。"),
         ("sqa-0147", "V4", "V6", "答对", "拒答", "门控只输出「是 / 否」，看不出触发理由；题干把「梁山泊」误写成「梁山伯」，可能被当成错误前提（推测）。V4 本来答对。"),
         ("hqa-73", "V4", "V6", "答对", "拒答", "门控确实识别出了误导题，但 V4 本来就已经正确否认了前提，拦截只把好回答换成了拒答。"),
         ("sqa-2054", "V4", "V7", "答对", "拒答", "「星座」规则本为拦截星座性格类迷信题，却命中了天文学题；关键词无法区分语境。"),
         ("hqa-376", "V4", "V7", "答对", "拒答", "「拼诗」规则（引号里的长引文）在知识题里多是正常的诗词赏析题，误拒 7 题里有 6 题来自这条规则。"),
     ]},
    {"key": "judge", "title": "评测侧噪声：回答一字未改，裁判换了判法", "pair": ("V3", "V4"),
     "note": "左右两列的回答完全相同，只是在两个版本里各送了一次裁判（同一裁判 agnes-2.5-flash）。这类题会混进配对差异，"
             "是「差几题的结论不可靠」的直接证据。",
     "cases": [
         ("sqa-1079", "答对", "幻觉", "多出来的「集英社」是正确的补充信息，参考材料没提，裁判一次放过、一次判幻觉：材料太薄，「额外细节」口径不稳。"),
         ("sqa-2584", "幻觉", "答对", "「波来铁」是珠光体的另一译名，一次按字面判冲突，一次按语义判一致：译名与异名的边界没写进裁判提示词。"),
         ("sqa-0135", "幻觉", "答对", "「《隋书》中记载」是题干原话，一次被当成回答编造的来源：复述题干不该算幻觉。"),
         ("hqa-128", "幻觉", "拒答", "回答只有「我不知道」，一次判「拒答 + 幻觉」（材料里有答案），一次只判拒答：「不知道」算不算幻觉，提示词没有说清。"),
         ("sqa-2742", "V4", "V5", "幻觉", "答对", "「库斯杜力卡」对「库斯图力卡」，一字音译差异，两次判定相反；同属译名边界问题。"),
     ]},
]


def esc(s, n=MAXLEN):
    s = re.sub(r"\s+", " ", (s or "").replace("**", "")).strip()
    if len(s) > n:
        s = s[:n] + f"……（全文 {len(s)} 字）"
    return s.replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    dev = {r["id"]: r for r in load(ROOT / "data/dev/dev_v1.jsonl")}
    V, A, REASON = {}, {}, {}
    for name, stem in {**RUNS_4B, **RUNS_8B}.items():
        judge = "agnes" if name in RUNS_4B else "gemini-flash-high"
        V[name], _ = verdicts(RUNS / f"judged-{stem}-{judge}.jsonl")
        A[name] = {r["id"]: r for r in load(RUNS / f"{stem}.jsonl")}
        REASON[name] = {}
        for r in load(RUNS / f"judged-{stem}-{judge}.jsonl"):  # 与 verdicts() 同序覆盖：取最后一条有效判定的理由
            if r.get("parsed_ok") and r.get("verdict") and r["verdict"].get("hallucinated") is not None:
                REASON[name][r["id"]] = r["verdict"].get("reason")
            elif r["id"] in V[name] and r["id"] not in REASON[name]:
                REASON[name][r["id"]] = None
    ret = {r["id"]: r for r in load(RUNS / RETRIEVAL["BM25+重排(N=50)"])}
    ret_bm25 = {r["id"]: r for r in load(RUNS / RETRIEVAL["BM25"])}
    gate = {g["id"] for g in load(RUNS / GATE_V6) if g["verdict"] == "是"}
    pats = {k: re.compile(v) for k, v in RULES_V7.items()}
    rule_hit = {i: [k for k, p in pats.items() if p.search(d["question"].strip())] for i, d in dev.items()}
    replay = {"V6": {i: ["门控判「是」"] for i in gate}, "V7": {i: h for i, h in rule_hit.items() if h}}

    def lab(ver, i):
        v = V[ver][i]
        return "幻觉（兼拒答）" if v["hallucinated"] and v["refused"] else label(v)

    def pool(va, vb, la, lb, same_answer=False):
        if vb in replay:
            return [i for i in replay[vb] if i in V[va] and label(V[va][i]) == la]
        ids = [i for i in dev if i in V[va] and i in V[vb] and label(V[va][i]) == la and label(V[vb][i]) == lb]
        if same_answer:  # 回答一字不差、幻觉标签不同；池不再按前后标签细分
            ids = [i for i in dev if i in V[va] and i in V[vb] and A[va][i]["answer"] == A[vb][i]["answer"]
                   and V[va][i]["hallucinated"] != V[vb][i]["hallucinated"]]
        return ids

    def retrieved(ver, i):
        x = A[ver].get(i, {}).get("x_retrieved")
        return "、".join(dict.fromkeys(t[1] for t in x)) if x else None

    out = ["# Badcase 画廊：按幻觉类型与实验版本", "",
           "> 本文件由 `python analysis/badcase_gallery.py` 从 `eval/runs/` 里已有的作答与裁判判定生成，**请勿手改**；"
           "`python scripts/check_site_numbers.py` 会重新生成一份并逐字比对。标签是本地 LLM 裁判的代理判定（口径同报告 §2：「答对」= 作答且未判幻觉）。",
           "> 选例方法：每类先用规则圈出候选池（版本对 × 判定转移，池大小见各条标题），再人工挑代表性样本。",
           "> 「成因」是整理时逐条阅读回答、裁判理由和检索词条后单人写的，未经第二人复核；标「推测」的没有单独验证。"
           "回答超过 " + str(MAXLEN) + " 字的截断显示。", "",
           "| # | 类别 | 版本对 | 样本数 |", "|---|---|---|---|"]
    body, total = [], 0
    for ci, cat in enumerate(CATS, 1):
        rows = []
        for c in cat["cases"]:
            if len(c) == 4:
                (qid, la, lb, why), (va, vb) = c, cat["pair"]
            else:
                qid, va, vb, la, lb, why = c
            same = cat["key"] == "judge"
            ids = pool(va, vb, la, lb, same)
            if qid not in ids or (same and (label(V[va][qid]), label(V[vb][qid])) != (la, lb)):
                got = (label(V[va][qid]) if qid in V[va] else "无判定",
                       "回放触发" if vb in replay and qid in replay[vb] else (label(V[vb][qid]) if vb in V and qid in V[vb] else "无判定"))
                sys.exit(f"{cat['key']} / {qid}：预期 {va} {la} → {vb} {lb}{'（回答相同）' if same else ''}，数据里是 {got}，请更新 CATS")
            rows.append((qid, va, vb, la, lb, why, len(ids)))
        total += len(rows)
        vs = sorted({f"{r[1]} → {r[2]}" for r in rows})
        out.append(f"| {ci} | [{cat['title'].split('（')[0]}](#{ci}-{cat['key']}) | {'；'.join(vs)} | {len(rows)} |")
        body += ["", f"<a id=\"{ci}-{cat['key']}\"></a>", f"## {ci}. {cat['title']}", "", cat["note"]]
        for k, (qid, va, vb, la, lb, why, n) in enumerate(rows, 1):
            d = dev[qid]
            same = "，回答一字不差、幻觉标签不同" if cat["key"] == "judge" else ""
            body += ["", f"### {ci}.{k} {qid}（{group_of(d)}；{va} {la} → {vb} {lb}{same}；同类候选 {n} 题）", "",
                     f"- **问题**：{esc(d['question'], 400)}", f"- **参考答案**：{esc(d['gold'][0], 120)}"]
            if d["source"] == "chinese_simpleqa" and qid in ret and (retrieved(va, qid) or retrieved(vb, qid)):
                r, b = ret[qid], ret_bm25[qid]
                rk = lambda x: f"第 {x} 段" if x else "未进前 20 段"  # noqa: E731
                body.append(f"- **检索（标准答案首次出现的位置）**：BM25 {rk(b['rank'])}；重排后 {rk(r['rank'])}"
                            f"{'；BM25 前 50 段里都没有' if not r.get('bm25_rank') else ''}")
            body += ["", f"| | {NAMES.get(va, va)} | {NAMES.get(vb, vb)} |", "|---|---|---|"]
            if vb in replay:
                body += [f"| 判定 | {lab(va, qid)} | 拒答（触发：{'、'.join(replay[vb][qid])}） |",
                         f"| 回答 | {esc(A[va][qid]['answer'])} | （离线回放：改答「我不知道」） |",
                         f"| 裁判理由 | {esc(REASON[va].get(qid) or '（宽松补解析，无理由字段）', 120)} | — |"]
            else:
                body += [f"| 判定 | {lab(va, qid)} | {lab(vb, qid)} |",
                         f"| 回答 | {esc(A[va][qid]['answer'])} | {esc(A[vb][qid]['answer'])} |",
                         f"| 裁判理由 | {esc(REASON[va].get(qid) or '（宽松补解析，无理由字段）', 120)} | "
                         f"{esc(REASON[vb].get(qid) or '（宽松补解析，无理由字段）', 120)} |"]
                ra, rb = retrieved(va, qid), retrieved(vb, qid)
                if ra or rb:
                    body.append(f"| 检索到的词条 | {esc(ra or '（无检索）', 120)} | {esc(rb or '（无检索）', 120)} |")
            body += ["", f"**成因**：{why}"]
    out.insert(out.index("| # | 类别 | 版本对 | 样本数 |"), f"共 {len(CATS)} 类、{total} 条。")
    out.insert(out.index("| # | 类别 | 版本对 | 样本数 |"), "")
    text = "\n".join(out + body) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"已生成 {Path(a.out).resolve().relative_to(ROOT) if Path(a.out).resolve().is_relative_to(ROOT) else a.out}：{len(CATS)} 类、{total} 条")


if __name__ == "__main__":
    main()
