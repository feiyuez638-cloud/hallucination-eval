"""核对作品展示页 site/index.html 里的每个数字；英文版文档和一页纸摘要里的数字也一并核对。

  python scripts/check_site_numbers.py        # 核对，有问题时列出并以非零状态退出
  python scripts/check_site_numbers.py -v     # 另外逐条打印页面上每个数字的来源和期望值

页面里的数字必须标注来源，二选一：
- <span data-k="键">数字</span>：键对应的值从 reproduce_metrics.py（--json 文件与打印输出）、judge_agreement.py、
  check_sql.py 的输出取得，按报告的写法格式化（百分数一位小数、p 值同 reproduce_metrics.fmt_p、κ 两位小数），
  与页面文字逐字比较。键的含义见 facts() 里的注释。
- <span data-r="原文片段">数字</span>：片段必须原样出现在中文报告或 README.md 里，并且包含这个数字。
  只用于脚本不输出的少数事实（如留出集题数）。
其余可见文字（含 alt / aria-label / title / meta description）去掉模型名、版本名等标识符后不许再出现数字，防止漏标。
页面里的本地链接、图片和页内锚点必须存在。

另外核对 report/Hallucination-Evaluation-Report.md、README.en.md、docs/一页纸摘要.md：其中出现的每个数字
都必须在中文报告或 README.md 里出现过（这些文档只换语言或压缩，不产生新数字）。

面试材料（docs/面试讲解稿.md、docs/项目深挖40问.md、slides/项目讲解.md）更严：
- 每个数字都必须在中文报告里出现过（只认报告，不认 README）；百分数要连「%」一起、p 值要连「p=」「p<」一起原样出现；
  题号、Marp 图片尺寸、列表序号、「1 分钟版」这类结构性数字先去掉再查；
- 本地链接和图片必须存在；40 问恰好 40 题、幻灯片不超过 12 页、讲解稿含 Mermaid 图。
analysis/badcase画廊.md 是脚本输出，回答原文里的数字不在报告里，所以改为重新运行 analysis/badcase_gallery.py
生成一份并逐字比对。以上新文件和画廊都不许出现邮箱、手机号这类联系方式。

只用标准库，不调用任何模型或 API。
"""
import argparse, ast, json, re, subprocess, sys, tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from reproduce_metrics import ROOT, fmt_p, holm, pct, wilson

SCRIPTS, SITE = ROOT / "scripts", ROOT / "site"
PAGE = SITE / "index.html"
ZH = [ROOT / "report/大模型幻觉评测报告.md", ROOT / "README.md"]
DERIVED_DOCS = [ROOT / "report/Hallucination-Evaluation-Report.md", ROOT / "README.en.md", ROOT / "docs/一页纸摘要.md"]
REPORT = ZH[0]
INTERVIEW_DOCS = [ROOT / "docs/面试讲解稿.md", ROOT / "docs/项目深挖40问.md", ROOT / "slides/项目讲解.md"]
GALLERY, GALLERY_SCRIPT = ROOT / "analysis/badcase画廊.md", ROOT / "analysis/badcase_gallery.py"
# 结构性数字：问题编号、Marp 图片尺寸、有序列表序号、讲稿时长 / 题数 / 页数、图号、章节号
STRUCT = re.compile(r"Q\d+|\b[wh]:\d+|^\s*\d+\.\s|\d+\s*分钟|\d+\s*问|\d+\s*页|图\s*\d+|§\d+(?:\.\d+)*", re.M)
PCT = re.compile(r"\d+(?:\.\d+)?%")
PVAL = re.compile(r"p\s*[=<]\s*\d+(?:\.\d+)?")
CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|(?<!\d)1[3-9]\d{9}(?!\d)|微信|wechat|QQ\s*[:：]?\s*\d", re.I)
MDLINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
# 英文里换了写法的数字：英文写法 -> 中文原文片段（片段必须出现在中文报告里）
EN_REWRITES = {"1.38": "138 万篇", "3.59": "359 万段"}
# 页面正文里允许出现的、不是指标的带数字标识符
IDENT = re.compile(r"(?<![A-Za-z0-9])(?:Qwen3-[48]B|b0-8B|V[1-8]-8B|V[1-8]|p0|[48]B|BM25|bge-reranker-v2-m3|"
                   r"agnes-2\.5-flash|gemini-3\.8-flash-high|gemini-3\.1-pro-high|dev_v1|holdout_v1)(?![A-Za-z0-9])")
NUM = re.compile(r"\d+(?:\.\d+)?")


def run(script, *args, check=True):
    p = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, encoding="utf-8")
    if check and p.returncode:
        sys.exit(f"{script} 运行失败：\n{p.stderr}")
    return p.stdout


def facts():
    """键 -> 页面上应写的字符串。"""
    tmp = Path(tempfile.mkdtemp()) / "m.json"
    out = run("reproduce_metrics.py", "--json", str(tmp))
    js = json.loads(tmp.read_text(encoding="utf-8"))
    F = {}

    # 开发集规模：打印输出第一段「开发集 dev_v1：450 题， {...}」
    m = re.search(r"开发集 dev_v1：(\d+) 题， (\{.*\})", out)
    grp = ast.literal_eval(m.group(2))
    F["dev.n"], F["dev.SimpleQA"] = m.group(1), str(grp["SimpleQA"])
    F["dev.HalluQA"] = str(grp["Knowledge"] + grp["Misleading"] + grp["Misleading-hard"])

    # 表 1～3：<版本>.<指标>（全部题）、<版本>.<题型>.<指标>
    for v, r in {**js["4b"], **js["8b"]}.items():
        for g, s in r["groups"].items():
            pre = v if g == "全部" else f"{v}.{g}"
            for k in ("n", "hal", "ref", "ok", "both"):
                F[f"{pre}.{k}"] = str(s[k])
            for k in ("hr", "hr_ex", "ref_rate"):
                F[f"{pre}.{k}"] = pct(s[k])

    # 相邻版本配对：pair.<a>-><b>.<指标>；only_*_hal 由转移表得到（标签「幻觉」⇔ hallucinated 为真）
    pairs = js["pairs"]
    keys = [(k, m) for k in pairs for m in ("p_hal", "p_ok")]
    adj = dict(zip(keys, holm([pairs[k][m] for k, m in keys])))
    F["holm.m"] = str(len(keys))
    for k, p in pairs.items():
        t, pre = p["trans"], f"pair.{k}"
        oa = t["幻觉->答对"] + t["幻觉->拒答"]
        ob = t["答对->幻觉"] + t["拒答->幻觉"]
        F.update({f"{pre}.common": str(p["common"]), f"{pre}.hal_a": str(p["hal"][0]), f"{pre}.hal_b": str(p["hal"][1]),
                  f"{pre}.ok_a": str(p["ok"][0]), f"{pre}.ok_b": str(p["ok"][1]),
                  f"{pre}.only_a_hal": str(oa), f"{pre}.only_b_hal": str(ob), f"{pre}.disc_hal": str(oa + ob),
                  f"{pre}.p_hal": fmt_p(p["p_hal"]), f"{pre}.p_ok": fmt_p(p["p_ok"]),
                  f"{pre}.holm_hal": fmt_p(adj[(k, "p_hal")]), f"{pre}.holm_ok": fmt_p(adj[(k, "p_ok")])})

    # 分题型配对：打印输出「分题型配对」表，gpair.<a>-><b>.<题型>.only_a / only_b / p
    sec = out.split("### 分题型配对", 1)[1].split("\n###", 1)[0]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    heads = [c.strip() for c in lines[0].strip("|").split("|")][1:]
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        a, b = cells[0].split(" → ")
        for g, c in zip(heads, cells[1:]):
            x, y, p = re.match(r"(\d+) / (\d+)，(p[=<][\d.]+)", c).groups()
            F.update({f"gpair.{a}->{b}.{g}.only_a": x, f"gpair.{a}->{b}.{g}.only_b": y, f"gpair.{a}->{b}.{g}.p": p})

    # V6 / V7 离线回放
    for v in ("v6", "v7"):
        r, V = js[v], v.upper()
        F.update({f"{V}.blocked": str(r["blocked"]), f"{V}.false_refuse": str(r["false_refuse"]), f"{V}.sqa_trig": str(r["sqa_trig"])})
        for k in ("hal", "ref", "ok"):
            F[f"{V}.after.{k}"] = str(r["after"][k])
        for k in ("hr", "hr_ex", "ref_rate"):
            F[f"{V}.after.{k}"] = pct(r["after"][k])

    # 裁判验收集：acc.n 条数；acc.lb = agnes-2.5-flash 一致率的 95% Wilson 下界
    F["acc.n"] = re.search(r"## 裁判验收集（(\d+) 条", out).group(1)
    a = js["acceptance"]["agnes-2.5-flash"]
    F["acc.lb"] = pct(wilson(a["strict_ok"] + a["lenient_ok"], a["strict"] + a["lenient"])[0])

    # 检索命中率（SimpleQA）：ret.bm25.* / ret.rerank50.*
    for name, key in (("BM25", "bm25"), ("BM25+重排(N=50)", "rerank50")):
        r = js["retrieval"][name]
        for k in ("1", "5", "20", "wiki5", "nonwiki5"):
            F[f"ret.{key}.{k}"] = pct(r[k])
    hit = js["retrieval_vs_answer"]["命中"]
    F["v3hit.n"], F["v3hit.ok_rate"] = str(hit["n"]), pct(hit["答对"] / hit["n"])

    # 按出处是否含维基分组：wiki.sqa_n（含维基链接的 SimpleQA 题数）、wiki.<版本>.wiki|nonwiki.ok_rate
    F["wiki.sqa_n"] = re.search(r"出处含维基链接 (\d+) 题", out).group(1)
    for v, d in js["wiki_split"].items():
        for w, s in d.items():
            F[f"wiki.{v}.{w}.ok_rate"] = pct(s["ok"] / s["n"])

    # 裁判一致性：judge_agreement.py 打印输出，ja.<cross|agnes|flash>.n / .<hal|ref>.k|ci|agree
    cur = None
    for l in run("judge_agreement.py").splitlines():
        if (m := re.match(r"## (跨裁判|同裁判重测：(agnes|gemini)).*（n=(\d+)", l)):
            cur = "cross" if m.group(1) == "跨裁判" else {"agnes": "agnes", "gemini": "flash"}[m.group(2)]
            F[f"ja.{cur}.n"] = m.group(3)
        elif cur and (m := re.match(r"\| (hallucinated|refused) \| \d+/\d+ \| ([\d.]+%) \[[^]]*\] \| ([\d.]+) \[([\d.]+), ([\d.]+)\]", l)):
            lab = m.group(1)[:3]
            F[f"ja.{cur}.{lab}.agree"] = m.group(2)
            F[f"ja.{cur}.{lab}.k"] = f"{float(m.group(3)):.2f}"
            F[f"ja.{cur}.{lab}.ci"] = f"{float(m.group(4)):.2f}–{float(m.group(5)):.2f}"
        elif l.startswith("## "):
            cur = None

    # SQL 与 Python 比对：check_sql.py 的最后一行（不一致时它以非零状态退出，这里照样读结果）
    m = re.search(r"共 (\d+) 项，一致 (\d+) 项，不一致 (\d+) 项", run("check_sql.py", check=False))
    F["sql.n"], F["sql.ok"] = m.group(1), m.group(2)
    return F


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip, self.cur = 0, None
        self.tagged, self.free, self.refs, self.ids = [], [], [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if a.get(k):
                self.refs.append((self.getpos()[0], a[k]))
        for k in ("alt", "aria-label", "title"):
            if a.get(k):
                self.free.append((self.getpos()[0], a[k]))
        if tag == "meta" and a.get("name") == "description":
            self.free.append((self.getpos()[0], a.get("content", "")))
        if tag == "span" and ("data-k" in a or "data-r" in a):
            self.cur = {"line": self.getpos()[0], "k": a.get("data-k"), "r": a.get("data-r"), "text": ""}

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag == "span" and self.cur:
            self.tagged.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cur:
            self.cur["text"] += data
        elif not self.skip:
            self.free.append((self.getpos()[0], data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    F = facts()
    zh = "\n".join(p.read_text(encoding="utf-8") for p in ZH)
    pg = Page()
    pg.feed(PAGE.read_text(encoding="utf-8"))
    bad = []

    n_k = n_r = 0
    for t in pg.tagged:
        text, where = t["text"].strip(), f"site/index.html:{t['line']}"
        if t["k"]:
            n_k += 1
            want = F.get(t["k"])
            if want is None:
                bad.append(f"{where} 未知的键 {t['k']!r}")
            elif text != want:
                bad.append(f"{where} {t['k']}：页面写 {text!r}，脚本输出为 {want!r}")
            if a.v:
                print(f"  {'✓' if text == want else '✗'} {t['k']:<40} {text}")
        else:
            n_r += 1
            if t["r"] not in zh or text not in t["r"] or not NUM.search(text):
                bad.append(f"{where} data-r 片段 {t['r']!r} 不在中文报告 / README 里，或不包含 {text!r}")
            if a.v:
                print(f"  {'✓' if t['r'] in zh else '✗'} 原文「{t['r']}」 {text}")

    n_free = 0
    for line, s in pg.free:
        for m in NUM.finditer(IDENT.sub("", s)):
            n_free += 1
            bad.append(f"site/index.html:{line} 未标注来源的数字 {m.group()!r}：…{s.strip()[:60]}…")

    n_ref = 0
    for line, ref in pg.refs:
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        n_ref += 1
        path, _, frag = ref.partition("#")
        if path and not (SITE / unquote(path)).resolve().exists():
            bad.append(f"site/index.html:{line} 链接目标不存在：{ref}")
        if not path and frag not in pg.ids:
            bad.append(f"site/index.html:{line} 页内锚点不存在：#{frag}")

    zh_nums = set(NUM.findall(zh))
    for k, snip in EN_REWRITES.items():
        if snip not in zh:
            bad.append(f"EN_REWRITES：{snip!r} 不在中文报告里")
    doc_stats = []
    for p in DERIVED_DOCS:
        if not p.exists():
            bad.append(f"{p.relative_to(ROOT)} 不存在")
            continue
        nums = NUM.findall(p.read_text(encoding="utf-8"))
        extra = sorted({x for x in nums if x not in zh_nums and x not in EN_REWRITES})
        doc_stats.append((p.relative_to(ROOT), len(nums), len(set(nums))))
        for x in extra:
            bad.append(f"{p.relative_to(ROOT)}：数字 {x!r} 没有出现在中文报告 / README 里")

    rep = REPORT.read_text(encoding="utf-8")
    rep_nums, rep_flat = set(NUM.findall(rep)), re.sub(r"\s+", "", rep)
    for p in INTERVIEW_DOCS:
        rel = p.relative_to(ROOT)
        if not p.exists():
            bad.append(f"{rel} 不存在")
            continue
        raw = p.read_text(encoding="utf-8")
        text = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)  # Marp 头部（样式里的字号不是指标）
        n_struct = len(STRUCT.findall(text))
        body = STRUCT.sub(" ", text)
        nums = NUM.findall(body)
        for x in sorted({x for x in nums if x not in rep_nums}):
            bad.append(f"{rel}：数字 {x!r} 没有出现在中文报告里")
        toks = PCT.findall(body) + PVAL.findall(body)
        for t in sorted({t for t in toks if re.sub(r"\s+", "", t) not in rep_flat}):
            bad.append(f"{rel}：{t!r} 没有原样出现在中文报告里")
        for ref in MDLINK.findall(raw):
            path = ref.split("#")[0]
            if path and not ref.startswith(("http://", "https://")) and not (p.parent / unquote(path)).exists():
                bad.append(f"{rel}：链接目标不存在：{ref}")
        if p.name == "项目深挖40问.md" and (k := len(re.findall(r"^\*\*Q\d+\.", raw, re.M))) != 40:
            bad.append(f"{rel}：应为 40 题，实际 {k} 题")
        if p.name == "项目讲解.md" and (k := len(re.findall(r"^---$", text, re.M)) + 1) > 12:
            bad.append(f"{rel}：幻灯片 {k} 页，超过 12 页")
        if p.name == "面试讲解稿.md" and "```mermaid" not in raw:
            bad.append(f"{rel}：缺少 Mermaid 白板图")
        doc_stats.append((rel, len(nums) + len(toks), n_struct))

    tmp = Path(tempfile.mkdtemp()) / "gallery.md"
    run_gal = subprocess.run([sys.executable, str(GALLERY_SCRIPT), "--out", str(tmp)], capture_output=True, text=True, encoding="utf-8")
    gal_ok = run_gal.returncode == 0 and GALLERY.exists() and tmp.read_text(encoding="utf-8") == GALLERY.read_text(encoding="utf-8")
    if run_gal.returncode:
        bad.append(f"badcase_gallery.py 运行失败：{run_gal.stderr.strip()[-300:]}")
    elif not gal_ok:
        bad.append(f"{GALLERY.relative_to(ROOT)} 与脚本重新生成的结果不一致（请运行 python analysis/badcase_gallery.py）")
    for p in INTERVIEW_DOCS + [GALLERY, GALLERY_SCRIPT]:
        if p.exists():
            for m in CONTACT.finditer(p.read_text(encoding="utf-8")):
                bad.append(f"{p.relative_to(ROOT)}：疑似联系方式 {m.group()!r}")

    print(f"展示页 data-k 数字 {n_k} 处（{len({t['k'] for t in pg.tagged if t['k']})} 个不同的键），"
          f"data-r 原文数字 {n_r} 处，未标注的数字 {n_free} 处；本地链接与锚点 {n_ref} 个")
    for rel, n, u in doc_stats:
        if rel.parts[0] in ("slides",) or rel.name in ("面试讲解稿.md", "项目深挖40问.md"):
            print(f"{rel}：数字 {n} 处（另有结构性编号 {u} 处已排除），逐个在中文报告里查找")
        else:
            print(f"{rel}：数字 {n} 处（{u} 个不同的值），逐个在中文报告 / README 里查找")
    if run_gal.returncode == 0:
        print(f"{GALLERY.relative_to(ROOT)}：{'与' if gal_ok else '不同于'} badcase_gallery.py 重新生成的结果{'逐字一致' if gal_ok else ''}")
    if bad:
        print(f"\n发现 {len(bad)} 个问题：")
        for b in bad:
            print("  ✗", b)
        sys.exit(1)
    print("全部一致。")


if __name__ == "__main__":
    main()
