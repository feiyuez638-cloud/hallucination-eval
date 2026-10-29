"""画报告插图，输出到 report/figures/（只读 eval/runs/ 已有判定，不调模型）。

  python scripts/make_figures.py

数字与 scripts/reproduce_metrics.py 同源（直接复用其函数）。需要 matplotlib；中文字体自动从系统里找。
"""
import re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_metrics as rm  # noqa: E402

OUT = rm.ROOT / "report/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def setup():
    for name in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "PingFang SC"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "font.size": 10})


def fig_versions(r4):
    names = list(r4)
    g = [r4[n]["groups"]["全部"] for n in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(names))
    w = 0.38
    for off, key, color, lab in ((-w / 2, "hr", BLUE, "HR（拒答算有效）"), (w / 2, "hr_ex", ORANGE, "HR（拒答剔除）")):
        bars = a.bar([i + off for i in x], [s[key] * 100 for s in g], w - 0.04, color=color, label=lab)
        for bar, s in zip(bars, g):
            a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{s[key] * 100:.1f}", ha="center",
                   fontsize=8, color=MUTED)
    a.set_xticks(list(x), names)
    a.set_ylabel("幻觉率 %")
    a.set_ylim(0, 85)
    a.set_title("幻觉率（越低越好）", loc="left", fontsize=11)
    a.legend(frameon=False, loc="upper right")
    bottom = [0] * len(names)
    for key, color, lab in (("ok", BLUE, "答对"), ("ref_only", AQUA, "只拒答"), ("hal", ORANGE, "幻觉")):
        vals = [s["ref"] - s["both"] if key == "ref_only" else s[key] for s in g]
        b.bar(names, vals, 0.6, bottom=bottom, color=color, label=lab, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            if v >= 25:
                b.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=8, color="white")
        bottom = [p + v for p, v in zip(bottom, vals)]
    b.set_ylabel("题数（有判定）")
    b.set_title("结果构成：答对 / 只拒答 / 幻觉", loc="left", fontsize=11)
    b.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Qwen3-4B 各版本（dev_v1 450 题，裁判 agnes-2.5-flash）", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_versions.png", dpi=160)


def fig_groups(r4):
    names = list(r4)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ends = []
    for grp, color in (("SimpleQA", BLUE), ("Knowledge", ORANGE), ("Misleading", AQUA), ("Misleading-hard", YELLOW)):
        ys = [r4[n]["groups"][grp]["hr"] * 100 for n in names]
        ax.plot(names, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=grp)
        ends.append([ys[-1], f"{grp} {ys[-1]:.1f}%"])
    ends.sort()
    for i in range(1, len(ends)):  # 末端标签相距不足 4 个百分点时往上错开
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 4)
    for y, text in ends:
        ax.text(len(names) - 1 + 0.12, y, text, va="center", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=9)
    ax.set_xlim(-0.3, len(names) - 0.2)
    ax.set_ylabel("HR（拒答算有效）%")
    ax.set_ylim(0, 100)
    ax.set_title("分题型幻觉率：检索压住了 SimpleQA，V3 起 HalluQA 三类都明显更高", loc="left", fontsize=11)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(OUT / "fig2_by_type.png", dpi=160)


def fig_replay(v6, v7):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = ["V6 门控（模型自判）", "V7 规则特征"]
    blocked = [v6["blocked"], v7["blocked"]]
    false = [v6["false_refuse"], v7["false_refuse"]]
    y = [1, 0]
    h = 0.36
    b1 = ax.barh([i + h / 2 for i in y], blocked, h - 0.04, color=BLUE, label="拦住的幻觉")
    b2 = ax.barh([i - h / 2 for i in y], false, h - 0.04, color=ORANGE, label="误拒的答对题")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va="center",
                    fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("题数（用 V4 判定离线回放，触发即改答「我不知道」）")
    ax.set_title("拒答式拦截：代价（误拒）与收益（拦住幻觉）", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gate_replay.png", dpi=160)


def fig_wiki(ws):
    names = list(ws)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    for off, key, color, lab in ((-w / 2, "wiki", BLUE, "出处含维基"), (w / 2, "nonwiki", ORANGE, "出处不含维基")):
        vals = [ws[n][key]["ok"] / ws[n][key]["n"] * 100 for n in names]
        bars = ax.bar([i + off for i in range(len(names))], vals, w - 0.04, color=color,
                      label=f"{lab}（n≈{ws[names[0]][key]['n']}）")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(names)), ["V2 无检索", "V3 BM25", "V4 BM25+重排"])
    ax.set_ylabel("SimpleQA 答对率 %")
    ax.set_ylim(0, 95)
    ax.set_title("出处为维基的题从检索中获益更多（知识库也是维基）", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_wiki_split.png", dpi=160)


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = {r["id"]: r for r in rm.load(rm.ROOT / "data/dev/dev_v1.jsonl")}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        r4 = rm.run_block(dev, rm.RUNS_4B)
        gate = {g["id"] for g in rm.load(rm.RUNS / rm.GATE_V6) if g["verdict"] == "是"}
        v6 = rm.replay(dev, r4["V4"]["v"], gate, "")
        pat = [re.compile(v) for v in rm.RULES_V7.values()]
        v7 = rm.replay(dev, r4["V4"]["v"], {i for i, d in dev.items() if any(p.search(d["question"].strip()) for p in pat)}, "")
        ws = rm.wiki_split(dev, {k: r4[k] for k in ("V2", "V3", "V4")})
    fig_versions(r4)
    fig_groups(r4)
    fig_replay(v6, v7)
    fig_wiki(ws)
    print("图已写入", OUT)


if __name__ == "__main__":
    main()
