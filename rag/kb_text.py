"""建库和检索共用的文本处理：繁转简、清洗、切段、分词、词哈希。

建库（build_kb.py）和查询（search.py）必须走同一套 tokens()，否则词对不上。
"""
import hashlib, re

import jieba
from opencc import OpenCC

jieba.setLogLevel(60)
_cc = OpenCC("t2s")

# 正文到这些小节标题为止，后面是参考文献、外链、分类等，不进库
TAIL_HEADS = {"参考文献", "参考资料", "外部链接", "注释", "脚注", "参见", "延伸阅读", "相关条目",
              "图片", "图集", "引用", "来源", "外部连结", "扩展阅读", "相关链接"}
ART_CAP = 8000      # 单篇正文最多取多少字（p90≈3600，取 8000 保住约 88% 的字）
PASSAGE_MAX = 350   # 每段最多多少字（不含标题）
SENT_SPLIT = re.compile(r"(?<=[。！？；])")

STOP = set("的 了 是 在 和 与 及 或 等 也 有 为 被 这 那 其 之 于 以 而 并 中 对 上 下 由 从 把 将 就 都 又 还"
           " 一个 什么 哪个 哪里 哪些 哪 谁 多少 几 吗 呢 吧 啊 请问 是否 如何 怎么 为什么 是什么 叫什么 称为 被称为"
           " 一 个 位 第 该 此 及其 以及 他 她 它 他们 我们 你 我".split())
WORD = re.compile(r"\w")


def t2s(s):
    return _cc.convert(s)


def clean(text):
    """繁转简后截掉尾部小节，限长。"""
    lines = t2s(text).split("\n")
    out, n = [], 0
    for ln in lines:
        s = ln.strip()
        if s in TAIL_HEADS:
            break
        if not s:
            continue
        out.append(s)
        n += len(s)
        if n >= ART_CAP:
            break
    return out


def chunk(lines):
    """按行累加成段，单行过长再按句切，句子过长硬切。"""
    pieces = []
    for s in lines:
        if len(s) <= PASSAGE_MAX:
            pieces.append(s)
            continue
        for sent in SENT_SPLIT.split(s):
            while len(sent) > PASSAGE_MAX:
                pieces.append(sent[:PASSAGE_MAX])
                sent = sent[PASSAGE_MAX:]
            if sent:
                pieces.append(sent)
    out, cur, n = [], [], 0
    for p in pieces:
        if cur and n + len(p) > PASSAGE_MAX:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += len(p)
    if cur:
        out.append("\n".join(cur))
    return out


def tokens(s):
    """分词并过滤：去标点空白、去停用词、英文转小写。输入应已是简体。"""
    return [w for w in (t.lower() for t in jieba.lcut(s)) if WORD.search(w) and w not in STOP]


def h64(w):
    return int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")
