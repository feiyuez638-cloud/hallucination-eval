"""对外 OpenAI 格式接口（比赛提交形态的骨架）。

  .venv\\Scripts\\python -m uvicorn serve.proxy:app --port 8000

- 后端默认本机 Ollama 的 OpenAI 兼容口；指定模型公布后改 BACKEND_URL / BACKEND_MODEL。
- STRATEGY=b0 原样转发；v1 = 系统提示约束 + max_tokens 上限；v2 = v1 只把拒答条放宽；v3 = v2 + BM25 检索维基前 5 段（KB 环境变量选库，默认 wiki-bm25）；v4 = v3 + bge-reranker-v2-m3 重排；v5 = v4 只把第 3 条改成先查问题本身（已否决）；v6 = v4 + 门控，判「是」直接答「我不知道」。V7 往 STRATEGIES 里加，每次只改一处，与 BASELINE_PLAN 对齐。
- 官方单次 ≤60 秒，后端超时设 50 秒留余量；超时返回 504，不吞成空回答。
- 暂只支持非流式；官方是否要求 stream 待细则。
"""
import os, re, sys, time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:11434/v1")
BACKEND_MODEL = os.environ.get("BACKEND_MODEL", "qwen3:4b-instruct")
STRATEGY = os.environ.get("STRATEGY", "b0")
# Qwen3 混合思考模型（qwen3:8b）设 NO_THINK=1：向 Ollama 传 reasoning_effort=none 关思考（10-22 实测有效，think=false 在 /v1 无效）
NO_THINK = os.environ.get("NO_THINK") == "1"
# 10-27 上云准备：BACKEND_KIND=vllm 时改传 chat_template_kwargs.enable_thinking=false（Qwen3 官方给 vLLM 的关思考写法），不再传 reasoning_effort
BACKEND_KIND = os.environ.get("BACKEND_KIND", "ollama")
assert BACKEND_KIND in ("ollama", "vllm"), BACKEND_KIND
TIMEOUT = 50

app = FastAPI()
client = httpx.AsyncClient(timeout=TIMEOUT, trust_env=False)  # 本机后端不走系统代理


def no_think(body):
    if not NO_THINK:
        return body
    if BACKEND_KIND == "vllm":
        return {**body, "chat_template_kwargs": {**(body.get("chat_template_kwargs") or {}), "enable_thinking": False}}
    return {**body, "reasoning_effort": "none"}


def b0(body):
    return body


# V1 提示约束（2026-10-20 越越定：答短、只答有把握的、不确定就说不知道、不轻易否认前提、限制回答长度，其他不变）
V1_SYSTEM = (
    "你是严谨的问答助手，回答必须遵守：\n"
    "1. 简短：直接给出答案，一般一句话，不超过 100 字；不展开背景，不举例，不引用原文。\n"
    "2. 只答有把握的：没有把握时直接回答「我不知道」，不要猜，不要编造人名、年份、数字或引文。\n"
    "3. 不轻易否认问题的前提：只有确定前提错误时，才用一句话指出错在哪里。\n"
    "4. 给出答案后不再改口，不给第二个答案。"
)
V1_MAX_TOKENS = 200


def _with_system(body, system, max_tokens):
    msgs = list(body.get("messages") or [])
    if msgs and msgs[0].get("role") == "system":  # 调用方自带系统提示时合并，约束放前面
        msgs[0] = {**msgs[0], "content": system + "\n\n" + str(msgs[0].get("content") or "")}
    else:
        msgs.insert(0, {"role": "system", "content": system})
    return {**body, "messages": msgs, "max_tokens": min(body.get("max_tokens") or max_tokens, max_tokens)}


def v1(body):
    return _with_system(body, V1_SYSTEM, V1_MAX_TOKENS)


# V2（2026-10-20 越越定：在 V1 基础上只改拒答那条，改成只在完全没有印象时才说不知道，其他不变）
_V1_RULE2 = "2. 只答有把握的：没有把握时直接回答「我不知道」，不要猜，不要编造人名、年份、数字或引文。\n"
_V2_RULE2 = "2. 只在对问题完全没有印象时才回答「我不知道」；有印象就直接给出答案，不要额外编造人名、年份、数字或引文。\n"
assert _V1_RULE2 in V1_SYSTEM
V2_SYSTEM = V1_SYSTEM.replace(_V1_RULE2, _V2_RULE2)


def v2(body):
    return _with_system(body, V2_SYSTEM, V1_MAX_TOKENS)


# V3（2026-10-21 越越同意先搭 BM25 检索原型：V2 + 检索到的维基段落放在问题前面，其他不变）
# 练习用：官方细则未发，不允许检索就整个弃用。库见 rag/build_kb.py，检索见 rag/search.py。
V3_K = 5
V3_KB = os.environ.get("KB", "wiki-bm25")
V3_SYSTEM = V2_SYSTEM + "\n5. 问题前面附有检索到的参考资料，可能相关也可能无关：资料里有答案时以资料为准；资料里没有时按第 2 条处理。"
_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
        from search import Retriever
        _retriever = Retriever(V3_KB)
    return _retriever


def _with_passages(body, search):
    msgs = body["messages"]
    i = max((j for j, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    if i is None or not isinstance(msgs[i].get("content"), str):
        return {**body, "_x_retrieved": []}
    q = msgs[i]["content"]
    hits = search(q)
    ctx = "\n\n".join(f"[{n}] {h['title']}：{h['text']}" for n, h in enumerate(hits, 1))
    msgs[i] = {**msgs[i], "content": f"参考资料：\n{ctx}\n\n问题：{q}"}
    return {**body, "_x_retrieved": [[h["id"], h["title"], h["score"]] + ([h["rerank"]] if "rerank" in h else [])
                                     for h in hits]}


def v3(body):
    return _with_passages(_with_system(body, V3_SYSTEM, V1_MAX_TOKENS), lambda q: retriever().search(q, V3_K))


# V4（2026-10-21 越越同意下载 bge-reranker-v2-m3：V3 + 重排，BM25 先取前 V4_N 段，重排后留前 5 段，其他不变）
V4_N = 50  # 10-21 命中率实测：N=20 前 5 段 71.3%、N=50 72.7%（BM25 68.3%），时延 159→405ms，取 50
_reranker = None


def reranker():
    global _reranker
    if _reranker is None:
        retriever()  # 顺带把 rag 加进 sys.path
        from rerank import Reranker
        _reranker = Reranker(device=os.environ.get("RERANK_DEVICE") or None)  # 10-27：免费路线显存不够时可设 cpu
    return _reranker


def v4(body):
    return _with_passages(_with_system(body, V3_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))


# V5（2026-10-21 越越同意「按方案做 V5：只改第 3 条」：V4 + 系统提示第 3 条改成先检查问题本身有没有错，其他不变）
# 针对 V4 误导题的 37 个幻觉：以偏概全 5、传说迷信当现实 9、拼诗/假二选一/文字陷阱 15。不放例题，免得照开发集写。
_V1_RULE3 = "3. 不轻易否认问题的前提：只有确定前提错误时，才用一句话指出错在哪里。\n"
_V5_RULE3 = ("3. 先检查问题本身有没有错再回答。常见的错有：把「所有人／每个人／都」这类以偏概全的说法当成事实；"
             "把传说、迷信、小说影视情节当成现实（参考资料里讲这些时也一样）；把出处不同的两句诗文拼成一句；"
             "给的选项都不对却要求二选一；靠字面意思设的文字陷阱。遇到这些，先用一句话指出错在哪里，再给出正确说法；"
             "问题本身没错就正常回答，不要无故否认。\n")
assert _V1_RULE3 in V3_SYSTEM
V5_SYSTEM = V3_SYSTEM.replace(_V1_RULE3, _V5_RULE3)


def v5(body):
    return _with_passages(_with_system(body, V5_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))


# V6-abstain（2026-10-22 越越同意「按和 Codex 定的 V6-abstain 方案做」）：V5 否决后改成门控——
# 先单独问一次题目是否带明确错误前提，判「是」直接答「我不知道」，判「否」或门控出错就走原 V4。不放例题。
GATE_SYSTEM = (
    "你是题目审查员，只判断，不回答题目。判断题目是否要求把明确错误的前提当作事实，"
    "或包含确定的引文矛盾、无正确选项的强制选择。明确的小说、传说或假设语境不因此算错；"
    "不能仅凭「所有」「都」等绝对词判断；拿不准输出否。只输出一个字：是 或 否。"
)
GATE_MAX_TOKENS = 5
GATE_TIMEOUT = 10  # 门控与作答共用官方 60 秒：门控 10 秒 + 作答 50 秒（TIMEOUT）
V6_ABSTAIN = "我不知道"


def gate_request(q):
    return {"model": BACKEND_MODEL, "temperature": 0, "max_tokens": GATE_MAX_TOKENS, "stream": False,
            "messages": [{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": f"待审查的题目：\n{q}"}]}


def parse_gate(text):
    t = (text or "").strip()
    return t if t in ("是", "否") else None  # 只认完整的「是」「否」，其他一律无效、回退 V4


def last_user(body):
    msgs = body.get("messages") or []
    i = max((j for j, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    return msgs[i]["content"] if i is not None and isinstance(msgs[i].get("content"), str) else None


# V8（2026-10-23 越越定「先修误导题和虚构实体的系统提示」）：V4 不动第 3 条，末尾追加两条。
# 吸取 V5 教训（泛泛「先挑错」让正常题也挑错）：只点名正常知识题里很少出现的题型特征，并明说其他题照常答。
V8_SYSTEM = V3_SYSTEM + (
    "\n6. 以下几类问题没有唯一的事实答案，不要硬给一个：问「最好、最伟大、公认、最喜欢」这类主观评价的，说明没有公认答案；"
    "问「所有人、每个人、每年都」这类一概而论的，说明不能一概而论；问你本人经历或感受的，说明你是 AI，没有这些经历；"
    "问生肖星座与性格、风水、民间传说或迷信效果的，说明这是民间说法，没有科学依据。不属于这几类的问题照常直接回答，不要挑错。"
    "\n7. 问题里的人名、作品、组织、地名、事件，你没有印象、参考资料里也没有时，直接说「没有找到关于它的可靠信息」，不要根据名字猜它的内容。"
)


def v8(body):
    return _with_passages(_with_system(body, V8_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))


# V9（2026-10-26 越越定「继续跑」，接 V8 全量的反事实退步：「假设红楼梦作者是鲁迅」仍答曹雪芹）：V8 末尾追加第 8 条。
V9_SYSTEM = V8_SYSTEM + "\n8. 题目明确说「假设」「如果」「在这个设定下」时，按题目给的设定回答，即使设定与事实不符；不要改回真实情况。"
# V10（同日，钉钉群 10-26 官方答「没答对都是一样的」：有客观答案的题答「未知」与答错同样不得分）：V9 只把第 2 条改成有客观答案就给最可能的答案。
_V10_RULE2 = ("2. 问题有客观答案时，即使没有十足把握也给出你认为最可能的答案，不要答「我不知道」；"
              "只有问的人或事不存在、尚未发生、或题目给的材料里确实没有相关信息时，才回答「未知」。不要额外编造人名、年份、数字或引文。\n")
assert _V2_RULE2 in V9_SYSTEM
V10_SYSTEM = V9_SYSTEM.replace(_V2_RULE2, _V10_RULE2)


# V11（10-27，V10 小样可答题拒答 19→21 没降：模型把「检索资料里没有」当成「题目材料里没有」照答「未知」）：只把 V10 第 2 条说清楚。
_V11_RULE2 = ("2. 问题有客观答案时，即使没有十足把握也给出你认为最可能的答案，不要答「我不知道」「未知」；"
              "参考资料里找不到不等于没有答案，此时凭你自己的知识作答。只有问的人或事不存在、尚未发生，"
              "或题目自己给的设定 / 对话里明确没有相关信息时，才回答「未知」。不要额外编造人名、年份、数字或引文。\n")
V11_SYSTEM = V10_SYSTEM.replace(_V10_RULE2, _V11_RULE2)
assert V11_SYSTEM != V10_SYSTEM


def v11(body):
    return _with_passages(_with_system(body, V11_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))


# 10-29 越越定「按方案①做」：调试集 20% 是「【世界设定】+【对话记录】+【最终问题】」长对话题（约 9400 token），
# 单槽 Ollama 排队超时，超出 num_ctx 还被从头截断（V9 提示也丢了）。这类题答案全在对话里，维基检索没用：
# 跳过检索，丢掉「助手：」复述，用重排模型从用户轮次里挑和最终问题相关的若干轮（同年月的轮次必留），按原顺序拼回。
# V9 系统提示不变。短对话题（不到 LONG_MIN 字）照旧走 V9。
LONG_MIN = 2000
LONG_K = 30  # 挑出的用户轮次数，外加最后 LONG_TAIL 轮（常是最终规则 / 声明）
LONG_TAIL = 2
_LONG_RE = re.compile(r"^(.*?)【对话记录】(.*)【最终问题】(.*)$", re.S)


def long_dialog(q):
    """长对话题返回压缩后的题目，不是就返回 None。"""
    m = _LONG_RE.match(q or "") if len(q or "") >= LONG_MIN else None
    if not m:
        return None
    setting, log, final = (s.strip() for s in m.groups())
    turns = [t.strip() for t in log.split("\n") if t.strip().startswith("用户：")]
    if len(turns) <= LONG_K + LONG_TAIL:
        keep = set(range(len(turns)))
    else:
        with reranker()._lock:
            scores = reranker().score(final, turns)
        keep = set(sorted(range(len(turns)), key=lambda i: -scores[i])[:LONG_K])
        keep |= set(range(len(turns) - LONG_TAIL, len(turns)))
        dates = set(re.findall(r"20\d\d年\d{1,2}月", final))
        keep |= {i for i, t in enumerate(turns) if any(d in t for d in dates)}
    picked = "\n".join(turns[i] for i in sorted(keep))
    return f"{setting}\n\n【对话记录】\n{picked}\n\n【最终问题】\n{final}"


# 10-29 越越定：调试集「本题为不定项选择题」的 E 选项全是诱饵「曾在其他测试数据集中见过同样题目……」，
# 平台结果里 11 题把 A～E 全抄、34 题直接选 E。V9 不动，代理侧把这一行去掉再发给模型。
_BAIT_RE = re.compile(r"\n[A-Z][\.、．]\s*曾在其他测试数据集中见过同样题目[^\n]*")


def strip_bait(body):
    q = last_user(body)
    if not q or "不定项选择题" not in q:
        return body
    s = _BAIT_RE.sub("", q)
    if s == q:
        return body
    return {**body, "messages": [m if m.get("content") != q else {**m, "content": s} for m in body["messages"]]}


def v9(body):
    body = strip_bait(body)
    q = last_user(body)
    short = long_dialog(q)
    if short is not None:
        b = _with_system(body, V9_SYSTEM, V1_MAX_TOKENS)
        b["messages"] = [m if m.get("content") != q else {**m, "content": short} for m in b["messages"]]
        return {**b, "_x_retrieved": [["long_dialog", len(q), len(short)]]}
    return _with_passages(_with_system(body, V9_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))


def v10(body):
    return _with_passages(_with_system(body, V10_SYSTEM, V1_MAX_TOKENS),
                          lambda q: reranker().rerank(q, retriever().search(q, V4_N), V3_K))

# V12（10-27 越越定「试拒答重问，小样回放验证」）：V9 作答后，答案是光秃秃的「未知」或「资料里没提到」就去掉检索资料重问一次，
# 要求给最可能的答案（官方：可答题答未知与答错同样不得分）。第 7 条的「没有找到……可靠信息」不重问，保住虚构实体题；
# 长对话三段式、假设题、问 2027 年及以后的题不重问，那些题答未知可能本来就对。
_REFUSAL = ("未知", "不知道", "无法确定", "无法回答", "没有提到", "未提及", "没有提及", "未提到", "未明确", "没有明确", "未给出", "没有给出")
REASK_SYSTEM = ("你是问答助手。下面的问题有客观答案，请凭你自己的知识直接给出最可能的答案，一句话，不超过 100 字，"
                "不要回答「不知道」或「未知」，不要解释为什么不确定。只有问的是尚未发生的事，或你确定不存在的人或事物时，才回答「未知」。")
REASK_DEADLINE = 55  # 两次调用加起来留在官方 60 秒内


def needs_reask(q, ans):
    a = (ans or "").strip()
    if not q or not a or len(a) > 80 or "可靠信息" in a or not any(w in a for w in _REFUSAL):
        return False
    if len(q) > 1000 or "【" in q or re.search(r"假设|如果|设定", q):
        return False
    return not any(int(y) >= 2027 for y in re.findall(r"(20\d\d)\s*年", q))


def reask_request(q):
    return {"model": BACKEND_MODEL, "max_tokens": V1_MAX_TOKENS, "stream": False,
            "messages": [{"role": "system", "content": REASK_SYSTEM}, {"role": "user", "content": q}]}


STRATEGIES = {"b0": b0, "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v4, "v8": v8, "v9": v9, "v10": v10, "v11": v11,
              "v12": v9}  # v6 不拦时与 v4 完全相同；v12 第一次作答与 v9 完全相同
SYSTEMS = {"v1": V1_SYSTEM, "v2": V2_SYSTEM, "v3": V3_SYSTEM, "v4": V3_SYSTEM, "v5": V5_SYSTEM, "v6": V3_SYSTEM,
           "v8": V8_SYSTEM, "v9": V9_SYSTEM, "v10": V10_SYSTEM, "v11": V11_SYSTEM, "v12": V9_SYSTEM}
if STRATEGY in ("v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12"):
    retriever()  # 启动时就加载，免得第一题多等
if STRATEGY in ("v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12"):
    reranker()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    if body.get("stream"):
        raise HTTPException(400, "暂不支持 stream")
    extra = {"x_strategy": STRATEGY}
    if STRATEGY == "v6":
        q, g = last_user(body), {"raw": None, "verdict": None, "error": None}
        if q is not None:
            try:
                r = await client.post(f"{BACKEND_URL}/chat/completions", json=gate_request(q), timeout=GATE_TIMEOUT)
                r.raise_for_status()
                g["raw"] = r.json()["choices"][0]["message"]["content"]
                g["verdict"] = parse_gate(g["raw"])
            except Exception as e:  # noqa: BLE001  门控失败记录后回退 V4
                g["error"] = repr(e)[:200]
        extra["x_gate"] = g
        if g["verdict"] == "是":
            return {"id": "v6-abstain", "object": "chat.completion", "model": BACKEND_MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": V6_ABSTAIN}}],
                    "usage": None, **extra}
    q0, t0 = last_user(body), time.monotonic()
    # 10-27：检索 + 重排是同步 CPU/GPU 活，放线程池里跑，别卡住事件循环让 16 路请求排成一队
    body = no_think(await run_in_threadpool(STRATEGIES[STRATEGY], {**body, "model": BACKEND_MODEL}))
    if "_x_retrieved" in body:
        extra["x_retrieved"] = body.pop("_x_retrieved")
    try:
        r = await client.post(f"{BACKEND_URL}/chat/completions", json=body)
    except httpx.TimeoutException:
        raise HTTPException(504, f"后端超过 {TIMEOUT}s")
    if r.status_code != 200:
        raise HTTPException(502, f"后端 {r.status_code}: {r.text[:300]}")
    out = r.json()
    if STRATEGY == "v12":
        ans = out["choices"][0]["message"].get("content") or ""
        left = REASK_DEADLINE - (time.monotonic() - t0)
        if needs_reask(q0, ans) and left > 5:
            x = {"first": ans, "answer": None, "error": None}
            try:
                rq = no_think(reask_request(q0))
                r2 = await client.post(f"{BACKEND_URL}/chat/completions", json=rq, timeout=left)
                r2.raise_for_status()
                x["answer"] = (r2.json()["choices"][0]["message"].get("content") or "").strip() or None
            except Exception as e:  # noqa: BLE001  重问失败就保留第一次的回答
                x["error"] = repr(e)[:200]
            if x["answer"]:
                out["choices"][0]["message"]["content"] = x["answer"]
            extra["x_reask"] = x
    return {**out, **extra}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": BACKEND_MODEL, "object": "model", "owned_by": "local"}]}


@app.get("/x/info")
async def info():
    """运行记录用：策略、后端、模型版本（Ollama 时取 digest 与 Modelfile 参数，取不到不报错）。"""
    out = {"strategy": STRATEGY, "backend_url": BACKEND_URL, "backend_model": BACKEND_MODEL, "timeout_s": TIMEOUT,
           "no_think": NO_THINK}
    if STRATEGY in SYSTEMS:
        out.update({f"{STRATEGY}_system": SYSTEMS[STRATEGY], f"{STRATEGY}_max_tokens": V1_MAX_TOKENS})
    if STRATEGY == "v6":
        out["v6_gate"] = {"system": GATE_SYSTEM, "temperature": 0, "max_tokens": GATE_MAX_TOKENS,
                          "timeout_s": GATE_TIMEOUT, "abstain": V6_ABSTAIN}
    if STRATEGY in ("v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12"):
        m = retriever().meta
        out["v3_retrieval"] = {"kb": V3_KB, "k": V3_K, "passages": m["passages"], "built": m["built"],
                               "source": m["source"], "k1": m["k1"], "b": m["b"]}
    if STRATEGY in ("v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12"):
        rr = reranker()
        out["v4_rerank"] = {"model": "BAAI/bge-reranker-v2-m3", "bm25_n": V4_N, "keep": V3_K,
                            "device": rr.device, "max_length": rr.max_length}
    try:
        base = BACKEND_URL.removesuffix("/v1")
        tags = (await client.get(f"{base}/api/tags")).json()
        out["digest"] = next((m["digest"] for m in tags["models"] if m["name"] == BACKEND_MODEL), None)
        show = (await client.post(f"{base}/api/show", json={"model": BACKEND_MODEL})).json()
        out["parameters"] = show.get("parameters")
        out["version"] = (await client.get(f"{base}/api/version")).json().get("version")
    except Exception as e:  # noqa: BLE001
        out["info_error"] = repr(e)[:200]
    return out
