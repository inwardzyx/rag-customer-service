# -*- coding: utf-8 -*-
"""evalset/probe_query_rewrite.py —— 最后一个一般化候选：让 LLM 把口语问题改写成制度用语。

为什么还试这一个：item 1 的三条免费修法已经被实测否决
（改融合 / 索引增强 / 换更大模型，见 RESULTS.md 三之补三）。
剩下的语义缺口（h8 勤工助学工时、h23 处分×奖助学金）两条路都捞不到。
唯一还没试的一般化手段就是**查询侧改写** —— 它不针对具体题目，
而是把"学生怎么说"翻译成"制度怎么写"，理论上对所有口语题都生效。

★ 它与 `normalize_query()`（那张手抄同义词表）的区别：
  同义词表只能覆盖我想到的（礼拜→周），而 LLM 改写能处理没想到的说法。
  代价也很明确：**每次查询多一次模型调用**（延迟 + 钱）。

★ 纪律：两套题一起量；改写 prompt 里**不许出现任何评测题**（否则就是变相调参）。

跑法：<py> evalset/probe_query_rewrite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import os  # noqa: E402

os.environ.setdefault("LANGSMITH_TRACING", "false")

import service as svc  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

K = 5
SETS = {
    "主评测集": REPO / "evalset" / "questions.json",
    "held-out": REPO / "evalset" / "heldout" / "questions.json",
}
CACHE = Path(r"D:\dsh-work\_rewrite_cache.json")

PROMPT = """把下面这句学生的口语问题，改写成**学校规章制度里会用的说法**，用于检索。

要求：
1. 只输出改写后的一句话，不要解释、不要引号、不要编号。
2. 保留原问题的**意图和关键限定**（时间、对象、金额、情形都不能丢）。
3. 把口语词换成制度用语（例："一个礼拜"→"一周"；"翘课"→"旷课"；"直接开掉"→"开除学籍"）。
4. 如果原句已经很书面，就原样返回，别画蛇添足。

学生的问题：{q}
改写后："""


def load_ans(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    qs = d.get("questions") if isinstance(d, dict) else d
    return [q for q in qs if q.get("type") == "answer"]


def hit(question, gold, k=K):
    top = [c for c, _ in svc.rag.search(question, k=k)]
    return (gold[0], gold[1]) in {(c["doc"], c["clause"]) for c in top}


svc.rag.startup()
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

print(f"\n改写 prompt 里没有任何评测题；只做**一次**改写（不多路，省一次调用）\n")
results = {}
for sname, p in SETS.items():
    ans = load_ans(p)
    base_ok, rw_ok, fixed, broke = [], [], [], []
    for q in ans:
        gold = q["gold"]
        if q["id"] not in cache:
            try:
                # ★ 用属性 `svc.rag.llm`（真正构造 ChatDeepSeek 的入口），
                #   不是 `_llm` —— 后者是"注入假模型"用的槽位，正常启动时是 None。
                #   第一版写成 _llm 导致**全部改写失败**，而下面的兜底又把原句填回去，
                #   于是打出一组看起来很像结论的数字（"改写后没变化"）。
                #   ⇒ 教训：**错误兜底绝不能产出看起来合理的结果**。
                cache[q["id"]] = svc._invoke_llm(
                    svc.rag.llm,
                    [HumanMessage(content=PROMPT.format(q=q["question"]))]).content.strip()
            except Exception as e:
                # ★ 改写失败就**中止**，不许拿原句顶替 —— 那会把"探针坏了"
                #   伪装成"改写没用"，正是本项目反复在抓的那类假象。
                raise SystemExit(
                    f"✘ {q['id']} 改写失败：{type(e).__name__}: {str(e)[:120]}\n"
                    f"  已中止（不拿原句兜底：那会产出看起来合理的假结论）")
        b = hit(q["question"], gold)
        r = hit(cache[q["id"]], gold)
        base_ok.append(b)
        rw_ok.append(r)
        if r and not b:
            fixed.append(q["id"])
        if b and not r:
            broke.append(q["id"])
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    results[sname] = (sum(base_ok), sum(rw_ok), len(ans), fixed, broke)
    print(f"{sname:8s} 现状 {sum(base_ok):>2}/{len(ans)}　"
          f"改写后 {sum(rw_ok):>2}/{len(ans)}　修好 {fixed}　弄坏 {broke}")

print("\n改写样例（看它到底干了什么）：")
for qid in ("h8", "h23", "h9", "q12"):
    if qid in cache:
        print(f"  {qid}: {cache[qid][:90]}")
