# -*- coding: utf-8 -*-
"""experiments/cache_feasibility.py —— 给 rerank 加缓存，前置可行性检查

**这个实验要回答的不是"缓存能快多少"（那不用测也知道），是"缓存到底该做在哪个键上"。**

★ 为什么必须先量键，不直接写缓存：
    `RAG.rerank()`（service.py:437）的 prompt 里**含 query**（service.py:447）
    ⇒ 精排分【依赖问题】。这意味着 "候选集相同就复用结果" 是错的：
       同一个候选集，问两个不同问题，打分应该不同。
    选错键 = 2434ms 降到 0，但答案悄悄变错，而且【没有任何报错】。

所以要量三件事（每件都可能直接判死这个实验）：

    第 1 步  精确重复率      —— 问题字符串一字不差地重复
    第 2 步  候选集重复率    —— top-5 集合完全一样的有几组
                               （这一项【不能直接当缓存命中数】，见第 5 步的坑）
    第 3 步  ★ 语义相似度分布 —— "同 gold"的题对 vs "不同 gold"的题对，
                               两组余弦能不能分开。分得开 ⇒ 语义缓存可行。

    第 4 步  前置断言        —— 分不开就判实验无效并退出，不硬写结论

★ 纪律（沿用 stale_doc_robustness.py 的教训）：
    这个脚本最容易输出的错误答案是 "0 命中" 和 "命中率 0%"。
    0 长得像 "系统很严谨"，但它其实只说明【这 20 道题是专门造的】，
    不说明真实流量没有重复。结论段会强制把这句话写出来。

跑法（仓库根目录）：
    D:\\Python-project\\.venv\\Scripts\\python.exe experiments/cache_feasibility.py
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evalset" / "questions.json"
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("LANGSMITH_TRACING", "false")

import numpy as np                                   # noqa: E402

import service as svc                                # noqa: E402


# ==================================================================
# 第 1 步 / 第 2 步 / 第 3 步 都要用到的东西
# ==================================================================
def load_questions() -> list[dict]:
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        return json.load(f)


def gold_key(q: dict) -> tuple:
    """这道题的 gold 归一成一个可哈希的键；拒答题没有 gold，返回空元组。"""
    return tuple(q.get("gold") or [])


def top_set(q: dict, k: int = 5) -> frozenset:
    """这个问题的粗捞 top-k 候选集（用 (doc, clause) 标识一个块）。"""
    cands = svc.rag.search(q["question"], k=k)
    return frozenset((c["doc"], c["clause"]) for c, _ in cands)


def embed_all(questions: list[dict]) -> dict:
    """一次把所有问题 embed 成向量，返回 {qid: np.array}。

    为什么一次批量而不是循环里单个调：query_embed 每次调用都有一次模型前向，
    20 个问题分 20 次会慢；批量只多花一点内存。
    """
    texts = [q["question"] for q in questions]
    vecs = list(svc.rag.model.query_embed(texts))
    return {q["id"]: np.asarray(v, dtype="float32") for q, v in zip(questions, vecs)}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。bge 的输出已经归一化，这里再点一次保证落在 [-1, 1]。"""
    return float(np.dot(a, b))


# ==================================================================
# 主流程
# ==================================================================
def main() -> int:
    qs = load_questions()
    svc.rag.startup()
    print(f"[前置] 入库 {len(svc.rag.chunks)} 块，拦下 {len(svc.rag.rejected)} 块")
    print(f"[前置] 题目 {len(qs)} 道\n")

    # ---------------- 第 1 步：精确重复率 ----------------
    print("=" * 68)
    print("第 1 步　精确重复率（问题字符串一字不差地重复）")
    print("=" * 68)
    seen: dict[str, list[str]] = {}
    for q in qs:
        seen.setdefault(q["question"].strip(), []).append(q["id"])
    dup_groups = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"去重后 {len(seen)} 种问法 / 共 {len(qs)} 道题")
    if dup_groups:
        for text, ids in dup_groups.items():
            print(f"  重复：{ids} → {text}")
    else:
        print("精确重复：0 组")
    print("\n★ 结论：精确字符串缓存【不可能有命中】，这一路直接判死。")
    print("  （这是预期内的 —— 评测集本来就要避免重复，否则等于送分。）\n")

    # ---------------- 第 2 步：候选集重复率 ----------------
    print("=" * 68)
    print("第 2 步　候选集重复率（top-5 集合完全一样的有几组）")
    print("=" * 68)
    tsets = {q["id"]: top_set(q) for q in qs}
    by_set: dict[frozenset, list[str]] = {}
    for qid, s in tsets.items():
        by_set.setdefault(s, []).append(qid)
    shared = {s: ids for s, ids in by_set.items() if len(ids) > 1}
    id2q = {q["id"]: q for q in qs}
    n_saved = sum(len(ids) - 1 for ids in shared.values())
    print(f"不同的 top-5 集合：{len(by_set)} 种 / {len(qs)} 道题")
    print(f"共享同一集合的组：{len(shared)} 组")
    for s, ids in shared.items():
        golds = {i: gold_key(id2q[i]) for i in ids}
        same = len(set(golds.values())) == 1
        mark = "gold 相同" if same else "★ gold 不同"
        print(f"  {ids} → {mark}")
        for i in ids:
            print(f"      {i}: {id2q[i]['question']}")
    print(f"\n如果（错误地）按候选集复用，能省 {n_saved} 次精排调用 "
          f"（{len(qs)} → {len(by_set)} 次，少 {n_saved / len(qs) * 100:.0f}%）。")
    print("★ 但这个数【不是】能省的钱 —— 见第 5 步，那里有个正确性陷阱。\n")

    # ---------------- 第 3 步：★ 语义相似度分布 ----------------
    print("=" * 68)
    print("第 3 步　★ 语义相似度分布：同 gold 的题对 vs 不同 gold 的题对")
    print("=" * 68)
    vecs = embed_all(qs)
    ids = [q["id"] for q in qs]

    same_pairs, diff_pairs = [], []
    for a, b in itertools.combinations(ids, 2):
        s = cos(vecs[a], vecs[b])
        (same_pairs if gold_key(id2q[a]) == gold_key(id2q[b]) else diff_pairs).append(
            (a, b, s))

    def stat(name, pairs):
        if not pairs:
            print(f"  {name}：0 对（无法比较）")
            return None
        vals = sorted(p[2] for p in pairs)
        n = len(vals)
        med = vals[n // 2]
        print(f"  {name}：{n} 对　最小 {vals[0]:.4f}　中位 {med:.4f}　最大 {vals[-1]:.4f}")
        return vals

    print(f"\n共 {len(ids) * (len(ids) - 1) // 2} 对题：")
    vs = stat("同 gold", same_pairs)
    vd = stat("不同 gold", diff_pairs)

    # ---------------- 第 4 步：★ 前置断言 ----------------
    print("\n" + "=" * 68)
    print("第 4 步　★ 前置断言：两组分不分开？分不开就判实验无效")
    print("=" * 68)
    if not vs or not vd:
        print("样本不足，无法判断（本次不出结论，退出）。")
        return 2
    # 分开的判据：同 gold 的【最小值】必须高于不同 gold 的【最大值】。
    # 只有完全 separable 才敢谈阈值；重叠就得靠更多数据，不能靠调阈值硬凑。
    sep = vs[0] > vd[-1]
    print(f"同 gold 最低 {vs[0]:.4f}　vs　不同 gold 最高 {vd[-1]:.4f}")
    if not sep:
        print("\n★★ 实验结论：两组【重叠】⇒ 靠余弦阈值分不开同义问法和不同问法。")
        print("   ⇒ 语义缓存【不可行】，任何阈值都会误命中，直接判死。")
        print("   ⇒ 要做只能上 LLM 判等价（更贵）或人工定白名单，不在本实验范围。")
        return 1
    print("\n两组完全 separable ⇒ 阈值可选，落在中间即可。")

    print("\n各阈值下的语义缓存表现（命中 = 与已问过的某题余弦 ≥ 阈值）：")
    print(f"  {'阈值':>6}  {'命中数':>6}  {'命中率':>8}  {'误命中':>6}  说明")
    best = None
    for th in (0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80):
        hit = wrong = 0
        order = [q["id"] for q in qs]
        cache: list[str] = []
        for qid in order:
            bests = max((cos(vecs[qid], vecs[c]) for c in cache), default=0.0)
            if bests >= th:
                hit += 1
                # 误命中 = 命中的那道题和它不是同一个 gold ⇒ 答案会错
                hit_ids = [c for c in cache if cos(vecs[qid], vecs[c]) >= th]
                if any(gold_key(id2q[c]) != gold_key(id2q[qid]) for c in hit_ids):
                    wrong += 1
            else:
                cache.append(qid)
        rate = hit / len(qs) * 100
        note = "安全" if wrong == 0 else f"⚠ 有 {wrong} 次答案会错"
        print(f"  {th:>6.2f}  {hit:>6}  {rate:>7.1f}%  {wrong:>6}  {note}")
        if wrong == 0 and (best is None or hit > best[1]):
            best = (th, hit, rate)
    if best:
        print(f"\n★ 最优安全阈值 {best[0]:.2f}：命中 {best[1]}/{len(qs)}（{best[2]:.1f}%），0 误命中。")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
