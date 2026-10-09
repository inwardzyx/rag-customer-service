# -*- coding: utf-8 -*-
"""探针 2：把「粗捞池扩到 10 救回 q12」这件事查清楚。

★ ★ 这个脚本推翻了一个我自己写进README 的说法，务必先读完再引用它的数字★ ★

上一个探针（probe_sensitivity.py）发现池=10 时 Recall 从 13/14 变成 14/14，
看起来像"扩池就能救回 q12"。**但这个脚本证明那是截断线放宽造成的假象**：
q12 在池=5 和池=10 里的位置【都是第 10 名】—— 它从来没变好，只是原来被5 这道线切掉了。

而真正的三路消融给出了本项目最重要的一条负面结果：

    14 题里 gold 进前 5 的题数：仅向量 14/14 · 仅 BM25 12/14 · RRF 融合 13/14

**"只用向量" 比 "RRF 融合" 更好。** 原因就出在 q12：
    · 向量路把它排第 5（进前5）
    · BM25 路把它排第 36（捞不到）
    · RRF 用了两路的加权相加，结果把它推到第 10 —— 掉出去了

⇒ RRF 的固有性质：**一篇文章在某一路里名次很差时，另一路的好名次救不回它**，
    因为贡献是【相加】而不是【取max】。
⇒ 所以"混合检索一定更好"是错的。诚实的话是：
    **混合检索在本项目上没能证明自己有效，且实测比单路向量差1 题。**

跑法（不需要 API key）：
    D:\\Python-project\\.venv\\Scripts\\python.exe evalset\\probe_ablation.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import jieba
import env_compat
env_compat.ensure_mmh3()
env_compat.ensure_uuid_utils()
import service as svc

svc.rag.startup()
QS = json.loads((ROOT / "evalset" / "questions.json").read_text(encoding="utf-8"))
ANS = [q for q in QS if q["type"] == "answer"]


def ranked(query, krrf=60):
    """返回按 RRF 排好序的候选列表（不截断）。"""
    rag = svc.rag
    q = np.array(list(rag.model.query_embed([query])), dtype="float32")[0]
    vec_scores = rag.vectors @ q
    vec_order = np.argsort(-vec_scores)
    tokens = list(jieba.cut_for_search(query))
    bm25_order = np.argsort(-rag.bm25.get_scores(tokens))
    rrf = {}
    for rank, idx in enumerate(vec_order):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (krrf + rank)
    for rank, idx in enumerate(bm25_order):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (krrf + rank)
    order = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return [(rag.chunks[i], float(vec_scores[i])) for i, _ in order]


print("=== Q1/Q2：gold 条款在池=5 和池=10 里的位置 ===")
print(f"{'题':>4} {'池=5':>6} {'池=10':>6}  {'变化':<18} 问题")
moved = []
for q in ANS:
    gold = tuple(q["gold"])
    r = ranked(q["question"])
    pos5 = next((i for i, (c, _) in enumerate(r, 1) if (c["doc"], c["clause"]) == gold), None)
    pos10 = next((i for i, (c, _) in enumerate(r[:10], 1) if (c["doc"], c["clause"]) == gold), None)
    delta = "—" if pos5 == pos10 else f"{pos5} → {pos10}"
    if pos5 != pos10:
        moved.append(q["id"])
    print(f"{q['id']:>4} {str(pos5):>6} {str(pos10):>6}  {delta:<18} {q['question'][:24]}")

print()
print(f"位置发生变化的题：{moved}")

print()
print("=== Q3：池子扩大的代价（进精排的条数 × LLM 成本）===")
for pool in [3, 5, 10, 20]:
    top3 = sum(1 for _ in [1] * 3)  # 占位，下面用真实数字
    # 真实代价：精排 prompt 里要塞 pool 条资料，每条都要模型读并打分
    chars = 0
    for q in ANS[:3]:  # 取 3 道题估个量级
        cands = ranked(q["question"])[:pool]
        chars += sum(len(c["text"]) for c, _ in cands)
    print(f"  池={pool:2d} → 精排要读约 {chars//3:4d} 字/题（3 题均摊）"
          f"，相比池=5 约 {chars//3/(5*180):.1f} 倍（按池5≈900 字估）")

print()
print("=== 补：向量单路 vs BM25 单路 vs 融合（真正的三路消融）===")
for q in ANS:
    gold = tuple(q["gold"])
    rag = svc.rag
    vec = np.array(list(rag.model.query_embed([q["question"]])), dtype="float32")[0]
    vs = rag.vectors @ vec
    v_rank = list(np.argsort(-vs))
    b_rank = list(np.argsort(-rag.bm25.get_scores(list(jieba.cut_for_search(q["question"])))))
    fused = ranked(q["question"])

    def pos_of(order_idx, chunks):
        return next((k for k, i in enumerate(order_idx, 1)
                     if (chunks[i]["doc"], chunks[i]["clause"]) == gold), None)

    pv = pos_of(v_rank, rag.chunks)
    pb = pos_of(b_rank, rag.chunks)
    pf = next((k for k, (c, _) in enumerate(fused, 1)
               if (c["doc"], c["clause"]) == gold), None)
    print(f"  {q['id']:>4}「{q['question'][:18]}」"
          f"向量第 {str(pv):>3} / BM25 第 {str(pb):>3} / 融合第 {str(pf):>3}")

# 汇总成"进前 5 的题数"这个可直接比较的口径
def count_in_top5(picker):
    n = 0
    for q in ANS:
        gold = tuple(q["gold"])
        if picker(q, gold):
            n += 1
    return n

def by_vec(q, gold):
    rag = svc.rag
    vs = rag.vectors @ np.array(list(rag.model.query_embed([q["question"]])), dtype="float32")[0]
    return gold in [(rag.chunks[i]["doc"], rag.chunks[i]["clause"]) for i in np.argsort(-vs)[:5]]

def by_bm25(q, gold):
    rag = svc.rag
    order = np.argsort(-rag.bm25.get_scores(list(jieba.cut_for_search(q["question"]))))[:5]
    return gold in [(rag.chunks[i]["doc"], rag.chunks[i]["clause"]) for i in order]

def by_fused(q, gold):
    return gold in [(c["doc"], c["clause"]) for c, _ in ranked(q["question"])[:5]]

print()
print(f"14 题里 gold 进前 5 的题数："
      f"仅向量 {count_in_top5(by_vec)}/14 · 仅 BM25 {count_in_top5(by_bm25)}/14 "
      f"· RRF 融合 {count_in_top5(by_fused)}/14")