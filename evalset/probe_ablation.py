# -*- coding: utf-8 -*-
"""探针 2：把「粗捞池扩到 10 救回 q12」这件事查清楚。

★ ★ 这个脚本推翻了一个我自己写进README 的说法，务必先读完再引用它的数字★ ★

上一个探针（probe_sensitivity.py）发现池=10 时 Recall 从 13/14 变成 14/14（★ 那会儿是 14 题；q18 改类后是 15 题，见下方修正记录），
看起来像"扩池就能救回 q12"。**但这个脚本证明那是截断线放宽造成的假象**：
q12 在池=5 和池=10 里的位置【都是第 10 名】—— 它从来没变好，只是原来被 5 这道线切掉了。

而真正的三路消融给出了本项目最重要的一条负面结果：

    15 题里 gold 进前 5 的题数：仅向量 15/15 · 仅 BM25 14/15 · RRF 融合 15/15（2026-10-10 修复 q12 后）

**三条路已经区分不开了（只有 BM25 差 1 题）⇒ 评测集饱和，消融失去分辨力。**
修复前的差异全部出在 q12 这一题（下面这组已过期，保留供对照）：
    · 向量路把它排第 7（进前 5）
    · BM25 路把它排第 80（捞不到）
    · RRF 融合排第 10（没进前 5）

★ 2026-10-10 修正：这份 docstring 原来写的是 "14 题…仅向量 14/14 ·
  BM25 12/14 · RRF 13/14"、且称"只用向量比 RRF 更好"、
  "向量路把它排第 5、BM25 排第 36" —— 三个数字全是旧的。
  原因：q18 语料扩容后从 refuse 改成 answer（14 → 15 题），
  而本文件的 print 分母我上次只抽成了变量（N = len(ANS)），
  **同一批数字的另一份拷贝留在了这段散文里**，于是它没跟着变。
  ⇒ 这个坑与 CLAUDE.md 里"两处各写一份判据"是同一个：
    抽变量只改了一处，另一处照旧。**改数字时要把全文搜一遍。**
  现值由 `python evalset/probe_ablation.py` 实测得出（2026-10-10）。

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


def _contrib(krrf, rank):
    """RRF 单项贡献 1/(k+rank)。

    ★ 与 probe_sensitivity.py 的同名函数保持一致：k=0 且 rank=0 时公式本身是 1/0，
      这是 RRF 的定义域边界不是 bug（说明"k 必须大于 0"）。返回 inf 让排序能跑完，
      顺便把"k=0 会爆"变成一个可报告的实测结果。
      —— 这个兜底是 DeepSeek Harness 做独立复核时发现漏掉才补上的：
      原实现直接写 1.0/(krrf+rank)，k=0 会 ZeroDivisionError。
    """
    try:
        return 1.0 / (krrf + rank)
    except ZeroDivisionError:
        return float("inf")


def ranked(query, krrf=60):
    """返回按 RRF 排好序的候选列表（不截断）。

    ★ 名次一律从 `svc.rag.route_orders()` 取，**不要在这里自己 embed / 自己分词**。
      自己写一份 = 复制检索实现，生产改了前处理这里不会跟着改。
      2026-10-10 就栽过：加了口语归一化后 run_eval 报 15/15，
      而本探针仍报 q12 融合第 10 名 —— 因为这一步被绕过了。
    """
    rag = svc.rag
    _, vec_scores, vec_order, bm25_order = rag.route_orders(query)
    rrf = {}
    for rank, idx in enumerate(vec_order):
        rrf[idx] = rrf.get(idx, 0.0) + _contrib(krrf, rank)
    for rank, idx in enumerate(bm25_order):
        rrf[idx] = rrf.get(idx, 0.0) + _contrib(krrf, rank)
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
    _, vs, v_order, b_order = rag.route_orders(q["question"])
    v_rank = list(v_order)
    b_rank = list(b_order)
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
    _, _, order, _ = rag.route_orders(q["question"])
    return gold in [(rag.chunks[i]["doc"], rag.chunks[i]["clause"]) for i in order[:5]]

def by_bm25(q, gold):
    rag = svc.rag
    _, _, _, order = rag.route_orders(q["question"])
    return gold in [(rag.chunks[i]["doc"], rag.chunks[i]["clause"]) for i in order[:5]]

def by_fused(q, gold):
    return gold in [(c["doc"], c["clause"]) for c, _ in ranked(q["question"])[:5]]

N = len(ANS)   # ★ 分母自动取应答集长度：写死 /14 在 q18 变成 answer 后会打出 15/14
print()
print(f"{N} 题里 gold 进前 5 的题数："
      f"仅向量 {count_in_top5(by_vec)}/{N} · 仅 BM25 {count_in_top5(by_bm25)}/{N} "
      f"· RRF 融合 {count_in_top5(by_fused)}/{N}")