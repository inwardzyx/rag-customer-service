# -*- coding: utf-8 -*-
"""临时探针：RRF 的 k 和粗捞池大小对 Recall@5 的影响。

为什么跑这个：
  · README 里承认"k=60 是论文经验值，没做敏感性实验"
  · 面试官很可能问"k 试过别的值吗"

跑法（不需要 API key）：
    D:\\Python-project\\.venv\\Scripts\\python.exe evalset\\_probe_sensitivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import jieba
import env_compat
env_compat.ensure_mmh3()
import service as svc

# ★ 必须手动调 startup()：service.py 里模型是懒加载的（@property），
#   不调的话 svc.rag.model 还不存在 —— 报AttributeError。
#   run_eval.py 那边也是这么做的（它的 main() 里 svc.rag.startup()）。
svc.rag.startup()

QS = svc.__dict__.get("_QS")  # 若无则自己读
import json
QS = json.loads((Path(__file__).resolve().parent / "questions.json").read_text(encoding="utf-8"))
ANS = [q for q in QS if q["type"] == "answer"]


def _contrib(krrf, rank):
    """RRF 单项贡献 1/(k+rank)。

    ★ k=0 且 rank=0 时公式本身是 1/0 —— 这是 RRF 的定义域边界，不是 bug：
      它说明"k 必须大于 0"。这里返回 inf 让排序能跑完（Python 里 float('inf') 参与比较是合法的），
      顺便把"k=0 会爆"这件事本身也变成一个可报告的实测结果。
    """
    try:
        return 1.0 / (krrf + rank)
    except ZeroDivisionError:
        return float("inf")


def rrf_search(query, k, krrf):
    """复制 service.py:407 的逻辑，只把 60 换成参数。"""
    rag = svc.rag
    q = np.array(list(rag.model.query_embed([query])), dtype="float32")[0]
    vec_scores = rag.vectors @ q
    vec_order = np.argsort(-vec_scores)
    tokens = list(jieba.cut_for_search(query))
    bm25_order = np.argsort(-rag.bm25.get_scores(tokens))
    rrf = {}
    for rank, idx in enumerate(vec_order):
        rrf[idx] = rrf.get(idx, 0.0) + _contrib(krrf, rank)
    for rank, idx in enumerate(bm25_order):
        rrf[idx] = rrf.get(idx, 0.0) + _contrib(krrf, rank)
    top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(rag.chunks[idx], float(vec_scores[idx])) for idx, _ in top]


print("=== RRF 的 k 敏感性（粗捞池固定 5，衡量 Recall@5）===")
for krrf in [0, 1, 10, 30, 60, 100, 1000]:
    hit = 0
    missed = []
    for q in ANS:
        cands = rrf_search(q["question"], 5, krrf)
        recalled = {(c["doc"], c["clause"]) for c, _ in cands}
        if tuple(q["gold"]) in recalled:
            hit += 1
        else:
            missed.append(q["id"])
    print(f"  k_rrf={krrf:5d}→ Recall@5 = {hit}/{len(ANS)}   漏: {missed}")

print()
print("=== 粗捞池大小敏感性（k_rrf=60，衡量 Recall@pool）===")
for pool in [1, 3, 5, 10, 20, 68]:
    hit = 0
    for q in ANS:
        cands = rrf_search(q["question"], pool, 60)
        recalled = {(c["doc"], c["clause"]) for c, _ in cands}
        if tuple(q["gold"]) in recalled:
            hit += 1
    print(f"  池={pool:3d} → Recall@{pool:<2d} = {hit}/{len(ANS)}")