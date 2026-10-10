# -*- coding: utf-8 -*-
"""evalset/probe_fusion.py —— 把 RRF 的几种融合写法放在**两套题上一起量**。

起因（2026-10-10，held-out 首跑）：5 道失手里有 4 道的病根不是词表，是**融合**——
  h5 向量第 4（进前 5）→ 融合第 33
  h9 向量第 5（进前 5）→ 融合第 30
  仅向量 17/19，融合 14/19。
机制：RRF 的贡献是 1/(60+名次) 的**相加**。一路顶尖、另一路很差的块，
会被"两路都在中游"的块超过 —— 这正是 README 里早就写过、但当时只当成 q12 个例的那条。

★ 纪律（这次特别重要）：**不许在 held-out 上单独调参。**
  候选写法一律在【主评测集 + held-out】两边同时量，两边都看，再决定。
  否则 held-out 就退化成第二个调参集，我们又会失去唯一的独立测量。

跑法：
    <py> evalset/probe_fusion.py --questions evalset/questions.json
    <py> evalset/probe_fusion.py --questions evalset/heldout/questions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import env_compat  # noqa: E402

env_compat.ensure_mmh3()
env_compat.ensure_uuid_utils()
import service as svc  # noqa: E402

K = 5          # 判据用的粗捞池大小（产品当前值）
RRF_K = 60     # 当前生产用的常数

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=str(REPO / "evalset" / "questions.json"))
A = ap.parse_args()

data = json.loads(Path(A.questions).read_text(encoding="utf-8"))
qs = data.get("questions") if isinstance(data, dict) else data
ans = [q for q in qs if q.get("type") == "answer"]

svc.rag.startup()
rag = svc.rag


def orders(q):
    """拿到两路名次。★ 走 route_orders —— 生产和探针的唯一入口（含口语归一化）。"""
    _, _vs, vec_order, bm25_order = rag.route_orders(q)
    return list(vec_order), list(bm25_order)


def fuse(vec_order, bm25_order, mode, topk=None):
    v = vec_order[:topk] if topk else vec_order
    b = bm25_order[:topk] if topk else bm25_order
    if mode == "vector":
        return list(vec_order)
    if mode == "bm25":
        return list(bm25_order)
    if mode == "sum":                       # 当前生产：两路全量相加
        s: dict[int, float] = {}
        for r, i in enumerate(v):
            s[i] = s.get(i, 0.0) + 1.0 / (RRF_K + r)
        for r, i in enumerate(b):
            s[i] = s.get(i, 0.0) + 1.0 / (RRF_K + r)
        return [i for i, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)]
    if mode == "max":                       # 取两路里更好的那个贡献
        s = {}
        for r, i in enumerate(v):
            s[i] = max(s.get(i, 0.0), 1.0 / (RRF_K + r))
        for r, i in enumerate(b):
            s[i] = max(s.get(i, 0.0), 1.0 / (RRF_K + r))
        return [i for i, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)]
    raise ValueError(mode)


MODES = [
    ("仅向量", "vector", None),
    ("仅 BM25", "bm25", None),
    ("RRF 相加（生产）", "sum", None),
    ("RRF 取 max", "max", None),
    ("RRF 相加·每路先取 20", "sum", 20),
    ("RRF 相加·每路先取 50", "sum", 50),
]

res: dict[str, list[bool]] = {}
detail: dict[str, list[str]] = {}
for name, mode, topk in MODES:
    ok, bad = 0, []
    for q in ans:
        gold = tuple(q["gold"])
        vo, bo = orders(q["question"])
        order = fuse(vo, bo, mode, topk)[:K]
        got = {(rag.chunks[i]["doc"], rag.chunks[i]["clause"]) for i in order}
        if gold in got:
            ok += 1
        else:
            bad.append(q["id"])
    # ★ 这里必须用 q["id"] 比对，不能写 `i not in bad` —— bad 里装的是 id（'h5'），
    #   不是下标；写成下标比对的话每个 i 都不在 bad 里，全表恒为 True，
    #   于是下面的 McNemar 永远输出"与生产完全一致"（第一版就是这么错的：
    #   聚合数字是对的，配对检验是静的 —— 又一个"看着在跑其实没跑"）。
    res[name] = [q["id"] not in bad for q in ans]
    detail[name] = bad
    print(f"  {name:24s} Recall@{K} = {ok}/{len(ans)}   失手 {bad}")

# ---- 配对检验：生产 vs 挑战者（McNemar 精确，双侧）----
base = res["RRF 相加（生产）"]
print("\n配对差异（对手 vs 生产；只列不一致的对）：")
from math import comb  # noqa: E402
for name, _m, _t in MODES:
    if name == "RRF 相加（生产）":
        continue
    other = res[name]
    b = sum(1 for i in range(len(ans)) if other[i] and not base[i])   # 对手对、生产错
    c = sum(1 for i in range(len(ans)) if base[i] and not other[i])   # 生产对、对手错
    if b + c == 0:
        print(f"  {name:24s} 与生产完全一致")
        continue
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n))
    verdict = "显著" if p < 0.05 else "**不显著**（方向性证据）"
    print(f"  {name:24s} 对手赢 {b} / 生产赢 {c}　McNemar p = {p:.3f} → {verdict}")
