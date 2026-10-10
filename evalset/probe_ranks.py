# -*- coding: utf-8 -*-
"""evalset/probe_ranks.py —— 逐题报 gold 在三条路里的名次，并给出"词表重合度"。

为什么需要它：`run_eval` 只说"这道题 MISS"，但**MISS 的修法取决于名次**：
  · gold 排在第 6~15 → 粗捞池太小，扩池就能救（改 k，不是改词表）
  · gold 排在第 50 开外 → 词表/语义对不上，扩池没用，得改查询或索引
光知道"没进前 5"是没法动手的。

顺带报一个诊断量：**问题里的词有几个在 gold 条款正文里出现过**。
重合度接近 0 = 纯词表断裂（可以照这个方向想一般化的办法）；
重合度不低却排在后面 = 排序问题，不是词表问题。

跑法：
    <py> evalset/probe_ranks.py                                  # 主评测集
    <py> evalset/probe_ranks.py --questions evalset/heldout/questions.json
    <py> evalset/probe_ranks.py --only h5,h6,h8,h9,h23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jieba  # noqa: E402
import env_compat  # noqa: E402

env_compat.ensure_mmh3()
env_compat.ensure_uuid_utils()
import service as svc  # noqa: E402

MAXR = 60          # 查到第几名就放弃（报 ">60"）

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=str(REPO / "evalset" / "questions.json"))
ap.add_argument("--only", default=None, help="只看这些 id，逗号分隔")
A = ap.parse_args()

data = json.loads(Path(A.questions).read_text(encoding="utf-8"))
qs = data.get("questions") if isinstance(data, dict) else data
ans = [q for q in qs if q.get("type") == "answer"]
if A.only:
    want = {x.strip() for x in A.only.split(",")}
    ans = [q for q in ans if q.get("id") in want]

svc.rag.startup()
rag = svc.rag
text_of = {(c["doc"], c["clause"]): c["text"] for c in rag.chunks}

print(f"\n题目文件 {A.questions}　应答题 {len(ans)} 道　库 {len(rag.chunks)} 块\n")
hdr = f"{'题':>5} {'向量':>6} {'BM25':>6} {'融合':>6} {'词重合':>7}  问题 / gold"
print(hdr)
print("-" * 110)

rows = []
for q in ans:
    gold = tuple(q["gold"])
    _, vec_scores, vec_order, bm25_order = rag.route_orders(q["question"])
    fused = rag.rrf_order(vec_scores, vec_order, bm25_order) if hasattr(rag, "rrf_order") else None

    def rank_of(order):
        for i, idx in enumerate(list(order)[:MAXR], 1):
            c = rag.chunks[idx]
            if (c["doc"], c["clause"]) == gold:
                return i
        return None

    rv, rb = rank_of(vec_order), rank_of(bm25_order)
    # 融合名次直接问生产：和产品走的完全同一条路（含 route_orders 里的口语归一化）
    top = [c for c, _ in rag.search(q["question"], k=MAXR)]
    rf = next((i for i, c in enumerate(top, 1) if (c["doc"], c["clause"]) == gold), None)

    # 词表重合度：问题分词后，有多少个词真的出现在 gold 条款正文里
    toks = [t for t in jieba.cut_for_search(q["question"]) if len(t.strip()) > 1]
    body = text_of.get(gold, "")
    hit = [t for t in toks if t in body]
    cov = f"{len(hit)}/{len(toks)}" if toks else "-"

    mark = "" if rf and rf <= 5 else "  ← MISS"
    print(f"{q['id']:>5} {str(rv):>6} {str(rb):>6} {str(rf):>6} {cov:>7}  {q['question'][:26]}{mark}")
    print(f"{'':>5} {'':>6} {'':>6} {'':>6} {'':>7}  gold={gold[0]}｜{gold[1]}　重合词={hit if hit else '（一个都没有）'}")
    rows.append((q["id"], rv, rb, rf, len(hit), len(toks)))

print("-" * 110)
miss = [r for r in rows if not r[3] or r[3] > 5]
print(f"\n未进前 5：{len(miss)} 道 → {[m[0] for m in miss]}")
near = [m for m in miss if m[3] and 5 < m[3] <= 15]
far = [m for m in miss if not m[3] or m[3] > 15]
print(f"  其中排在 6~15 名（扩池有机会）：{len(near)} 道 {[m[0] for m in near]}")
print(f"  排在 15 名之外 / 捞不到（扩池没用）：{len(far)} 道 {[m[0] for m in far]}")
