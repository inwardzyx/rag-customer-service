#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_pool_size.py —— 粗捞池 k=5 vs 10，端到端到底有没有差别？

## 为什么要做（README〈还没做的〉里挂了很久的一条）

README 现在写着「**粗捞池只有 5 条** …… 扩池必须用 eval 数字证明有效」——
也就是说：**"不扩池"是拍板，不是量出来的**。
而 `service.py:703` 的 `ChatRequest.top_k` 本来就支持 1~10，只是 `run_eval` 没把它透出来。
⇒ 这条待办的正确形态是**量一遍**，然后要么改成 k、要么把"不扩"钉成有依据的决定。

## 判据（跑之前先定，防事后找理由）

| 结果 | 结论 |
|---|---|
| 扩到 10 后 **漏答率下降 或 拒答准确率上升**，且两套题都不变差 | 改 k=10 |
| 三者基本不动 | **照实写"在本语料上扩池无可测收益"**，把待办改成"已量，决定不扩" |
| 拒答准确率下降（池子大 ⇒ 给了精排更多机会捞到像的） | 明确写"扩池有负面作用"，并给出数字 |

## 注意：这里量的是**端到端**，不是 Recall@k

`Recall@5` 在池=10 时语义会变（池子大了当然更容易"进池"），拿它比就没意义。
真正该看的是产品指标：**漏答率**与**拒答准确率**（都用 `svc.chat()`，与产品同一条路）。
另报一个诊断量 `gold 进池率`（池子大小的检索天花板）。

跑法：
    <py> evalset/probe_pool_size.py --questions evalset/heldout/questions.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("LANGSMITH_TRACING", "false")

import service as svc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=str(REPO / "evalset" / "heldout" / "questions.json"))
ap.add_argument("--pools", default="5,10")
A = ap.parse_args()

pools = [int(x) for x in A.pools.split(",") if x.strip()]
data = json.loads(Path(A.questions).read_text(encoding="utf-8"))
qs = data.get("questions") if isinstance(data, dict) else data
ans = [q for q in qs if q.get("type") == "answer"]
ref = [q for q in qs if q.get("type") == "refuse"]

svc.rag.startup()
print(f"\n题目 {Path(A.questions).name}　应答题 {len(ans)}　拒答题 {len(ref)}")
print(f"池子 {pools}　★ 产品默认路径：use_rerank=True + use_rewrite=True\n")

print(f"{'池':>3} {'gold进池':>9} {'漏答率':>9} {'拒答准确率':>11}   备注")
print("-" * 62)
results = {}
for k in pools:
    inpool, miss, ok_ref, refused_ids = 0, [], 0, []
    for q in ans:
        cands = svc.rag.search(q["question"], k=k)          # 检索天花板（含改写，与产品一致）
        if tuple(q["gold"]) in {(c["doc"], c["clause"]) for c, _ in cands}:
            inpool += 1
        hit = svc.chat(svc.ChatRequest(question=q["question"], use_rerank=True,
                                       use_rewrite=True, top_k=k)).knowledge_hit
        if not hit:
            miss.append(q["id"])
    for q in ref:
        hit = svc.chat(svc.ChatRequest(question=q["question"], use_rerank=True,
                                       use_rewrite=True, top_k=k)).knowledge_hit
        if not hit:
            ok_ref += 1
        else:
            refused_ids.append(q["id"])
    results[k] = (inpool, miss, ok_ref, refused_ids)
    note = "" if k == 5 else "  ← 候选"
    print(f"{k:>3} {inpool:>5}/{len(ans):<3} {len(miss):>4}/{len(ans):<3} "
          f"{ok_ref:>6}/{len(ref):<3}{note}")
    print(f"      漏答的是 {miss}")
    print(f"      误答的是 {refused_ids}")
print("-" * 62)
print("\n判读（判据见文件顶部）：")
for k in pools:
    inpool, miss, ok_ref, _ = results[k]
    print(f"  k={k:<2} gold 进池 {inpool}/{len(ans)}　漏答 {len(miss)}　拒答对 {ok_ref}/{len(ref)}")
