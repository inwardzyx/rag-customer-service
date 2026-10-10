# -*- coding: utf-8 -*-
"""evalset/probe_embedding_model.py —— 换更大的 embedding 模型，值不值？

背景：两条免费的一般化修法已经被实测否决了：
  · 改融合（RRF 相加 → 仅向量）：39 题上 34 vs 33，p=1.000，是噪声
  · 索引增强（把文件名/条款名嵌进去）：held-out 33 → 32/31/28，**更差**
剩下的语义缺口（h8 勤工助学工时、h23 处分×奖助学金）两条路都捞不到，
最可信的一般化手段只剩"换更强的向量模型"。

★ 代价必须一起报，否则不算结论：
  模型更大 ⇒ 首次下载更大、启动更慢、内存更高。**只有 held-out 明显变好才值得换。**

跑法：<py> evalset/probe_embedding_model.py
      <py> evalset/probe_embedding_model.py --models BAAI/bge-small-zh-v1.5,BAAI/bge-base-zh-v1.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jieba  # noqa: E402
import numpy as np  # noqa: E402
import env_compat  # noqa: E402

env_compat.ensure_mmh3()
env_compat.ensure_uuid_utils()
import service as svc  # noqa: E402

K, RRF_K = 5, 60

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="BAAI/bge-small-zh-v1.5,BAAI/bge-base-zh-v1.5")
A = ap.parse_args()

SETS = {
    "主评测集": REPO / "evalset" / "questions.json",
    "held-out": REPO / "evalset" / "heldout" / "questions.json",
}


def load_ans(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    qs = d.get("questions") if isinstance(d, dict) else d
    return [q for q in qs if q.get("type") == "answer"]


svc.rag.startup()
rag = svc.rag
chunks = rag.chunks
from fastembed import TextEmbedding  # noqa: E402

for model_name in [m.strip() for m in A.models.split(",") if m.strip()]:
    print(f"\n{'=' * 74}\n模型 {model_name}")
    try:
        t0 = time.time()
        model = TextEmbedding(model_name=model_name)
        vecs = np.array(list(model.passage_embed([c["text"] for c in chunks])),
                        dtype="float32")
        load_s = time.time() - t0
    except Exception as e:                                   # 下载失败/不支持
        print(f"  ✗ 加载失败：{type(e).__name__}: {str(e)[:160]}")
        continue
    print(f"  维度 {vecs.shape[1]}　建索引耗时 {load_s:.1f}s")

    for sname, p in SETS.items():
        ans = load_ans(p)
        ok, bad = 0, []
        for q in ans:
            text = svc.normalize_query(q["question"])
            qv = np.array(list(model.query_embed([text])), dtype="float32")[0]
            v_order = np.argsort(-(vecs @ qv))
            b_order = np.argsort(-rag.bm25.get_scores(list(jieba.cut_for_search(text))))
            s: dict[int, float] = {}
            for r, i in enumerate(v_order):
                s[i] = s.get(i, 0.0) + 1.0 / (RRF_K + r)
            for r, i in enumerate(b_order):
                s[i] = s.get(i, 0.0) + 1.0 / (RRF_K + r)
            top = [i for i, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:K]]
            got = {(chunks[i]["doc"], chunks[i]["clause"]) for i in top}
            if tuple(q["gold"]) in got:
                ok += 1
            else:
                bad.append(q["id"])
        print(f"  {sname:8s} Recall@{K} = {ok}/{len(ans)}　失手 {bad}")
