# -*- coding: utf-8 -*-
"""evalset/probe_index_augment.py —— 索引侧的一般化修法：把文件名/条款名也嵌进去。

动机（held-out 实测出来的真缺口，不是猜的）：
  · h8「在图书馆做勤工助学，一个月最多能排多少个小时的班？」
    gold = 学生勤工助学管理办法.md｜第十八条　向量第 6 / BM25 第 20 —— 两条路都差一点
  · h23「大一考试作弊被留校察看了，以后是不是奖学金、助学金都没我份了？」
    gold = 全日制本专科生国家奖助学金实施办法.md｜第六条　向量第 21 / BM25 第 25
    ★ 问句里明明白白有「奖助学金」，而**文件名里就有这三个字** ——
      可 `service.py:445` 建索引时只嵌正文，文件名和条款名**根本没进向量**。

这是一条**一般化**的修法（不是给某几道题加同义词）：所有块都多带一点自己的"名字"。
代价为零：不增加查询侧开销、不多调模型，只是建索引时多嵌几个字。

★ 纪律：四个写法在【主评测集 + held-out】两套题上同时量，绝不在 held-out 上单独挑。

跑法：<py> evalset/probe_index_augment.py
"""
from __future__ import annotations

import json
import sys
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

K = 5
RRF_K = 60

VARIANTS = {
    "现状：只嵌正文": lambda c: c["text"],
    "正文 + 条款名": lambda c: f'{c["clause"]}\n{c["text"]}',
    "正文 + 文件名": lambda c: f'{c["doc"]}\n{c["text"]}',
    "正文 + 文件名 + 条款名": lambda c: f'{c["doc"]}｜{c["clause"]}\n{c["text"]}',
}

SETS = {
    "主评测集": REPO / "evalset" / "questions.json",
    "held-out": REPO / "evalset" / "heldout" / "questions.json",
}

svc.rag.startup()
rag = svc.rag
chunks = rag.chunks


def load_ans(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    qs = d.get("questions") if isinstance(d, dict) else d
    return [q for q in qs if q.get("type") == "answer"]


def eval_variant(vecs):
    """给定一套文档向量，按**生产的融合口径**（RRF 相加）算两套题的 Recall@5。"""
    out = {}
    for name, p in SETS.items():
        ans = load_ans(p)
        ok, bad = 0, []
        for q in ans:
            text = svc.normalize_query(q["question"])          # ★ 走生产同一个入口
            qv = np.array(list(rag.model.query_embed([text])), dtype="float32")[0]
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
        out[name] = (ok, len(ans), bad)
    return out


print("\n建四种索引各嵌一遍（每种都要过一遍 271 块）……\n")
results = {}
for vname, fn in VARIANTS.items():
    vecs = np.array(list(rag.model.passage_embed([fn(c) for c in chunks])), dtype="float32")
    results[vname] = eval_variant(vecs)
    a = results[vname]["主评测集"]
    b = results[vname]["held-out"]
    print(f"{vname:22s} 主集 {a[0]:>2}/{a[1]}　held-out {b[0]:>2}/{b[1]}　"
          f"held-out 失手 {b[2]}")

print("\n" + "=" * 78)
base = results["现状：只嵌正文"]["held-out"]
print("相对现状的配对差异（held-out）：")
for vname, r in results.items():
    if vname == "现状：只嵌正文":
        continue
    fixed = [x for x in base[2] if x not in r["held-out"][2]]   # 现状错、新写法对
    broke = [x for x in r["held-out"][2] if x not in base[2]]   # 现状对、新写法错
    print(f"  {vname:22s} 修好 {fixed}　弄坏 {broke}")
