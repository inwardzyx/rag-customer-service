# -*- coding: utf-8 -*-
"""一次性测量脚本（不入库、不提交）：
把 docs/ 下所有块两两算余弦，按「是否同一条款身份」分组，看两组分布差多少。
目的是给"近似重复关"的阈值找证据，而不是拍脑袋。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from kb.loader import load_documents
from service import CACHE_DIR, DOCS_DIR
from fastembed import TextEmbedding

docs, _load_errors = load_documents(DOCS_DIR)
model = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=CACHE_DIR)

texts = [c["text"] for c in docs]
vecs = np.array(list(model.passage_embed(texts)), dtype="float32")

same_id, diff_id = [], []
for i in range(len(docs)):
    for j in range(i + 1, len(docs)):
        cos = float(vecs[i] @ vecs[j])
        a, b = docs[i], docs[j]
        pair = (cos, a["doc"], a["clause"], b["doc"], b["clause"],
                a["text"][:28], b["text"][:28])
        # 身份：同一份文档的同一条 —— 限定最严的一种
        if (a["doc"], a["clause"]) == (b["doc"], b["clause"]):
            same_id.append(pair)
        else:
            diff_id.append(pair)

same_id.sort(reverse=True)
diff_id.sort(reverse=True)


def show(title, rows, n=12):
    print(f"\n===== {title}（共 {len(rows)} 对，显示最像的 {min(n, len(rows))} 对）=====")
    for cos, ad, ac, bd, bc, at, bt in rows[:n]:
        print(f"{cos:.4f} | {ad}/{ac}  vs  {bd}/{bc}")
        print(f"         A: {at}")
        print(f"         B: {bt}")


show("【同一文档+同一条款】", same_id)
show("【身份不同】", diff_id)

print("\n===== 结论用得上 =======")
print(f"身份相同  最高 {same_id[0][0]:.4f}" if same_id else "身份相同：无样本")
print(f"身份不同  最高 {diff_id[0][0]:.4f}")
print(f"身份不同  top5: {[round(r[0], 4) for r in diff_id[:5]]}")
