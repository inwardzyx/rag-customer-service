# -*- coding: utf-8 -*-
"""一次性测量脚本（不入库、不提交）：
量出「真重复 / 同义改写 / 不同条款」三类配对的余弦各是多少，
用来给 DUP_THRESHOLD 选一个能同时放过后者、拦住前两者的值。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from service import CACHE_DIR
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=CACHE_DIR)

CASES = [
    ("真重复·一字之差",
     "学生在校学习期间离校应当由本人办理请假手续，并附有关证明材料。",
     "学生在校学习期间离校应当由本人办理请假手续，并附上有关证明材料。"),
    ("真重复·完全相同",
     "内设机构",
     "内设机构"),
    ("同义改写A（测试里那条）",
     "现货商品在付款后 48 小时内发出，预售商品的发货时间以商品页面标注为准。",
     "现货商品付款后 48 小时内发货，预售商品以商品页面标注的发货时间为准。"),
    ("同义改写C（测试里那条）",
     "普通会员累计消费满 1000 元自动升级为 VIP 会员，等级次日生效。",
     "普通会员消费累计满 1000 元即可自动升级为 VIP 会员，等级在次日生效。")  ,
    ("不同条款·不该拦（实测最高）",
     "第二十四条 考试（考查）中有以下情节之一的，按作弊处理，给予警告及以上处分。",
     "第二十五条 发生以下考试作弊行为之一的，给予开除学籍处分。"),
]

texts = []
for name, a, b in CASES:
    texts.extend([a, b])
vecs = np.array(list(model.passage_embed(texts)), dtype="float32")

print(f"{'类别':<28}{'余弦':>8}   判定（阈值 0.96）")
print("-" * 62)
for k, (name, a, b) in enumerate(CASES):
    cos = float(vecs[2 * k] @ vecs[2 * k + 1])
    verdict = "拦" if cos > 0.96 else "放过"
    print(f"{name:<28}{cos:>8.4f}   {verdict}")
