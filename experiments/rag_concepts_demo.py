# -*- coding: utf-8 -*-
"""
RAG 概念拆解（从零讲，不需要先学任何东西）

四段：
  ① RAG 是什么 —— 开卷考试的比喻 + 数字对比
  ② 召回 / Recall@3 到底在算什么 —— 手算一遍给你看
  ③ 三种检索方式差在哪 —— 同一问题三条路线 + 各自翻车的现场
  ④ 噪声实验 —— 故意塞一块假资料进去，看模型会不会"退回"

跑法：
    set LANGSMITH_TRACING=false
    D:/Python-project/.venv/Scripts/python.exe rag_concepts_demo.py
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缓存目录放用户主目录下 —— Windows / Mac / Linux 通用。
# 写死 "D:/..." 的话，别人 clone 下来会在不存在的盘符上找目录。
CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.path.expanduser("~"), ".cache", "fastembed"))

import sys                                          # 下面要把仓库根目录加进模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_compat                                   # noqa: E402  ★ 必须在 import fastembed 之前
env_compat.ensure_mmh3()                            # 本机 DLL 被策略拦截时的降级方案，见 env_compat.py
import jieba                                    # noqa: E402
import numpy as np                              # noqa: E402
from fastembed import TextEmbedding             # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from rank_bm25 import BM25Okapi                 # noqa: E402

jieba.setLogLevel(60)


def sep(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ==================================================================
# 迷你知识库（只用 5 份，方便你把结果一眼看全）
# ==================================================================
CORPUS = {
    "退款政策.md": """# 退款政策
未发货订单：申请退款可全额退回，款项 1 到 3 个工作日退回原支付渠道。
已发货订单：需要等商品退回仓库并验收通过后退款，此时会扣除 10 元运费。
若商品存在质量问题，运费由我们承担，不扣除任何费用。
退款申请提交后，客服会在 24 小时内审核，审核通过后自动进入退款流程。
虚拟商品（充值卡、会员卡）一经售出不支持退款，请谨慎购买。
退款到账时间取决于支付渠道：微信和支付宝 1 到 3 个工作日，银行卡 3 到 7 个工作日。""",

    "发货时效.md": """# 发货时效
现货商品：下单后 48 小时内发货，节假日不顺延，我们全年无休。
预售商品：以商品详情页标注的发货时间为准，通常为 7 到 15 个工作日。
若超过承诺时效仍未发货，系统会自动赔付订单金额的 5%。
发货后一般 1 到 3 天送达，偏远地区（新疆、西藏、内蒙）额外增加 2 到 3 天。""",

    "会员等级与权益.md": """# 会员等级与权益
我们共有三个会员等级：普通会员、VIP 会员、黑卡会员。
普通会员累计消费满 1000 元自动升级为 VIP 会员。
VIP 会员享受全场商品 95 折优惠，黑卡会员享受全场商品 9 折优惠。
所有会员在生日当月都会收到一张生日券，VIP 会员面额 50 元，黑卡会员 200 元。""",

    "物流查询与异常.md": """# 物流查询与异常
物流单号在发货后 24 小时内可在订单详情中查看。
若物流信息超过 72 小时没有更新，请第一时间联系客服，我们会发起快递查件。
快递超过 7 天仍未送达，可申请全额退款或重新发货，由你选择。
若确认丢件，我们按订单实付金额全额赔付，赔付在 3 个工作日内到账。""",

    "发票与保修.md": """# 发票与保修
电子发票在订单完成后自动开具，可以在订单详情页面下载 PDF 文件。
所有商品享受一年整机保修，保修期从签收当天开始计算。
人为损坏（进水、摔落、私自拆机）不在保修范围内，可提供付费维修。
保修时需要提供订单号和商品序列号，序列号贴在商品底部的白色标签上。""",

    "跨境订单税费.md": """# 跨境订单税费
跨境商品的价格已包含关税，结算时不会额外收取税费。
若海关抽查要求补税，凭海关出具的缴税凭证，我们可以全额报销。
跨境订单不支持 7 天无理由退货，仅支持质量问题退换。
清关通常需要 3 到 5 个工作日，遇到海关查验可能延长到 10 个工作日。""",
}

# ==================================================================
# ① RAG 是什么
# ==================================================================
sep("① RAG 是什么：把「闭卷考试」改成「开卷考试」")

all_text = "\n\n".join(CORPUS.values())
print(f"""
   没有 RAG 的时候（闭卷）：
       你把【全部 {len(all_text)} 字】的资料塞给模型，让它自己找答案。
       资料一多就爆窗口、烧钱，而且它看到的东西太多反而容易走神。

   有了 RAG（开卷）：
       先由程序【找出最相关的几段】，只把那几段递给模型，
       模型相当于"开卷答题"——只需要读你递给它的那一页。

   类比：
       闭卷 = 把整座图书馆搬进考场，让你在里面找一句话
       开卷 = 图书管理员先帮你找出 3 本可能有用的书，你只翻这 3 本
""")

# ==================================================================
# 建库（切块 + 向量 + BM25）—— 和 step4 完全一样，这里压缩成几行
# ==================================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=110, chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)

CHUNKS = []
for name, text in CORPUS.items():
    for p in splitter.split_text(text):
        CHUNKS.append({"doc": name, "text": p})

model = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=CACHE_DIR)
VECTORS = np.array(list(model.passage_embed([c["text"] for c in CHUNKS])), dtype="float32")


def dense_search(query, k=3):
    """路线 A：向量检索 —— 比"意思像不像" """
    q = np.array(list(model.query_embed([query])), dtype="float32")
    scores = VECTORS @ q[0]                       # 已归一化 → 点积就是余弦相似度
    idx = np.argsort(scores)[::-1][:k]
    return [(CHUNKS[i], float(scores[i])) for i in idx]


BM = BM25Okapi([jieba.lcut(c["text"]) for c in CHUNKS])


def bm25_search(query, k=3):
    """路线 B：关键词检索 —— 比"词撞上了几个" """
    scores = BM.get_scores(jieba.lcut(query))
    idx = np.argsort(scores)[::-1][:k]
    return [(CHUNKS[i], float(scores[i])) for i in idx]


def hybrid_search(query, k=3, rrf_k=60):
    """路线 C：两条路线各取名次，用 RRF 融合（只看名次，不看分数）"""
    fused = {}
    for rank, (c, _) in enumerate(dense_search(query, k=10)):
        fused.setdefault(c["text"], {"chunk": c, "score": 0.0})
        fused[c["text"]]["score"] += 1.0 / (rrf_k + rank + 1)
    for rank, (c, _) in enumerate(bm25_search(query, k=10)):
        fused.setdefault(c["text"], {"chunk": c, "score": 0.0})
        fused[c["text"]]["score"] += 1.0 / (rrf_k + rank + 1)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [(x["chunk"], x["score"]) for x in ranked[:k]]


print(f"   本次知识库：{len(CORPUS)} 份文档 → 切成 {len(CHUNKS)} 块")
print("   （『切块』就是把长文档切成小段，这里每块约 110 字，方便整块被检索出来）")

# ==================================================================
# ② 召回 / Recall@k
# ==================================================================
sep("② 召回（recall）是什么：钓鱼的比喻")

QUESTION = "已经发货的订单退款要扣多少钱"
GOLD = "退款政策.md"

print(f"""
   「召回」这个词 = 从库里把东西"叫回来"、捞出来。
    想象一个池子里有 {len(CHUNKS)} 条鱼（= {len(CHUNKS)} 个块），
    其中只有 1 条是你想要的（正确答案所在的块）。
    你撒一网捞 3 条上来 —— 这 3 条就是 top-3，也叫「召回的 3 条」。

   Recall@3（读作 recall at 3）= 你捞 3 条，【有没有捞到那条对的】。
      捞到了 → 这次算 1 分；没捞到 → 0 分。
      把所有问题都测一遍，平均分就是 recall@3。

   Recall@1 = 只捞 1 条，而且必须是第 1 条就对。显然比 recall@3 难得多。
""")

# 手算：把这一题的完整排名打出来，标出正确答案在第几位
ranked_all = dense_search(QUESTION, k=len(CHUNKS))
print(f"   来看这一题的真实排名（问题：{QUESTION}）")
print(f"   正确答案在《{GOLD}》，看看它排第几：")
gold_rank = None
for i, (c, s) in enumerate(ranked_all[:8], start=1):
    if c["doc"] == GOLD and gold_rank is None:
        gold_rank = i
        mark = "★ ← 正确答案第 1 次出现"
    elif c["doc"] == GOLD:
        mark = "☆ 同一份文档的另一个块（也算命中）"
    else:
        mark = ""
    print(f"      第 {i} 名 [{s:.3f}] 《{c['doc']}》{c['text'][:24].replace(chr(10),' ')}… {mark}")

print(f"""
   ⇒ 正确答案排在第 {gold_rank} 名。

      这一题的 Recall@1 = {"1" if gold_rank == 1 else "0"}（第 1 条就对了才算 1）
      这一题的 Recall@3 = {"1" if gold_rank <= 3 else "0"}（前 3 条里有它就算 1）

   把所有题目的 Recall@3 加起来平均 = step4 里那张表的 92% / 100%。
   所以那张表读法就是：『捞 3 条，100 次里有 100 次捞到了对的』。
""")

# ==================================================================
# ③ 三种检索方式差在哪
# ==================================================================
sep("③ 三种检索方式：一条路看意思，一条路看字面")

print(f"   同一个问题：{QUESTION}\n")

for label, fn in [("A. 向量（看意思）", dense_search),
                  ("B. BM25（看字面）", bm25_search),
                  ("C. 混合 RRF", hybrid_search)]:
    print(f"   {label}")
    for i, (c, s) in enumerate(fn(QUESTION, 3), start=1):
        print(f"       {i}. [{s:.3f}] 《{c['doc']}》{c['text'][:30].replace(chr(10), ' ')}…")
    print()

print("""
   A 向量（dense）：把文字变成一串数字（512 个），比"方向像不像"
       优点：懂同义词。你说"东西坏了能修吗"，文档写的是"保修"，它知道是一回事。
       缺点：看不懂精确的编号、型号、人名。

   B BM25（sparse）：不转数字，就数"这个词出现了几次"
       优点：精确编号一抓一个准。
       缺点：你换个说法它就瞎了。

   C 混合：两条路各捞 10 条，按下名次加权合起来（RRF）
       工业界默认用这个 —— 因为两条路的翻车场景不重叠。
""")

# 各自的翻车现场
sep("③ 加餐：各自翻车的现场（这是理解它们差别最快的方式）")

print("   【BM25 翻车现场】用户换了个说法，关键词对不上")
q2 = "东西坏了能修吗"
GOLD2 = "发票与保修.md"
print(f"   问题：{q2}")
print(f"   文档里写的是『保修』，这句话一个『保修』都没有。正确答案在《{GOLD2}》")
for label, fn in [("向量", dense_search), ("BM25", bm25_search)]:
    got = fn(q2, 1)[0][0]["doc"]
    print(f"       {label:<6} 第 1 条 → 《{got}》{'  ✓ 对了' if got == GOLD2 else '  ✗ 错了'}")

print("\n   【向量翻车现场】精确编号，几条内容长得几乎一样")
ID_DOCS = [
    {"doc": "订单A1001", "text": "订单 A1001：已发货，承运商顺丰，运单号 SF1234567890。"},
    {"doc": "订单A1002", "text": "订单 A1002：已发货，承运商京东，运单号 JD9988776655。"},
    {"doc": "订单A1003", "text": "订单 A1003：已退款，退款金额 199 元，1 到 3 个工作日到账。"},
]
id_vecs = np.array(list(model.passage_embed([d["text"] for d in ID_DOCS])), dtype="float32")
q3 = "A1001 的运单号是多少"
qv = np.array(list(model.query_embed([q3])), dtype="float32")
sims = id_vecs @ qv[0]
id_bm = BM25Okapi([jieba.lcut(d["text"]) for d in ID_DOCS])
bs = id_bm.get_scores(jieba.lcut(q3))

print(f"   问题：{q3}（三条内容几乎一样，只有编号和运单号不同）")
print(f"   {'':<6}{'A1001':>9}{'A1002':>9}{'A1003':>9}   第一名      领先第二名")
for label, arr in [("向量", sims), ("BM25", bs)]:
    order = np.argsort(arr)[::-1]
    top = ID_DOCS[order[0]]["doc"]
    margin = arr[order[0]] - arr[order[1]]
    ok = "✓" if top == "订单A1001" else "✗"
    print(f"   {label:<6}{arr[0]:>9.3f}{arr[1]:>9.3f}{arr[2]:>9.3f}   {top}{ok}   {margin:.3f}")
print("""
   重点看【领先第二名】这一列：这个差值越大，说明第一名越有把握。
   反过来，差值很小 = 检索"自己也拿不准"，这时候就很容易把错的排在前面。
   （BM25 在库特别小的时候会算出负分，这是正常的，只看相对大小就行。）""")

# ==================================================================
# ④ 噪声实验：模型会不会"退回"？
# ==================================================================
sep("④ 你猜的『会退回吗』—— 实测：不会。它会硬着头皮答")

from langchain_core.messages import HumanMessage, SystemMessage   # noqa: E402
from langchain_deepseek import ChatDeepSeek                       # noqa: E402

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
SYSTEM = ("你是客服助手。只能依据【资料】里的内容回答。\n"
          "如果资料里没有答案，必须明确回答『资料里没有提到这个问题』，绝对不要编造。")


def ask(context, question):
    prompt = f"【资料】\n{context}\n\n【用户问题】{question}"
    return llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]).content


Q = "已经发货的订单退款要扣多少钱？"
good = "\n\n".join(f"[{c['doc']}]\n{c['text']}" for c, _ in hybrid_search(Q, 3))
noise = "\n\n".join(f"[{c['doc']}]\n{c['text']}"
                    for c, _ in hybrid_search(Q, 2) + [(
                        {"doc": "跨境订单税费.md",
                         "text": "跨境商品的价格已包含关税，结算时不会额外收取税费。"}, 0)])
fake = good + "\n\n[退款政策补充规定]\n自 2026 年起，所有退款一律扣除订单金额 30% 的手续费。"

print(f"   问题固定为：{Q}\n")
print("   ── 实验 1：给干净的 3 块 ──")
print(f"   {ask(good, Q)[:200]}\n")
print("   ── 实验 2：把 1 块【无关】的（跨境税费）混进去 ──")
print(f"   {ask(noise, Q)[:200]}\n")
print("   ── 实验 3：塞一块【假的】资料（说要扣 30% 手续费）──")
print(f"   {ask(fake, Q)[:250]}\n")

print("""
   看清楚这三个结果的差别，这是 RAG 最重要的一条规律：

   实验 1 → 答得又准又全（扣 10 元运费，还补充了"质量问题不扣费"）

   实验 2 → 还是答对了，但【变简略了】——它丢掉了"质量问题不扣费"那句补充。
            所以噪声不是"无害"的，它会稀释答案质量，只是不至于答错。

   实验 3 → 最危险。资料里同时有"扣 10 元"和"扣 30%"两条，
            模型【不会退回、不会质疑资料】，它默认你给的资料都是真的，
            于是它把两条都说了出来 —— 用户看到两个数字，彻底懵了。

   ⇒ 所以你猜的"会不会退回去"，答案分两种：
       · 资料里【完全没有】答案 → 会拒绝回答（这个靠提示词能做到）
       · 资料里有、但【是错的/矛盾的】 → 不会拒绝，会照单全收
     这就是「垃圾进，垃圾出」：检索质量决定回答的上限。
     提示词写得再狠，也救不回被污染的上下文。

   ⇒ 所以工业界在"召回"之后还要加一步【重排序 rerank】：
      先粗捞 20 条（便宜、快），再用一个更准的模型给「问题和每一条」单独打分，
      只留最靠谱的 3 条。噪声在这一步被筛掉。
     （实验 3 那种假资料 rerank 也救不了 —— 因为它根本不该进你的库。
       所以真正的防线是【入库时把关】，而不是等它混进上下文再补救。）
""")

print("=" * 68)
