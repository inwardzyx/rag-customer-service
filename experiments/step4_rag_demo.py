# -*- coding: utf-8 -*-
"""
Step 4：RAG 完整链路（真家伙版）

    文档 → 切块(chunk) → 转向量(embedding) → 存向量库 → 用时检索(top-k)

本次用到的真库（都已装好）：
    langchain-text-splitters  递归字符切块器（工业标准切法）
    fastembed + bge-small-zh  真中文 embedding 模型，512 维，本地 ONNX 推理
    faiss-cpu                 Meta 的向量索引（本机会被安全策略拦，已做自动降级）
    rank_bm25 + jieba         稀疏检索（关键词路线），和向量路线做混合
    langgraph + deepseek      检索结果喂给真大模型作答

跑法（务必先关 trace）：
    set LANGSMITH_TRACING=false
    python step4_rag_demo.py

首次运行会加载本地模型（约 1 秒，之后缓存在用户主目录下的 ~/.cache/fastembed）。
"""

import os
import time
from pathlib import Path
from typing import TypedDict

# ⚠️ 这两行必须在 import 之前：告诉下载器走国内镜像，并指定模型缓存位置
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

# FAISS 是"可选加速件"：这台机器的应用程序控制策略有时会拦掉它的 DLL，
# 拦了就退回 numpy —— 几万块以内 numpy 完全够用，真到百万级再想办法装 FAISS。
try:
    import faiss

    HAVE_FAISS = True
except Exception as e:                          # ImportError / OSError(DLL 被拦) 都算
    faiss = None
    HAVE_FAISS = False
    FAISS_ERR = f"{type(e).__name__}: {e}"

jieba.setLogLevel(60)                           # 关掉 jieba 的加载日志

DOCS_DIR = Path(__file__).parent / "step4_docs"

# ==================================================================
# 知识库：10 份文档（脚本自动写到 step4_docs/，你可以打开看）
# 故意写得长一点 —— 文档太短的话，"切多大一块"这件事就没差别了
# ==================================================================
CORPUS = {
    "会员等级与权益.md": """# 会员等级与权益
我们共有三个会员等级：普通会员、VIP 会员、黑卡会员。
普通会员累计消费满 1000 元自动升级为 VIP 会员，VIP 会员年消费满 20000 元升级为黑卡会员。
会员等级在达到条件的第二天凌晨自动生效，不需要手动申请。

VIP 会员享受全场商品 95 折优惠，黑卡会员享受全场商品 9 折优惠。
折扣在结算页面自动计算，不需要手动领取或使用任何券码。
部分特殊商品（黄金、数码新品、限量款）不参与会员折扣，商品详情页会单独标注。

所有会员在生日当月都会收到一张生日券。
VIP 会员的生日券面额为 50 元，黑卡会员为 200 元，普通会员为 20 元。
生日券有效期为发放后 30 天，过期作废不补发，且不能与其他优惠券叠加使用。

会员等级每年 1 月 1 日重新计算，以过去 12 个月的累计消费金额为准。
若上一年消费未达到当前等级的门槛，会在 1 月 5 日降级并短信通知。
升级永远即时生效，降级一年只做一次。""",

    "发货时效.md": """# 发货时效
现货商品：下单后 48 小时内发货，节假日不顺延，我们全年无休。
预售商品：以商品详情页标注的发货时间为准，通常为 7 到 15 个工作日。
定制商品：需要 15 到 30 个工作日，具体时间由客服在下单后 24 小时内电话确认。

若超过承诺时效仍未发货，系统会自动赔付订单金额的 5%。
赔付以无门槛券形式发放到账户，有效期 90 天，可以叠加使用。
赔付不需要申请，系统每天凌晨扫描一次超时订单并自动发放。

发货后一般 1 到 3 天送达，偏远地区（新疆、西藏、内蒙）额外增加 2 到 3 天。
遇到极端天气、疫情管控等不可抗力，送达时间顺延，我们会在订单页推送通知。

发货后 2 小时内可以自助修改收货地址，超过 2 小时需要联系客服拦截。
已经出库的商品无法改地址，只能等送达后申请拒收退回。""",

    "退款政策.md": """# 退款政策
未发货订单：申请退款可全额退回，款项 1 到 3 个工作日退回原支付渠道。
已发货订单：需要等商品退回仓库并验收通过后退款，此时会扣除 10 元运费。
若使用过优惠券，退款金额按实付金额计算，优惠券会在 3 个工作日内退回账户。

若商品存在质量问题，来回运费全部由我们承担，不扣除任何费用。
质量问题需要提供照片或视频证据，客服审核通过后生成免费退货上门取件单。
虚拟商品（充值卡、会员卡、软件授权）一经售出不支持退款，请谨慎购买。

退款申请提交后，客服会在 24 小时内审核，审核通过后自动进入退款流程。
退款到账时间取决于支付渠道：微信和支付宝 1 到 3 个工作日，银行卡 3 到 7 个工作日。
超过 7 个工作日仍未到账，请提供退款单号联系客服查询。

已经退款的订单无法撤销，如需继续购买请重新下单。
同一订单只能申请一次退款，部分退款需要联系客服人工处理。""",

    "换货流程.md": """# 换货流程
支持 7 天无理由换货，时间从签收当天开始计算，第 8 天起不再受理。
换货前请先拍照：商品外观、吊牌、外包装，共三张照片，缺一不可。
照片需要清晰可见，模糊或过度曝光的照片会被驳回并要求重新提交。

换货商品必须保持吊牌完整、未洗涤、未影响二次销售。
内衣、袜子、化妆品等贴身或易耗品一经拆封不支持换货，除非存在质量问题。
换货时可以选择同款不同尺码或颜色，不支持换成完全不同的商品。

换货产生的来回运费由买家承担，除非商品本身存在质量问题。
运费在寄回时先行垫付，换货完成后以无门槛券形式返还，需要主动申请。
换货申请通过后，请于 5 天内寄回，逾期视为放弃换货，申请自动关闭。

换货商品寄出后，新商品会在仓库验收合格后 2 个工作日内发出。
整个换货流程通常需要 7 到 10 天，可以在订单详情页查看进度。""",

    "优惠券使用规则.md": """# 优惠券使用规则
优惠券分为满减券、折扣券、无门槛券三类，获取途径包括活动领取、会员赠送、积分兑换。
同一个订单只能使用一张优惠券，不支持叠加使用，也不支持与会员折扣同时享受。
系统会默认帮你选择最优惠的那张券，也可以在结算页手动切换。

满减券需满足门槛金额才能使用，门槛按商品原价计算，不含运费。
部分商品（黄金、数码新品）被标记为特殊商品，不参与任何满减活动。
跨店满减只统计参与活动的商品金额，未参与的商品不计入门槛。

优惠券有效期一般为 30 天，具体以券面标注为准，过期后无法延期。
过期前 3 天会通过短信和站内信提醒，建议提前使用避免浪费。
优惠券不能转赠他人，也不能提现或折换成现金。

订单退款后，使用的优惠券会在 3 个工作日内退回账户，有效期内可继续使用。
若优惠券在退款时已经过期，则不再返还，这一点请特别注意。""",

    "物流查询与异常.md": """# 物流查询与异常
物流单号在发货后 24 小时内可在订单详情中查看，也可以在快递公司官网查询。
若发货后 48 小时仍查不到单号，可能是仓库漏扫，请联系客服核实。

若物流信息超过 72 小时没有更新，请第一时间联系客服，我们会发起快递查件。
查件通常需要 1 到 2 个工作日，结果会通过短信通知你。
常见的停滞原因是中转站爆仓或面单破损，多数情况下会自行恢复。

快递超过 7 天仍未送达，可申请全额退款或重新发货，由你选择。
若确认丢件，我们按订单实付金额全额赔付，赔付在 3 个工作日内到账。
赔付不需要你提供快递公司的证明，我们直接与承运商结算。

签收时请当面验货，若外包装破损请直接拒收并拍照留证。
已签收后发现商品破损，需要在 48 小时内提交照片证据，否则难以认定责任。""",

    "发票与保修.md": """# 发票与保修
电子发票在订单完成后自动开具，可以在订单详情页面下载 PDF 文件。
需要增值税专用发票的企业客户，请在下单时备注税号和公司全称。
发票开具后 30 天内可以申请换开，超过 30 天不再受理。

所有商品享受一年整机保修，保修期从签收当天开始计算。
人为损坏（进水、摔落、私自拆机）不在保修范围内，可提供付费维修。
保修时需要提供订单号和商品序列号，序列号贴在商品底部的白色标签上。

保修期内出现非人为故障，来回运费由我们承担。
维修周期通常为 7 到 15 个工作日，复杂故障可能延长到 30 天。
超过 30 天无法修复的，可以换新或全额退款，由你选择。

付费维修会先出检测报告并报价，你确认后才会开始维修。
若不认可报价，可以选择不修，只需承担 20 元检测费和退回运费。""",

    "账号与安全.md": """# 账号与安全
忘记密码可以在登录页点击"忘记密码"，通过手机号验证码重置。
验证码 5 分钟内有效，一小时内最多发送 5 次，超出请稍后再试。
若手机号同时停用，只能通过人工申诉找回账号。

一个手机号只能注册一个账号，实名认证后不可修改。
若手机号停用，需要人工申诉，请准备身份证正反面照片和最近一笔订单号。
申诉处理时间为 3 到 5 个工作日，结果会通过邮件通知。

账号可以注销，注销后所有订单记录和会员权益清空且不可恢复。
注销前请先处理完所有进行中的订单和售后申请，否则会造成损失。
注销申请提交后有 7 天冷静期，期间登录即可撤销注销。

建议开启两步验证，防止账号被盗用造成损失。
不要使用与其他网站相同的密码，也不要把验证码告诉任何人。
我们绝不会主动索要你的密码或验证码，遇到此类情况请立即举报。""",

    "跨境订单税费.md": """# 跨境订单税费
跨境商品的价格已包含关税，结算时不会额外收取税费。
若海关抽查要求补税，凭海关出具的缴税凭证，我们可以全额报销。
报销需要提供缴税凭证照片和订单号，审核通过后 5 个工作日内到账。

跨境订单不支持 7 天无理由退货，仅支持质量问题退换。
下单前请务必确认尺码和型号，避免因尺码不合造成无法退货。
部分国家（地区）因海关政策限制，我们暂时无法发货，下单时会提示。

清关通常需要 3 到 5 个工作日，遇到海关查验可能延长到 10 个工作日。
超出 15 个工作日仍未清关的，可以联系我们申请全额退款。
清关失败被退回的商品，我们会在收到退件后 3 个工作日内退款。

跨境订单的物流单号在清关完成后才会更新，请耐心等待。
在清关期间查询物流可能显示"无记录"，这是正常的，不是漏发货。""",

    "售后联系方式.md": """# 售后联系方式
在线客服工作时间：每天 9:00 到 22:00，全年无休。
客服电话 400-123-4567，工作日 9:00 到 18:00 有人接听，周末仅上午。
节假日电话客服可能排队较长，建议优先使用在线客服。

紧急情况可以发邮件到 support@example.com，我们在 12 小时内回复。
邮件请附上订单号和问题描述，附上照片可以大幅加快处理速度。
垃圾邮件过滤可能导致漏收，若 24 小时没回复请换个渠道再联系一次。

复杂问题建议提交工单，工单会在 48 小时内由专人处理并电话回访。
工单适合处理纠纷、投诉、批量售后等需要留痕的场景。
工单状态可以在"我的工单"页面实时查看，每次回复都会短信提醒。

投诉建议请直接致电客服主管专线 400-123-4568。
我们承诺投诉在 3 个工作日内给出处理方案，超时可申请升级处理。""",
}


def sep(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


DOCS_DIR.mkdir(exist_ok=True)
for name, text in CORPUS.items():
    (DOCS_DIR / name).write_text(text, encoding="utf-8")   # ⚠️ Windows 必须写 encoding

# ==================================================================
# ① 问题有多大
# ==================================================================
sep("① 为什么必须做 RAG：全文塞进去要多少钱")
all_text = "\n\n".join(CORPUS.values())
print(f"   现在：{len(CORPUS)} 份文档，共 {len(all_text)} 字 ≈ {int(len(all_text) * 0.7)} tokens")
print(f"   真实项目：300 份 → 约 30 倍 → 一次提问就吃掉几万 token")
print(f"   而且多轮对话【每轮都要重发一遍历史】，成本是线性叠加的。")
print()
print("   ⇒ RAG 的全部动机就一句：只把相关的那几段塞进上下文。")

# ==================================================================
# ② 切块：手写硬切 vs 工业切块器
# ==================================================================
sep("② 切块：硬切 vs RecursiveCharacterTextSplitter")


def chunk_hard(text, size=80):
    """方案 A：不看内容，每 80 字一刀（会切断句子）"""
    return [text[i:i + size] for i in range(0, len(text), size)]


def chunk_recursive(text, size=300, overlap=50):
    """方案 B：递归字符切块器 —— 先按段落切，切不开再按句子，最后才按字硬切"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        # separators 的顺序就是"退让顺序"：先试空行，再换行，再句号……
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


sample = CORPUS["退款政策.md"]
print(f"   同一份《退款政策.md》（{len(sample)} 字）")
print(f"\n   硬切 80 字 → {len(chunk_hard(sample))} 块，看其中两块：")
for c in chunk_hard(sample)[:2]:
    print(f"       「…{c[-40:]}」")
print(f"\n   递归切块 300 字/重叠 50 → {len(chunk_recursive(sample))} 块，看其中两块：")
for c in chunk_recursive(sample)[:2]:
    print(f"       「{c[:52]}…」")
print()
print("   硬切的问题：句子被拦腰截断，比如『此时会扣除 10 元运』+『费。』，")
print("   单独召回任何一半都不完整 → 模型只能瞎猜。")
print("   递归切块器会优先在句号、换行处下刀，保住语义完整。")


def build_chunks(method, size=300, overlap=50):
    """把整个知识库切成带来源的块（每个块记住自己来自哪份文档）"""
    out = []
    for name, text in CORPUS.items():
        pieces = chunk_hard(text) if method == "hard" else chunk_recursive(text, size, overlap)
        for p in pieces:
            out.append({"doc": name, "text": p})
    return out


# ==================================================================
# ③ 向量化 + 向量库
# ==================================================================
sep("③ 转向量：bge-small-zh-v1.5（真中文 embedding 模型）")

t0 = time.time()
model = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=CACHE_DIR)
print(f"   模型加载：{time.time() - t0:.2f} 秒（已缓存，不联网）")

demo_vecs = np.array(list(model.passage_embed(["退款要扣多少运费"])), dtype="float32")
print(f"   向量形状：{demo_vecs.shape}  ← 512 维的 float32 数组")
print(f"   范数 = {np.linalg.norm(demo_vecs[0]):.4f}  ← 已归一化，点积就等于余弦相似度")
print()
print("   归一化是个关键技巧：算相似度时只要做一次点积(a @ b)，")
print("   不用再除以两个模长 —— 向量库正是靠这个提速。")


def _topk(scores, k):
    """取分数最大的 k 个下标，并按分数从高到低排好"""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]        # 只分区不排序，比 argsort 快
    return idx[np.argsort(-scores[idx])]             # 对这 k 个再排一次序


class VectorStore:
    """
    向量库：把每块的向量堆成一个矩阵，检索时做一次矩阵×向量。

    有 FAISS 就用 FAISS（IndexFlatIP = 内积索引）；
    没有就用 numpy 的矩阵乘法 —— 因为向量已归一化，点积就是余弦相似度。
    """

    def __init__(self, chunks):
        self.chunks = chunks
        t = time.time()
        # passage_embed：给"文档块"用的编码（bge 系列对查询和文档用不同前缀）
        self.vectors = np.array(list(model.passage_embed([c["text"] for c in chunks])),
                                dtype="float32")
        self.dim = self.vectors.shape[1]
        if HAVE_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)   # IP = Inner Product 内积
            self.index.add(self.vectors)               # 一次性灌进索引
        self.build_time = time.time() - t

    def search(self, query, k=3):
        q = np.array(list(model.query_embed([query])), dtype="float32")
        if HAVE_FAISS:
            scores, ids = self.index.search(q, k)      # 返回 (分数二维数组, 下标二维数组)
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        scores = self.vectors @ q[0]                   # ← 一次矩阵乘法，全库相似度都出来了
        return [(self.chunks[i], float(scores[i])) for i in _topk(scores, k)]


CHUNKS = build_chunks("recursive")
backend = "FAISS IndexFlatIP" if HAVE_FAISS else "numpy 矩阵乘法"
print(f"\n   检索后端：{backend}")
if not HAVE_FAISS:
    print(f"      原因：{FAISS_ERR}")
    print(f"      几万块以内 numpy 够快；真到百万级再想办法装 FAISS。")

vs = VectorStore(CHUNKS)
print(f"   建库：{len(CHUNKS)} 块 → 维度 {vs.dim}，耗时 {vs.build_time:.2f} 秒")

q = "已经发货的订单退款要扣多少运费？"
print(f"\n   向量检索「{q}」：")
for c, s in vs.search(q, k=3):
    print(f"       [{s:.3f}] 《{c['doc']}》{c['text'][:34].replace(chr(10), ' ')}…")

hit = sum(len(c["text"]) for c, _ in vs.search(q, k=3))
print(f"\n   只塞这 3 块进上下文：{hit} 字 vs 全文 {len(all_text)} 字")
print(f"   ⇒ 省掉 {100 - int(hit / len(all_text) * 100)}% 的 token。300 份文档时差距会放大几十倍。")

# ==================================================================
# ④ 另一条路线：BM25 关键词检索 + 混合检索
# ==================================================================
sep("④ 另一条路线：BM25 关键词检索，以及两者混合")


class BM25Store:
    """BM25：老牌关键词检索。不需要 embedding，靠词频和逆文档频率打分"""

    def __init__(self, chunks):
        self.chunks = chunks
        # jieba 分词：中文必须先切成词，BM25 才知道"退款"是一个词
        self.tokens = [jieba.lcut(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query, k=3):
        scores = self.bm25.get_scores(jieba.lcut(query))
        # argsort 是从小到大，[::-1] 翻过来就是从大到小
        return [(self.chunks[i], float(scores[i])) for i in np.argsort(scores)[::-1][:k]]


bs = BM25Store(CHUNKS)
print(f"   BM25 检索同一个问题：")
for c, s in bs.search(q, k=3):
    print(f"       [{s:.2f}] 《{c['doc']}》{c['text'][:34].replace(chr(10), ' ')}…")

print("""
   两条路线的脾气不一样：
     向量(dense)  → 懂语义同义词，问「快递没动静」也能找到讲物流的文档
                    但它看不懂精确型号、编号、专有名词
     BM25(sparse) → 关键词精确匹配，「A1001」这种编号它一抓一个准
                    但用户换个说法它就瞎了

   所以工业界几乎都用【混合检索】：两条路线各取 top-k，再融合排序。""")


def hybrid_search(store_v, store_b, query, k=3, rrf_k=60):
    """RRF 融合：不看原始分数（两条路线分数不可比），只看排名"""
    fused = {}
    for rank, (c, _) in enumerate(store_v.search(query, k=10)):
        fused.setdefault(c["text"], {"chunk": c, "score": 0.0})
        fused[c["text"]]["score"] += 1.0 / (rrf_k + rank + 1)   # 名次越靠前加分越多
    for rank, (c, _) in enumerate(store_b.search(query, k=10)):
        fused.setdefault(c["text"], {"chunk": c, "score": 0.0})
        fused[c["text"]]["score"] += 1.0 / (rrf_k + rank + 1)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [(x["chunk"], x["score"]) for x in ranked[:k]]


print(f"   混合检索（RRF）：")
for c, s in hybrid_search(vs, bs, q, k=3):
    print(f"       [{s:.4f}] 《{c['doc']}》{c['text'][:34].replace(chr(10), ' ')}…")

# ==================================================================
# ⑤ 评测：这才是能写进简历的东西
# ==================================================================
sep("⑤ 召回评测：14 道题的标准考卷，三条路线 PK")

EVAL = [
    # —— 基础题：关键词和文档高度重合 ——
    ("VIP 会员生日有什么优惠", "会员等级与权益.md"),
    ("预售商品什么时候发货", "发货时效.md"),
    ("已经发货的订单退款要扣多少钱", "退款政策.md"),
    ("我要换货需要准备什么材料", "换货流程.md"),
    ("两张优惠券可以一起用吗", "优惠券使用规则.md"),
    ("商品保修期是多久", "发票与保修.md"),
    ("怎么下载电子发票", "发票与保修.md"),
    ("忘记密码了怎么找回", "账号与安全.md"),
    ("客服几点上班", "售后联系方式.md"),
    # —— 换说法：用户不会照着文档说话 ——
    ("快递一直没动静，好几天了", "物流查询与异常.md"),
    ("我买的东西想退掉，但已经寄出来了", "退款政策.md"),
    ("东西坏了能修吗", "发票与保修.md"),
    ("海淘的东西被海关扣了怎么办", "跨境订单税费.md"),
    ("会员等级怎么升上去", "会员等级与权益.md"),
    # —— 刁钻题：答案埋在文档的第二段、第三段 ——
    ("下单一周了还没发货，有赔偿吗", "发货时效.md"),
    ("退款的钱会退到哪里", "退款政策.md"),
    ("想改收货地址还来得及吗", "发货时效.md"),
    ("账号不想用了能删掉吗", "账号与安全.md"),
    ("客服电话一直打不通怎么办", "售后联系方式.md"),
    ("保修要带什么凭证", "发票与保修.md"),
    ("券过期了还能用吗", "优惠券使用规则.md"),
    ("换货来回运费谁出", "换货流程.md"),
    ("快递超过多少天没到可以退款", "物流查询与异常.md"),
    ("公司要开专票怎么弄", "发票与保修.md"),
]


def evaluate(search_fn, k=3):
    """考卷打分：top1 命中率 + top3 命中率"""
    hit1 = hitk = 0
    wrong = []
    for question, gold in EVAL:
        docs = [c["doc"] for c, _ in search_fn(question, k=k)]
        if docs and docs[0] == gold:
            hit1 += 1
        if gold in docs:
            hitk += 1
        else:
            wrong.append((question, gold, docs[0] if docs else "-"))
    return hit1 / len(EVAL), hitk / len(EVAL), wrong


rows = [
    ("BM25 关键词", lambda qq, k: bs.search(qq, k)),
    ("向量 dense", lambda qq, k: vs.search(qq, k)),
    ("混合 RRF", lambda qq, k: hybrid_search(vs, bs, qq, k)),
]

print(f"{'检索方式':<14}{'第1条就命中':<14}{'前3条里命中':<14}")
print("-" * 42)
results = {}
for label, fn in rows:
    r1, r3, wrong = evaluate(fn)
    results[label] = (r1, r3, wrong)
    print(f"{label:<14}{r1:>8.0%}      {r3:>10.0%}")

print("\n   ↑ 这组数字就是简历/面试上那句话的来源：")
print(f"     『我用 {len(EVAL)} 条考卷做召回评测，把 recall@1 从 "
      f"{results['BM25 关键词'][0]:.0%} 提到 {results['混合 RRF'][0]:.0%}』")
print("     没有这张表，你只能说『我做了个 RAG』—— 那是人人都写的一句废话。")

print("\n   错题分析（混合检索没在 top3 找到的）：")
if results["混合 RRF"][2]:
    for question, gold, got in results["混合 RRF"][2]:
        print(f"       ✗ {question[:22]:<24} 期望《{gold}》 实际《{got}》")
else:
    print("       （top3 全对）")
    print("""
   ⚠️ 注意这个现象：库太小（只有 20 块）时，top3 必中，recall@3 会【饱和】。
   这不是你的检索做得好，是考卷失去区分度了。真实项目里：
       · 库变大（几百上千块），recall@3 才有意义
       · 或者改用更严的指标：recall@1（本例 88%）/ MRR
       · 一句话：指标要用【能拉开差距】的那一个，别挑好看的报。""")

print("\n   再测一个变量：切块大小对召回的影响（混合检索）")
print(f"   {'块大小':<8}{'重叠':<8}{'块数':<8}{'recall@1':<10}{'recall@3':<10}")
for size in (80, 120, 200, 300, 400):
    ch = build_chunks("recursive", size=size, overlap=size // 6)
    vs2 = VectorStore(ch)
    bs2 = BM25Store(ch)
    r1, r3, _ = evaluate(lambda qq, k=3: hybrid_search(vs2, bs2, qq, k))   # 立即调用，不受迟绑定影响
    print(f"   {size:<8}{size // 6:<8}{len(ch):<8}{r1:>6.0%}    {r3:>6.0%}")
print("\n   ⇒ 块太小语义被切断，块太大噪声变多。挑『召回涨不动』的那个拐点。")
print("      本例 recall@3 已经饱和，所以要看 recall@1 那一列的差别。")

# ==================================================================
# ⑥ 接进 LangGraph
# ==================================================================
sep("⑥ 接进 LangGraph：检索节点 → 回答节点")

from langchain_core.messages import HumanMessage, SystemMessage   # noqa: E402
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek                       # noqa: E402
from langgraph.graph import END, START, StateGraph                # noqa: E402


class State(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state):
    """节点 1：检索（纯本地，不花钱）"""
    hits = hybrid_search(vs, bs, state["question"], k=3)
    ctx = "\n\n".join(f"[{c['doc']}] {c['text']}" for c, _ in hits)
    print(f"   [retrieve] 召回 {len(hits)} 块 / {len(ctx)} 字（全文是 {len(all_text)} 字）")
    return {"context": ctx}


SYSTEM = (
    "你是客服助手。只能依据【资料】里的内容回答。\n"
    "如果资料里没有答案，必须明确回答『资料里没有提到这个问题』，绝对不要自己编造。\n"
    "回答时在末尾注明信息来源的文档名。"
)

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


def answer(state):
    """节点 2：资料 + 问题一起交给模型"""
    prompt = f"【资料】\n{state['context']}\n\n【用户问题】{state['question']}"
    resp = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
    return {"answer": resp.content}


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("answer", answer)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "answer")
g.add_edge("answer", END)
app = g.compile()

for qq in ["已经发货的订单退款要扣多少钱？多久到账？",
           "我买的东西坏了，能修吗？",
           "你们支持分期付款吗？"]:          # ← 故意问知识库里没有的
    print(f"\n   用户问：{qq}")
    try:
        out = app.invoke({"question": qq, "context": "", "answer": ""})
        print(f"   回答：{out['answer'][:200]}")
    except Exception as e:
        print(f"   [调用失败 {type(e).__name__}] {str(e)[:80]}（网络/额度问题，检索部分不受影响）")

print("""
   第三问是故意的：资料里根本没有「分期付款」。
   看它有没有老实说「没有提到」—— 这是 RAG 最重要的防线。
   如果它开始编，说明 SYSTEM 提示词的约束还不够硬，要继续加码。""")

# ==================================================================
# ⑦ 坑
# ==================================================================
sep("⑦ 这四步里最容易踩的坑")
print("""
1. 【切块】不是越小越好。中文经验值 200~500 字/块，重叠 10%~20%。
   重叠是为了防止答案正好横跨两个块。

2. 【top-k】不是越大越好。k 大 → 召回率高，但噪声和 token 也多。
   正确做法：用考卷测 k=1/3/5/10，挑「召回率涨不动」的拐点。

3. 【召不回就一定编】。所以 SYSTEM 必须写死「没有就说没有」，
   而且这句话要用考卷验证过（第 ⑥ 段第三问）。

4. 【换 embedding = 全库重算】。所以原文一定要留着，向量只当缓存。

5. 【Windows 读文件永远写 encoding="utf-8"】，不写会用 GBK → UnicodeDecodeError。

6. 【混合检索的分数不能直接相加】：向量分是 0~1 的余弦，BM25 分能到十几。
   所以要用 RRF（只看名次不看分数），这就是 1/(60+rank) 的由来。

7. 【评测集要先于优化存在】。没有考卷就去调参数，等于闭着眼睛调，
   你根本不知道改了是变好还是变坏。

8. 【循环图 + 开着 LANGSMITH_TRACING = 卡死】。跑练习脚本前先 set LANGSMITH_TRACING=false。
""")

print("=" * 68)
print(f"知识库文件：{DOCS_DIR}")
print(f"模型缓存：{CACHE_DIR}（91MB，别删，删了要重下 4 分钟）")
print("=" * 68)
