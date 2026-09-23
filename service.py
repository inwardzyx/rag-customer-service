# -*- coding: utf-8 -*-
"""
Step 6：把 RAG 整条链包成 HTTP 服务（FastAPI）

前面 5 步做的东西都跑在一个 .py 脚本里 —— 只有你自己能用。
这一步把它变成【别人能用 HTTP 访问的服务】：

    你的脚本（自己跑）  →  HTTP 服务（别人访问）
    print 输出结果      →  浏览器 / 手机 / 前端 / 其他程序都能调

三个新名词，一句话解释：
    FastAPI   = 写接口的框架。你写一个函数，它帮你变成 HTTP 接口
    uvicorn   = 真正跑起来的服务器（FastAPI 只负责"定义"，uvicorn 负责"运行"）
    pydantic  = 校验请求体的工具（规定"你 POST 过来的 JSON 必须长这样"）

跑法（务必先关 trace，在仓库根目录跑）：
    D:/Python-project/.venv/Scripts/python.exe service.py

跑起来后：
    打开浏览器访问 http://127.0.0.1:8000    ← 一个能聊天的网页
    接口文档（自动生成）http://127.0.0.1:8000/docs
    命令行测试见文件底部注释

按 Ctrl+C 停止服务。
"""

import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

# 先读仓库根目录的 .env（没有这个文件就静默跳过，什么都不影响）。
# 为什么用 .env 而不是 setx / export？
#   · setx 是 Windows 专有命令，服务器（Linux）上不存在 —— 部署时那套直接失效
#   · setx 设完要重开终端才生效，而且改的是系统级环境变量
#   · .env 跟着项目走，clone 下来复制一份就能跑，三个平台一致
# 已经有同名环境变量时不会覆盖它（load_dotenv 默认 override=False），
# 所以想临时换 key，直接设环境变量比改文件方便。
# 模板见 .env.example。
from dotenv import load_dotenv                        # noqa: E402
load_dotenv()

# ⚠️ 必须在 import 之前：走国内镜像 + 指定模型缓存
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缓存目录默认放在【用户主目录】下（Windows / Mac / Linux 通用）。
#   想换位置就设环境变量 FASTEMBED_CACHE_PATH。
#   os.path.expanduser("~") 会把 ~ 展开成当前用户的主目录，
#   比写死 "D:/..." 好 —— 别人 clone 下来不会在你不存在的盘符上找目录。
CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.path.expanduser("~"), ".cache", "fastembed"))

import numpy as np                                  # noqa: E402
import jieba                                        # noqa: E402
import uvicorn                                      # noqa: E402
import env_compat                                   # noqa: E402  ★ 必须放在 import fastembed 之前
env_compat.ensure_mmh3()                            # 本机 DLL 被策略拦截时的降级方案，见 env_compat.py
from fastapi import FastAPI                         # noqa: E402
from fastapi.responses import HTMLResponse          # noqa: E402
from fastembed import TextEmbedding                 # noqa: E402
from langchain_core.messages import HumanMessage    # noqa: E402
from langchain_deepseek import ChatDeepSeek         # noqa: E402
from pydantic import BaseModel, Field               # noqa: E402
from rank_bm25 import BM25Okapi                     # noqa: E402

# ==================================================================
# ① 知识库 + 入库把关（和 Step 5 完全一样的逻辑，这里精简成一份）
# ==================================================================
RAW_DOCS = [
    dict(doc="退款政策.md", clause="退款-已发货", version="2026-03-01", source="官网帮助中心",
         text="已发货的订单申请退款，需扣除 10 元运费，其余金额在 1 到 3 个工作日内原路退回。"),
    dict(doc="退款政策.md", clause="退款-未发货", version="2026-03-01", source="官网帮助中心",
         text="未发货订单可全额退款，不扣任何费用，审核通过后 24 小时内到账。"),
    dict(doc="发货时效.md", clause="现货发货", version="2026-02-10", source="官网帮助中心",
         text="现货商品在付款后 48 小时内发出，预售商品以商品页面标注的发货时间为准。"),
    dict(doc="发票与保修.md", clause="保修范围", version="2026-01-20", source="官网帮助中心",
         text="商品享受一年整机保修，保修期自签收次日开始计算，人为损坏不在保修范围内。"),
    dict(doc="会员等级与权益.md", clause="升级规则", version="2026-02-01", source="官网帮助中心",
         text="普通会员累计消费满 1000 元自动升级为 VIP 会员，等级在达到条件的次日生效。"),
    dict(doc="跨境订单税费.md", clause="税费缴纳", version="2026-01-05", source="官网帮助中心",
         text="跨境订单需缴纳进口税，税费在清关时由承运商代收，具体金额以海关核定为准。"),
    dict(doc="物流查询与异常.md", clause="单号查询", version="2026-02-20", source="官网帮助中心",
         text="物流单号在发货后 24 小时内可查，超过 72 小时未更新可联系客服发起查件。"),
    # ↓ 下面是脏数据，会被入库把关拦掉（故意留着，证明关卡真的在工作）
    dict(doc="退款政策.md", clause="退款-已发货", version="2024-05-01", source="旧版帮助中心（已下线）",
         text="已发货订单申请退款，需扣除订单金额 30% 的手续费，退款周期 15 个工作日。"),
    dict(doc="售后联系方式.md", clause="客户信息", version="2026-02-01", source="客服工单导出",
         text="客户张先生的联系方式是 13812345678，身份证号 440301199001011234，请妥善保管。"),
    dict(doc="退款政策.md", clause="退款-已发货", version="2026-03-01", source="客服话术库",
         text="已发货订单若要退款，会扣 10 元运费，剩下的钱 1 至 3 个工作日退回原支付账户。"),
]

MIN_LEN = 15
DUP_THRESHOLD = 0.90
PII_PATTERNS = [(r"1[3-9]\d{9}", "手机号"), (r"\d{17}[\dXx]", "身份证号"), (r"\d{16,19}", "银行卡号")]

# 关掉 rerank 时，用【向量分】判断要不要拒答（精排分此时不存在）。
# 同样是实测定的，不是拍脑袋：在本项目这 7 条语料上跑 12 个问题 ——
#     该答的（7 题）向量分 0.6541 ~ 0.8425   ← 最低 0.6541
#     该拒的（5 题）向量分 0.2619 ~ 0.4373   ← 最高 0.4373
# 两边中间空着 0.22 的间隔，取中间值 0.55，离两边都留了余量。
# ★ 和 DUP_THRESHOLD=0.90 是同一套做法：先量出分布，再取中间，不拍脑袋。
VEC_REJECT_THRESHOLD = 0.55


def norm(text):
    return re.sub(r"[\s\W_]+", "", text)


# ==================================================================
# ② 全局资源：模型和库只在【服务启动时】加载一次
#    ★ 这是服务化最重要的一条：绝不能在每个请求里重新加载模型！
# ==================================================================
class RAG:
    """把所有重家伙（模型、向量库）装在里面，全局只有一份"""

    def __init__(self):
        self.ready = False
        self._llm = None          # 真正的大模型对象，第一次用到才建（见下面的 llm）

    @property
    def llm(self):
        """大模型【第一次要用的时候才创建】。

        @property 的作用：让 `rag.llm` 用起来像一个普通属性（不用加括号），
        但每次访问它会走一遍这个函数 —— 也就是"用到才建、只建一次"。

        为什么要这么绕，不在 startup() 里直接建？
            ChatDeepSeek(...) 在【构造的时候】就会校验 DEEPSEEK_API_KEY，
            没配 key 直接抛：
                ValidationError: If using default api base, DEEPSEEK_API_KEY must be set.
            那样一来，只想跑入库把关自检（这条路径压根用不到大模型）的人也会被卡住 ——
            面试官 clone 下来第一步就失败，CI 也跑不了。
        ★ 自检 / 测试这条路径全程不碰 llm，所以没 key 也能跑完。
        """
        if self._llm is None:
            self._llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        return self._llm

    def startup(self):
        t0 = time.time()
        self.model = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=CACHE_DIR)
        # ★ 这里【故意不建 LLM】，原因见上面的 llm 属性

        kept, rejected = self._guard(RAW_DOCS)
        self.chunks = kept
        self.rejected = rejected
        self.vectors = np.array(
            list(self.model.passage_embed([c["text"] for c in kept])), dtype="float32")
        # BM25 索引（字面检索）：jieba 分词后建索引，和向量检索互补
        #   向量看"意思像不像"，BM25 看"字面有没有出现" —— 两者都会单用时翻车，见 search() 注释
        self.bm25 = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in kept])
        self.ready = True
        self.boot_time = time.time() - t0

    def _guard(self, docs):
        """五道关卡 + 版本冲突（逻辑和 Step 5 一样）"""
        kept, rejected = [], []
        for c in docs:
            if len(c["text"].strip()) < MIN_LEN:
                rejected.append((c, "太短碎屑"))
            else:
                kept.append(c)

        seen, stage2 = set(), []
        for c in kept:
            h = hashlib.md5(norm(c["text"]).encode()).hexdigest()
            if h in seen:
                rejected.append((c, "完全重复"))
            else:
                seen.add(h)
                stage2.append(c)
        kept = stage2

        vecs = np.array(list(self.model.passage_embed([c["text"] for c in kept])), dtype="float32")
        stage3 = []
        for i, c in enumerate(kept):
            # ★ stage3 里存的是【下标】，所以要比对的直接就是 vecs[j]。
            #   不能写成 range(len(stage3)) —— 那样 j 变成"第几个通过的"，
            #   一旦前面拦掉过东西，两套编号就错位了：
            #   新块会去比【已被拦掉的】，却漏掉【真正该比的已通过块】，于是漏拦。
            #   tests/test_guard.py::test_near_dup_after_earlier_rejection 守住这里。
            dup = any(float(vecs[i] @ vecs[j]) > DUP_THRESHOLD for j in stage3)
            if dup:
                rejected.append((c, "近似重复"))
            else:
                stage3.append(i)
        kept = [kept[j] for j in stage3]

        stage4 = []
        for c in kept:
            hit = next((name for pat, name in PII_PATTERNS if re.search(pat, c["text"])), None)
            if hit:
                rejected.append((c, f"含个人隐私（{hit}）"))
            else:
                stage4.append(c)
        kept = stage4

        # 版本冲突：同一 (doc, clause) 只留 version 最大的
        # ★ 这里的 key 必须是 (文档, 条款) 两样一起，不能只用文档名！
        #   只用文档名的话，"退款政策.md"里【已发货】和【未发货】是两条完全不同的规定，
        #   会被误判成"同一条的新旧两版"，结果一条把另一条挤掉 —— 库里凭空少一条规则。
        newest = {}
        for c in kept:
            key = (c["doc"], c["clause"])          # ← 两样一起当身份证
            if key not in newest:
                newest[key] = c
                continue
            old = newest[key]
            if c["version"] > old["version"]:       # 新来的更新 → 换掉旧的
                rejected.append((old, f"旧版本，被新版取代（{old['version']} → {c['version']}）"))
                newest[key] = c
            else:                                   # 旧版本 / 同版本后来者 → 新来的被拦
                rejected.append((c, f"旧版本，库里已有更新的 {old['version']}"))
        kept = list(newest.values())
        return kept, rejected

    def search(self, query, k=5):
        """混合检索：向量（看意思）+ BM25（看字面），用 RRF 融合两路的排名

        为什么不只用向量？（都是本项目实测过的翻车现场）
            · "东西坏了能修吗"（文档里写的是"保修"）→ 向量能找到，BM25 找不到
            · "A1001 运单号"（三条只差编号）→ BM25 稳，向量第一二名只差 0.008（拿不准）
        两路互补，谁也别丢。

        ★ RRF（倒数排名融合）：每路的贡献是 1/(60+名次)。
          用【名次】而不是【原始分数】，是因为向量分在 0~1 之间、BM25 分能到十几，
          直接相加的话 BM25 会把向量完全淹没 —— 不同量纲的分数不能相加，这是常见踩坑。
        """
        q = np.array(list(self.model.query_embed([query])), dtype="float32")[0]
        vec_scores = self.vectors @ q                       # 每块与问题的向量相似度
        vec_order = np.argsort(-vec_scores)                 # 按分数从高到低的下标

        tokens = list(jieba.cut_for_search(query))          # 中文按"检索粒度"分词
        bm25_order = np.argsort(-self.bm25.get_scores(tokens))

        rrf = {}
        for rank, idx in enumerate(vec_order):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_order):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

        top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        # 返回向量分只是为了展示（网页上那个"向量分"标签），真正决定顺序的是 RRF
        return [(self.chunks[idx], float(vec_scores[idx])) for idx, _ in top]

    def rerank(self, query, candidates):
        """让大模型给每条候选打分，返回 [(块, 向量分, 精排分, 理由)]，按精排分从高到低

        为什么两个分数都要留着？
            向量分 = 粗捞阶段的（电脑按字面/语义相似度算的，快但粗）
            精排分 = 大模型重新看的（慢但准）
        两个都返回，你就能在网页上直观看到"粗捞排第一的，精排未必第一" —— 这正是 rerank 的意义。
        """
        numbered = "\n".join(f"{i}. {c['text']}" for i, (c, _) in enumerate(candidates, 1))
        prompt = (
            "判断下列每条资料对回答这个问题有多大帮助。\n"
            f"问题：{query}\n\n候选资料：\n{numbered}\n\n"
            "只输出 JSON：{\"scores\":[{\"id\":1,\"score\":0到10的整数,\"reason\":\"一句话\"}]}"
        )
        resp = self.llm.invoke([HumanMessage(content=prompt)]).content
        raw = re.sub(r"^```(?:json)?|```$", "", resp.strip(), flags=re.M).strip()
        try:
            data = json.loads(raw)
            sc = {int(x["id"]): (int(x["score"]), x.get("reason", "")) for x in data["scores"]}
        except Exception:
            sc = {i: (0, "") for i in range(1, len(candidates) + 1)}
        # 大模型只回 id 和分数，我们按 id 把【原始候选】捞回来（candidates 下标从 0 开始，id 从 1 开始）
        out = []
        for i, (score, reason) in sorted(sc.items(), key=lambda kv: kv[1][0], reverse=True):
            chunk, vec_score = candidates[i - 1]
            out.append((chunk, float(vec_score), score, reason))
        return out

    def answer(self, question, chunks):
        ctx = "\n".join(f"- {c['text']}" for c in chunks)
        prompt = ("你是客服助手。只根据下面的资料回答，资料里没有的就明确说不知道。\n"
                  f"资料：\n{ctx}\n\n问题：{question}")
        return self.llm.invoke([HumanMessage(content=prompt)]).content


rag = RAG()


# ==================================================================
# ③ FastAPI：把函数变成 HTTP 接口
# ==================================================================
# lifespan：服务启动时干一次（加载模型），关闭时收尾。
# @ 开头的叫【装饰器】，它把下面那个函数"包装"一下交给框架 —— 你不用手动调用它。
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("服务启动中：加载模型 + 建库（只在启动时做一次）……")
    rag.startup()
    print(f"✓ 就绪：{len(rag.chunks)} 块入库，拦下 {len(rag.rejected)} 块，"
          f"耗时 {rag.boot_time:.1f} 秒")
    yield                       # ← 这一行表示"服务开始接客"
    print("服务关闭")


app = FastAPI(title="客服知识库问答", lifespan=lifespan)


# pydantic 模型：规定"请求体必须长这样"，传错了框架自动返回 422，不用你写判断
class ChatRequest(BaseModel):
    question: str = Field(..., description="用户的问题", min_length=1)
    top_k: int = Field(5, description="粗捞几条", ge=1, le=10)
    use_rerank: bool = Field(True, description="是否启用 rerank 精排")


class Source(BaseModel):
    doc: str
    text: str
    score: float
    rerank_score: Optional[int] = None
    reason: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    took_ms: int
    rerank_used: bool
    # 是否命中了知识库。False = 库里没有，已拒答，一条资料都没喂给模型。
    # ★ 这个字段是"可追溯"的一部分：调用方能明确区分「答错了」和「拒绝作答」。
    knowledge_hit: bool = True


@app.get("/health")
def health():
    """健康检查：部署后第一件事就是访问它，看服务活着没"""
    return {"status": "ok" if rag.ready else "loading",
            "chunks": len(rag.chunks),
            "rejected": len(rag.rejected)}


@app.get("/guard-report")
def guard_report():
    """入库把关报告：哪些被拦了、为什么 —— 这是你敢把仓库给人看的底气"""
    return {"kept": [{"doc": c["doc"], "clause": c["clause"], "text": c["text"]}
                     for c in rag.chunks],
            "rejected": [{"doc": c["doc"], "clause": c.get("clause", ""),
                          "text": c["text"][:40], "reason": r}
                         for c, r in rag.rejected]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """核心接口：问题进来 → 粗捞 → rerank → 生成 → 带出处返回"""
    t0 = time.time()
    if not rag.ready:
        return {"answer": "服务还在加载模型，请稍等几秒再试", "sources": [],
                "took_ms": 0, "rerank_used": False}

    cands = rag.search(req.question, k=req.top_k)

    took = lambda: int((time.time() - t0) * 1000)          # 匿名函数，算"到现在用了多少毫秒"

    if req.use_rerank:
        ranked = rag.rerank(req.question, cands)          # [(块, 向量分, 精排分, 理由)]

        # ★★ 拒答硬短路：连最高分的候选都不到 5 分 = 库里根本没这个东西。
        #    这时【直接返回，压根不调用生成模型】。
        #    之前的写法是"全部低分也硬塞第一条进去"，寄希望于提示词让模型拒答 ——
        #    但提示词会漏，喂了假资料模型就会照着编。真正可靠的做法是【不喂】。
        if not ranked or ranked[0][2] < 5:
            return ChatResponse(
                answer="知识库里没有能回答这个问题的资料，我不编。",
                sources=[], took_ms=took(), rerank_used=True, knowledge_hit=False)

        picked = [x for x in ranked if x[2] >= 5][:3]     # 只留 5 分以上的，最多 3 条
        sources = [Source(doc=f"{c['doc']}｜{c['clause']}", text=c["text"],
                          score=round(v, 3), rerank_score=s, reason=r)
                   for c, v, s, r in picked]
        ctx = [c for c, _, _, _ in picked]
    else:
        # ★★ 关掉 rerank 时【照样要拒答】。
        #    之前的写法是"没有分数可判断，只能照常生成" —— 那等于网页上取消勾选一下，
        #    整个拒答机制就被绕过去了，而且 knowledge_hit 还是 true（调用方以为命中了）。
        #    精排分不存在，就用向量分（0~1 的余弦相似度），阈值见 VEC_REJECT_THRESHOLD 的注释。
        if not cands or cands[0][1] < VEC_REJECT_THRESHOLD:
            return ChatResponse(
                answer="知识库里没有能回答这个问题的资料，我不编。",
                sources=[], took_ms=took(), rerank_used=False, knowledge_hit=False)

        picked = [x for x in cands if x[1] >= VEC_REJECT_THRESHOLD][:3]
        sources = [Source(doc=f"{c['doc']}｜{c['clause']}", text=c["text"],
                          score=round(s, 3)) for c, s in picked]
        ctx = [c for c, _ in picked]
        # 关掉 rerank 时用的是粗捞的向量分，比精排分粗（"意思接近但答非所问"更容易漏进来），
        # 所以默认还是开着 rerank —— 关掉只是留一个对比开关，不是关闭防线。

    answer = rag.answer(req.question, ctx)
    return ChatResponse(answer=answer, sources=sources,
                        took_ms=took(), rerank_used=req.use_rerank)


# ==================================================================
# ④ 一个能直接聊天的网页（GET / ）
#    意义：简历上的链接点开是这个页面，不是一串 JSON —— 面试官体验完全不同
# ==================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>客服知识库问答</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.7}
 h2{font-weight:500} textarea{width:100%;height:60px;font-size:14px;padding:8px}
 button{padding:8px 20px;font-size:14px;cursor:pointer;margin-top:8px}
 .box{background:#f6f6f4;border-radius:8px;padding:12px 16px;margin-top:14px;white-space:pre-wrap}
 .src{font-size:13px;color:#666;border-left:3px solid #ddd;padding-left:10px;margin-top:6px}
 .tag{display:inline-block;font-size:12px;background:#e8f0fe;color:#185FA5;border-radius:4px;padding:1px 6px;margin-right:6px}
</style></head><body>
<h2>客服知识库问答（RAG）</h2>
<p style="color:#666;font-size:14px">bge-small-zh 向量检索 + BM25 混合 + 入库把关 + 大模型精排 + DeepSeek 生成</p>
<textarea id="q" placeholder="试试：已发货的订单退款要扣多少钱？ / 保修多久？ / 支持分期付款吗？"></textarea><br>
<button onclick="ask()">提问</button>
<label style="margin-left:12px;font-size:13px"><input type="checkbox" id="rr" checked> 启用 rerank</label>
<div class="box" id="ans" style="display:none"></div>
<div id="src"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q){alert('请输入问题');return;}
  const rr=document.getElementById('rr').checked;
  const a=document.getElementById('ans'),s=document.getElementById('src');
  a.style.display='block';a.textContent='思考中…';s.innerHTML='';
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q,use_rerank:rr})});
  const d=await r.json();
  a.textContent=d.answer;
  if(d.knowledge_hit===false){
    s.innerHTML='<p style="font-size:13px;color:#b45f04;margin-top:12px">未命中知识库 · 已拒绝作答（一条资料都没喂给模型）</p>';
    return;
  }
  s.innerHTML='<p style="font-size:13px;color:#888;margin-top:12px">用时 '+d.took_ms+' ms　引用资料：</p>'+
    (d.sources||[]).map(x=>'<div class="src"><span class="tag">'+x.doc+'</span>'+
      (x.rerank_score!=null?'<span class="tag">精排 '+x.rerank_score+' 分</span>':'')+
      x.text+(x.reason?'<br><i style="color:#999">'+x.reason+'</i>':'')+'</div>').join('');
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


# ==================================================================
# ⑤ 启动（只有 python xxx.py 直接运行时才会执行；被 import 时不会）
# ==================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("服务启动后访问：")
    print("   http://127.0.0.1:8000        聊天页面")
    print("   http://127.0.0.1:8000/docs   自动生成的接口文档")
    print("   http://127.0.0.1:8000/health 健康检查")
    print("按 Ctrl+C 停止")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
