# -*- coding: utf-8 -*-
"""
入库把关 + 拒答短路的【回归测试】

和旧版（打印型自检）的区别，一句话：
    旧版 print("通过"/"失败")  →  退出码永远是 0，改坏了没人知道
    新版 assert                →  失败时 pytest 报 FAILED 并让退出码变 1，CI 能看到红

跑法（在仓库根目录）：
    D:/Python-project/.venv/Scripts/python.exe -m pytest tests/ -v

想顺便看"留了谁 / 拦了谁"：
    D:/Python-project/.venv/Scripts/python.exe -m pytest tests/ -v -s
    （-s = 不吞掉 print，报告会打出来）

★ 不需要 DEEPSEEK_API_KEY：
    这些用例只跑"入库把关 / 检索 / 拒答"这三条路径，一条都不调生成模型。
    service.py 里的 llm 是懒加载的（用到才建），所以没配 key 也能全绿。

★ 变异测试（验证这套断言真的有用）：
    把 service.py 里的 DUP_THRESHOLD 从 0.90 改成 0.99，再跑一次 ——
    应该看到 FAILED。这就是"改坏了会立刻报错"真正的样子。
"""

import os
import sys

# 这两行必须放在 import 项目模块之前：
# os.environ 是"这个 Python 进程自己的环境变量字典"，改它只影响当前进程。
os.environ["LANGSMITH_TRACING"] = "false"          # 自检不调大模型，关掉 trace 免得联网卡住
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# sys.path 是"Python 去哪些目录找模块"的列表。
# 本文件在 tests/ 子目录里，而 service.py 在上一层，所以需要把上一层加进去。
# __file__ = 当前文件路径；dirname 取所在目录；再 dirname 一次 = 上一层。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest                                      # noqa: E402
import service as svc                              # noqa: E402 导入不会启动服务（有 __main__ 保护）


# ---- 测试用的"假大模型" ----
# 为什么要假货？因为真模型要联网、要 API key、每次结果还可能不一样。
# 测试关心的是"流程对不对"（该拒答时拒答、该命中时命中），不是"模型说了什么"，
# 所以把模型换成一个人偶，让流程可以脱离网络被反复验证。
class _FakeResp:
    """假返回：只需要有 .content 这个属性，和真模型返回的对象长得一样"""

    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """假模型：不管问什么都返回指定字符串"""

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return _FakeResp(self.reply)


@pytest.fixture(scope="module")
def rag():
    """准备材料：加载模型 + 把关 + 建库，整个文件只做一次。

    fixture = pytest 的"准备材料"机制。scope="module" 表示本文件共用这一份，
    因为加载 embedding 模型要十几秒，不能每条用例都重新加载一遍。
    """
    svc.rag.startup()
    print(f"\n入库 {len(svc.rag.chunks)} 条，拦下 {len(svc.rag.rejected)} 条，"
          f"耗时 {svc.rag.boot_time:.1f} 秒")
    return svc.rag


@pytest.fixture
def clauses(rag):
    """留下来的条款集合，元素是 (文档名, 条款名) 这样的二元组"""
    return {(c["doc"], c["clause"]) for c in rag.chunks}


# ==================================================================
# ① 总数：少一条、多一条都要报警
#    ★ 没有这一条的话，"误伤型"改动（把关太严、把该留的也杀了）会全绿通过
# ==================================================================
def test_kept_count_is_7(rag):
    assert len(rag.chunks) == 7, f"放行了 {len(rag.chunks)} 条，应该是 7 条"


def test_rejected_count_is_3(rag):
    assert len(rag.rejected) == 3, f"拦下了 {len(rag.rejected)} 条，应该是 3 条"


# ==================================================================
# ② 该留的必须在：直接盯住那个修过的 bug
# ==================================================================
def test_kept_has_shipped_refund(clauses):
    """已发货退款条款还在"""
    assert ("退款政策.md", "退款-已发货") in clauses


def test_kept_has_unshipped_refund(clauses):
    """未发货退款条款还在 ★ 这是以前踩过的坑：
    版本冲突关一开始只按【文档名】分组，结果同一份 退款政策.md 里的
    【已发货】和【未发货】被当成"同一条的新旧版"互相挤掉，库里凭空少一条规则。
    改成按 (文档 + 条款) 分组才修好 —— 这条断言就是守住这个修复。"""
    assert ("退款政策.md", "退款-未发货") in clauses


def test_kept_has_delivery_time(clauses):
    """发货时效条款还在"""
    assert ("发货时效.md", "现货发货") in clauses


# ==================================================================
# ③ 该拦的必须被拦：三道关卡各盯一条
# ==================================================================
def test_rejected_old_version(rag):
    """旧版本条款被拦（版本冲突关）"""
    assert any("旧版本" in reason for _, reason in rag.rejected)


def test_rejected_privacy(rag):
    """含手机号 / 身份证的那条被拦（个人隐私关）"""
    assert any("隐私" in reason for _, reason in rag.rejected)


def test_rejected_duplicate(rag):
    """话术库的同义改写版被拦（近似重复关）"""
    assert any("重复" in reason for _, reason in rag.rejected)


def test_near_dup_after_earlier_rejection(rag):
    """★ 回归：前面有块被拦掉时，后面的近似重复【仍然要被拦住】

    这条专门盯 stage3 的下标写法。
    如果写成 `for j in range(len(stage3))`，j 遍历的是【位置】而不是【已通过块的下标】：
    一旦前面拦掉过东西，后面的比对对象就全错位 ——
    拿新块去比【已被拦掉的】，却漏掉【真正该比的已通过块】。

    为什么上面那条 test_rejected_duplicate 抓不到？
    它只断言"有东西因重复被拦了"。而本项目那条约会话术库恰好排在最后一个被检查，
    前面没有块被拦过，错位根本没被触发。所以必须自己造数据，把错位造出来。
    """
    docs = [
        dict(doc="t.md", clause="A", version="2026-01-01", source="测试",
             text="现货商品在付款后 48 小时内发出，预售商品的发货时间以商品页面标注为准。"),
        dict(doc="t.md", clause="A2", version="2026-01-01", source="测试",
             text="现货商品付款后 48 小时内发货，预售商品以商品页面标注的发货时间为准。"),
        dict(doc="t.md", clause="C", version="2026-01-01", source="测试",
             text="普通会员累计消费满 1000 元自动升级为 VIP 会员，等级次日生效。"),
        dict(doc="t.md", clause="C2", version="2026-01-01", source="测试",
             text="普通会员消费累计满 1000 元即可自动升级为 VIP 会员，等级在次日生效。"),
    ]
    kept, rejected = rag._guard(docs)
    kept_clauses = {c["clause"] for c in kept}
    reasons = {c["clause"]: r for c, r in rejected}

    # 前提检查：得先确认"错位"真的被造出来了 —— A2 被近似重复挡住
    assert "A2" not in kept_clauses, (
        f"测试前提不成立：A2 没被判为与 A 近似重复。实际保留 {sorted(kept_clauses)}，"
        f"原因 {reasons}。（多半是这两句改写得不够像、相似度掉到 0.90 以下了 —— "
        "改测试文本，不要改断言）")
    assert "近似重复" in reasons.get("A2", ""), (
        f"A2 是被别的关卡拦的，不是近似重复关：{reasons.get('A2')}")

    # ★ 被盯住的那件事：前面拦掉 A2 之后，C2 依然要被拦住
    assert "C2" not in kept_clauses, (
        "近似重复检测漏拦了！A2 被拦后 stage3 下标错位，"
        f"C2 没能和 C 比对。实际保留 {sorted(kept_clauses)}")


# ==================================================================
# ④ 拒答短路：这是整个项目最该被守住的承诺
# ==================================================================
def test_refuse_when_rerank_scores_low(rag, monkeypatch):
    """rerank 分支：大模型给所有候选都打了低分 → 拒答，一条资料都不喂

    monkeypatch 是 pytest 的"临时替换"工具，测试跑完自动还原，不会污染其他用例。
    这里用它把模型换成人偶：让它给候选打 1 分（满分 10），模拟"全都不相关"。
    """
    monkeypatch.setattr(rag, "_llm", _FakeLLM(
        '{"scores":[{"id":1,"score":1,"reason":"完全不相关"}]}'))

    resp = svc.chat(svc.ChatRequest(question="支持分期付款吗？", use_rerank=True))
    assert resp.knowledge_hit is False, "精排分低于 5 却不拒答"
    assert resp.sources == [], "拒答时不该返回任何资料"
    assert "我不编" in resp.answer


def test_refuse_even_rerank_off(rag):
    """★ 关掉 rerank 也【不许绕过拒答】。

    以前的写法是"关掉 rerank 没分数可判断，只能照常生成" ——
    那等于网页上取消勾选一下，整个拒答机制就没了，而且 knowledge_hit 还是 true。
    这条断言守住：不管开关怎么拨，拒答都得在。"""
    resp = svc.chat(svc.ChatRequest(question="支持分期付款吗？", use_rerank=False))
    assert resp.knowledge_hit is False, "库里没有却没拒答，防线被绕过了"
    assert resp.sources == [], "拒答时不该返回任何资料"
    assert "我不编" in resp.answer


def test_hit_returns_sources(rag, monkeypatch):
    """库里有的 → 正常命中并返回出处（同样用人偶模型，不需要 API key）"""
    monkeypatch.setattr(rag, "_llm", _FakeLLM("（假模型返回，未真实调用）"))

    resp = svc.chat(svc.ChatRequest(question="已发货的订单退款要扣多少钱？", use_rerank=False))
    assert resp.knowledge_hit is True, "库里明明有，却被当成没命中"
    assert len(resp.sources) > 0, "命中了却没给出处"


# ==================================================================
# ⑤ 阈值本身也要有守护
#    VEC_REJECT_THRESHOLD = 0.55 是在本项目语料上实测出来的：
#        该答的（7 题）向量分 0.6541 ~ 0.8425
#        该拒的（5 题）向量分 0.2619 ~ 0.4373
#    换语料之后这条会红 —— 它不是 bug，是提醒你"阈值该重新量了"。
# ==================================================================
SHOULD_HIT = [
    "已发货的订单退款要扣多少钱？",
    "东西坏了能修吗？",
    "会员怎么升级成 VIP？",
    "跨境订单要交税吗？",
    "物流单号多久能查到？",
    "现货什么时候发货？",
    "保修期从什么时候开始算？",
]

SHOULD_REFUSE = [
    "支持分期付款吗？",
    "你们公司 CEO 是谁？",
    "今天天气怎么样？",
    "优惠券怎么用？",
    "可以用花呗吗？",
]


def test_vec_threshold_separates_hit_and_miss(rag):
    """阈值必须真的能把"该答的"和"该拒的"分开"""
    for q in SHOULD_HIT:
        top = rag.search(q, k=5)[0][1]
        assert top >= svc.VEC_REJECT_THRESHOLD, (
            f"「{q}」库里有，却因向量分 {top:.4f} 低于阈值 {svc.VEC_REJECT_THRESHOLD} 被误拒")
    for q in SHOULD_REFUSE:
        top = rag.search(q, k=5)[0][1]
        assert top < svc.VEC_REJECT_THRESHOLD, (
            f"「{q}」库里没有，却因向量分 {top:.4f} 高于阈值 {svc.VEC_REJECT_THRESHOLD} 被放行")
