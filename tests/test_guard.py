# -*- coding: utf-8 -*-
"""
入库把关 + 拒答短路的【回归测试】

和旧版（打印型自检）的区别，一句话：
    旧版 print("通过"/"失败")  →  退出码永远是 0，改坏了没人知道
    新版 assert                →  失败时 pytest 报 FAILED 并让退出码变 1，CI 能看到红

跑法（在仓库根目录）：
    python -m pytest tests/ -v

想顺便看"留了谁 / 拦了谁"：
    python -m pytest tests/ -v -s
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
def test_kept_count_matches_real_corpus(rag):
    """真实语料（学校公开制度）入库块数。

    ★ 这个数字会随 docs/ 增删而变 —— 这不是缺点，恰恰是
      "往 docs/ 扔一个 .md 库就变大"这条能力的可执行证据。
      换语料后要同步改这里，但【只改数字不够】：下面还有盯具体条款的断言，
      否则"库里少了一半"这种事只要跟着改数字就能骗过去。
    """
    assert len(rag.chunks) == 67, f"放行了 {len(rag.chunks)} 条，应该是 67 条"


def test_rejected_count_is_16(rag):
    """被拦下的块数：当前全部来自 docs/inbox/ 的网页抓取残留（太短碎屑关）"""
    assert len(rag.rejected) == 16, f"拦下了 {len(rag.rejected)} 条，应该是 16 条"


# ==================================================================
# ② 该留的必须在：直接盯住那个修过的 bug
# ==================================================================
def test_kept_has_punishment_kinds(clauses):
    """处分种类那条还在（真实语料里最常被问的一条：警告/严重警告/记过/留校察看/开除学籍）"""
    assert ("学生纪律处分管理规定.md", "第二章-第四条") in clauses


def test_same_doc_different_clauses_all_kept(clauses):
    """★ 以前踩过的坑，换语料后这条语义仍要守住：
    版本冲突关一开始只按【文档名】分组，结果同一份文档里的不同条款
    被当成"同一条的新旧版"互相挤掉，库里凭空少规则。
    改成按 (文档 + 条款) 分组才修好。
    真实语料里对应的场景是同一份《纪律处分管理规定》开头连续三条，
    断言它们必须各自独立保留 —— 少一条就说明分组又退化了。"""
    for c in ("第一章-第一条", "第一章-第二条", "第一章-第三条"):
        assert ("学生纪律处分管理规定.md", c) in clauses, f"{c} 被误挤掉了"


def test_kept_has_leave_approval(clauses):
    """请销假制度的请假审批权限那条还在（1天班主任 / 1周二级学院 / 1周以上三级）"""
    assert ("学生请销假制度.md", "总则-第三条") in clauses


# ==================================================================
# ③ 该拦的必须被拦：三道关卡各盯一条
# ==================================================================
def test_rejected_old_version(rag):
    """版本冲突关：同 (doc, clause) 出现两个 version 时，旧的那个要被拦下。

    ★ 为什么改成自己造数据：真实语料是学校公开制度，只有单一版本，
      没有"新旧版冲突"的样本，所以这一关在真实入库里【根本不会触发】。
      想确认它还在工作，只能构造样本直接喂给 _guard。
      （同理见下面两条 —— 这是换真实语料后必须诚实说明的事，别假装关卡被验证过。）
    """
    docs = [
        dict(doc="t.md", clause="同一条", version="2026-01-01", source="测试",
             text="学生请假一天以内由班主任审批，一周以内由二级学院审批，一周以上须三级审批。"),
        dict(doc="t.md", clause="同一条", version="2024-01-01", source="测试（旧版）",
             text="宿舍内禁止使用大功率电器，一经发现没收器具并给予通报批评处理。"),
    ]
    kept, rejected = rag._guard(docs)
    assert len(kept) == 1, f"同条款两个版本应只留一个，实际留了 {len(kept)}"
    assert any("旧版本" in r for _, r in rejected), f"旧版本没被拦：{rejected}"


def test_rejected_privacy(rag):
    """个人隐私关：含手机号 / 身份证的文本要被拦下，不能进库。
    （真实公开制度里没有 PII，所以同样靠构造样本验证这一关还活着。）"""
    docs = [dict(doc="t.md", clause="含隐私", version="2026-01-01", source="测试",
                 text="联系方式：13812345678，身份证号 440301199001011234，请核实后办理。")]
    kept, rejected = rag._guard(docs)
    assert kept == [], f"含隐私的文本不该入库：{kept}"
    assert any("隐私" in r for _, r in rejected), f"隐私关没拦：{rejected}"


def test_rejected_duplicate(rag):
    """近似重复关：措辞微调、语义几乎一样的改写版要被拦下。
    两句只差几个字，确保余弦相似度稳过 0.90 —— 换真实语料后同样无现成样本。"""
    docs = [
        dict(doc="t.md", clause="D1", version="2026-01-01", source="测试",
             text="学生在校学习期间离校应当由本人办理请假手续，并附有关证明材料。"),
        dict(doc="t.md", clause="D2", version="2026-01-01", source="测试",
             text="学生在校学习期间离校应当由本人办理请假手续，并附上有关证明材料。"),
    ]
    kept, rejected = rag._guard(docs)
    assert len(kept) == 1, f"近似重复的两块应只留一条，实际留了 {len(kept)}"
    assert any("重复" in r for _, r in rejected), f"近似重复没被拦：{rejected}"


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


def test_short_text_rejected(rag):
    """★ 补盲区：太短碎屑关（MIN_LEN）在当前真实语料里从不触发
    （所有块都 > 15 字），所以这个关要是被删了，最终 7/3 数字不变、
    上面所有断言全绿 —— 等于没测。这里自己造一条短文本，确保它真的在拦。"""
    docs = [dict(doc="短.md", clause="S", version="2026-01-01", source="测试",
                 text="你好")]
    kept, rejected = rag._guard(docs)
    assert kept == [], f"太短的文本不该被放行：{kept}"
    assert any("太短" in r for _, r in rejected), (
        f"太短关没拦住「你好」，原因：{[r for _, r in rejected]}")


def test_exact_duplicate_rejected(rag):
    """★ 补盲区：完全重复（md5 相同）关在当前真实语料里也不触发
    （没有任何两块一字不差）。删了它最终数字同样不变，断言全绿。
    这里造两条完全相同的文本，确认后一条被『完全重复』拦下、前一条留下。"""
    same = "未发货订单可全额退款，不扣任何费用，审核通过后 24 小时内到账。"
    docs = [
        dict(doc="dup.md", clause="A", version="2026-01-01", source="测试", text=same),
        dict(doc="dup.md", clause="A", version="2026-01-01", source="测试", text=same),
    ]
    kept, rejected = rag._guard(docs)
    assert len(kept) == 1, f"完全相同的两块应只留一条，实际留了 {len(kept)}"
    assert any("完全重复" in r for _, r in rejected), (
        f"完全重复关没拦住，原因：{[r for _, r in rejected]}（被近似重复关兜底说明完全重复关失效）")


def test_missing_source_rejected(rag):
    """★ 补盲区：关卡5「缺少来源/日期」在真实语料里同样从不触发
    （10 条都带 source + version），删掉它最终 7/3 数字不变、上面断言全绿。

    这条同时守着一个"文档与实现不一致"的历史缺陷：service.py 的 docstring
    曾写着"五道关卡"，实现却只有 4 关 + 版本冲突 —— 读者以为有兜底，其实没有。
    （当年没移植是误以为第 5 关是"来源白名单"，去读 step5 才发现只是判空。）

    ⚠️ 别误解这一关的强度：它只保证字段【非空】，不保证来源【可信】。
    自己填个 source=官网帮助中心 照样能过 —— 所以 docs/inbox/ 仍需人工审核。
    """
    docs = [dict(doc="无来源.md", clause="N", version="", source="",
                 text="现货商品在付款后 48 小时内发出，预售商品以页面标注的发货时间为准。")]
    kept, rejected = rag._guard(docs)
    assert kept == [], f"缺来源/日期的文本不该被放行：{kept}"
    assert any("来源" in r for _, r in rejected), (
        f"关卡5 没拦住缺来源的条目，原因：{[r for _, r in rejected]}")


def test_guard_survives_missing_keys(rag):
    """★ 关卡5 用 .get() 而不是 c["source"] 的原因：缺键时要『拦下并说明』，
    而不是抛 KeyError 把服务带崩 —— 把关的失败方式也该是"拦"，不是"炸"。"""
    docs = [dict(doc="残.md", clause="B",
                 text="跨境订单需缴纳进口税，税费在清关时由承运商代收，以海关核定为准。")]
    kept, rejected = rag._guard(docs)          # 缺 version / source 两个键，不该抛异常
    assert kept == [], f"缺键的条目不该被放行：{kept}"
    assert any("来源" in r for _, r in rejected), (
        f"缺键时应被关卡5 拦下，原因：{[r for _, r in rejected]}")


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

    resp = svc.chat(svc.ChatRequest(question="图书馆几点关门？", use_rerank=True))
    assert resp.knowledge_hit is False, "精排分低于 5 却不拒答"
    assert resp.sources == [], "拒答时不该返回任何资料"
    assert "我不编" in resp.answer


def test_refuse_even_rerank_off(rag):
    """★ 关掉 rerank 也【不许绕过拒答】。

    以前的写法是"关掉 rerank 没分数可判断，只能照常生成" ——
    那等于网页上取消勾选一下，整个拒答机制就没了，而且 knowledge_hit 还是 true。
    这条断言守住：不管开关怎么拨，拒答都得在。"""
    resp = svc.chat(svc.ChatRequest(question="图书馆几点关门？", use_rerank=False))
    assert resp.knowledge_hit is False, "库里没有却没拒答，防线被绕过了"
    assert resp.sources == [], "拒答时不该返回任何资料"
    assert "我不编" in resp.answer


def test_hit_returns_sources(rag, monkeypatch):
    """库里有的 → 正常命中并返回出处（同样用人偶模型，不需要 API key）"""
    monkeypatch.setattr(rag, "_llm", _FakeLLM("（假模型返回，未真实调用）"))

    resp = svc.chat(svc.ChatRequest(question="什么情况会被开除学籍？", use_rerank=False))
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
    "什么情况会被开除学籍？",
    "处分有哪几种？",
    "从轻处分的情形有哪些？",
    "从重处分的情形有哪些？",
    "请假一天谁批准？",
    "病假要交什么证明？",
    "假满不销假会怎样？",
    "用欺骗手段请假怎么处罚？",
]

SHOULD_REFUSE = [
    "图书馆几点关门？",
    "学费一年多少钱？",
    "今天天气怎么样？",
    "食堂几点开饭？",
    "怎么申请助学贷款？",
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


def test_guard_report_masks_pii(rag):
    """★ 补 PII 盲区：/guard-report 返回的 rejected 不得明文含手机号/身份证。
    之前 `_mask_pii` 定义了却从未调用，rejected 只做了 `[:40]` 截断，
    而那条隐私数据仅 30 字，截断无效 → 手机号/身份证原样泄露。这条守住它。"""
    rep = svc.guard_report()
    for item in rep["rejected"]:
        assert "13812345678" not in item["text"], f"手机号明文泄露：{item['text']}"
        assert "440301199001011234" not in item["text"], f"身份证明文泄露：{item['text']}"


# ==================================================================
# ⑥ 网页渲染不得用 innerHTML（XSS 回归）
# ==================================================================
def test_web_page_has_no_innerhtml():
    """★ 补 XSS 盲区：知识库改成从文件读之后，.md 文本会经 sources 进网页。
    若用 innerHTML 拼接，文档里混进 `<img onerror=...>` 就会被当标签执行。
    必须走 textContent / createTextNode（纯文本，不解析标签）。

    注意：只禁【属性访问】`.innerHTML` —— 代码注释里提到 "innerHTML" 不算违规。
    """
    assert ".innerHTML" not in svc.HTML_PAGE, "HTML_PAGE 又用回 innerHTML 了（XSS 风险）"
    assert "textContent" in svc.HTML_PAGE, "渲染应走 textContent 纯文本路径"
