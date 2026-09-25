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
    把 service.py 里的 DUP_THRESHOLD 从 0.96 改成 0.99，再跑一次 ——
    应该看到 FAILED。这就是"改坏了会立刻报错"真正的样子。

★ 2026-09-25 实测过的 4 个变异，全部被抓住（没一个是"测试写了但抓不住"）：
    DUP_THRESHOLD 0.96→0.99  → test_near_dup_after_earlier_rejection 红
    DUP_THRESHOLD 0.96→0.90  → test_dup_threshold_has_margin 红（误伤第二十五条复现）
    隐私关 hit 恒为 None      → test_rejected_privacy + 脏数据夹具 红
    版本冲突永不换新          → 脏数据夹具 红
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
    assert len(rag.chunks) == 68, f"放行了 {len(rag.chunks)} 条，应该是 68 条"


def test_rejected_count_is_15(rag):
    """被拦下的块数：当前全部来自 docs/inbox/ 的网页抓取残留（太短碎屑关 + 近似重复关）"""
    assert len(rag.rejected) == 15, f"拦下了 {len(rag.rejected)} 条，应该是 15 条"


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


def test_version_compare_survives_unpadded_date(rag):
    """★ 版本新旧不许靠"ISO 补零"这个巧合（2026-09-25 实测）。

    旧代码是 `c["version"] > old["version"]` 直接比字符串，而
    `'2026-9-5' > '2026-09-25'` 竟然是 True —— 字符串逐位比字符编码，
    比到第 6 位 `'9' > '0'` 就下结论了，它不知道"9 月 < 10 月"。
    于是【一个没补零的旧版会把库里真正的新版挤掉】，
    而拒绝理由还写成「被新版取代（2026-09-25 → 2026-9-5）」—— 新旧完全说反。

    ★ 顺序刻意让 2026-09-25 先进（模拟"库里已经有一版"），未补零的后到。
    ★ 两条正文用的是 test_rejected_old_version 里那两条已知能走通前面几关的文本，
      所以这条测的确实是版本比较，不是被别的关卡拦在了前面。
    """
    docs = [
        dict(doc="t.md", clause="同一条", version="2026-09-25", source="测试",
             text="宿舍内禁止使用大功率电器，一经发现没收器具并给予通报批评处理。"),
        dict(doc="t.md", clause="同一条", version="2026-9-5", source="测试（未补零）",
             text="学生请假一天以内由班主任审批，一周以内由二级学院审批，一周以上须三级审批。"),
    ]
    kept, rejected = rag._guard(docs)
    assert len(kept) == 1
    assert kept[0]["version"] == "2026-09-25", (
        f"未补零的 2026-9-5 把更新的 2026-09-25 挤掉了（还在按字符串比）："
        f"留下的是 {kept[0]['version']}，拒绝理由 {rejected}")


def test_unparseable_version_rejected_not_guessed(rag):
    """★ 比不出新旧时【不许猜】—— 明确拦下并说清原因。

    为什么专门守这条：`_version_key` 解析失败返回 None，调用方必须显式处理。
    如果哪天有人图省事改成"解析不出来就退回字符串比较"，
    就等于"看着校验过了，其实没有"，比不校验更危险（下一个读代码的人会以为这里安全）。
    """
    docs = [
        dict(doc="t.md", clause="同一条", version="2026-09-25", source="测试",
             text="宿舍内禁止使用大功率电器，一经发现没收器具并给予通报批评处理。"),
        dict(doc="t.md", clause="同一条", version="2026/09/25", source="测试（斜杠）",
             text="学生请假一天以内由班主任审批，一周以内由二级学院审批，一周以上须三级审批。"),
    ]
    kept, rejected = rag._guard(docs)
    assert len(kept) == 1
    assert any("无法比较新旧" in r for _, r in rejected), (
        f"日期格式比不出来时应当拦下并说明原因，实际 rejected={rejected}")


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

    ★ 这里 clause 故意写成【不同】（D1 / D2）：判重只看文本像不像，不看条款名。
      这不是随便定的 —— 2026-09-25 踩过坑：真实语料里那条该拦的完全相同文本
      （余弦 1.0000）两条 clause 名就是不一样的，一度加过"必须同条款才判重"的
      条件，结果把它放行了。所以这条同时守住"别再加身份判断回去"。
      反过来，不同条款但主题相近（余弦 0.9230）不该被拦 —— 见下面那条测试。
    """
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

    # 应只留 A 和 C 各一条，两条改写版都被近似重复关拦掉。
    assert sorted(c["clause"] for c in kept) == ["A", "C"], (
        f"4 条里应只留 A 和 C 各一条。实际留了：{[c['clause'] for c in kept]}")
    assert sum(1 for _, r in rejected if "重复" in r) == 2, (
        f"应有两条因重复被拦，实际：{[(c['text'][:18], r) for c, r in rejected]}")


def test_dup_threshold_has_margin(rag):
    """★ 守住"阈值不误伤"这件事本身，而不只是守住今天的数字。

    背景：DUP_THRESHOLD 一度是 0.90，真实语料里【第二十四条 vs 第二十五条】
    余弦 0.9230 > 0.90，于是第二十五条被当成重复删掉 —— 库里凭空少一条规则，
    而所有"计数"类断言只要跟着改数字就能全绿，根本发现不了。

    所以这里盯的不是"拦了几条"，而是【那条最像的非重复配对，离阈值还有多远】。
    以后往 docs/ 加文档、把相似度顶上去了，这条会先红，
    逼你重新量分布（scripts/measure_dup_distribution.py），而不是静默删内容。
    """
    from service import DUP_THRESHOLD
    import numpy as np

    # 原 bug 的主角必须在库里：第二十五条一度被第二十四条挤掉（0.9230 > 旧的 0.90）
    by = {(c["doc"], c["clause"]) for c in rag.chunks}
    for c in ("第三章-第二十四条", "第三章-第二十五条"):
        assert ("学生纪律处分管理规定.md", c) in by, f"{c} 不在库里 —— 被误删了"

    # ★ 扫【全库所有存活块的两两配对】取最大值，而不是只盯 24/25 那一对。
    #   只盯一对的话，以后往 docs/ 加了新文档、最像的一对换了人，这条看不见。
    vecs = np.array(list(rag.model.passage_embed([c["text"] for c in rag.chunks])),
                    dtype="float32")
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)                       # 自己和自己不算
    i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
    top = float(sim[i, j])

    assert top < DUP_THRESHOLD, (
        f"库里最像的一对非重复块余弦 {top:.4f} 已经顶到阈值 {DUP_THRESHOLD} 了 "
        f"（{rag.chunks[i]['clause']} vs {rag.chunks[j]['clause']}），再靠近就要误删条款。"
        f"去重跑 scripts/measure_dup_distribution.py 重新定阈值。")
    # 余量也要够：贴着阈值站住说明下次加文档就会翻车。
    assert DUP_THRESHOLD - top > 0.02, (
        f"余量只剩 {DUP_THRESHOLD - top:.4f}（{rag.chunks[i]['clause']} vs "
        f"{rag.chunks[j]['clause']}），太薄，需要重新量分布定阈值。")


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
    """阈值必须真的能把"该答的"和"该拒的"分开。

    ★ 2026-09-25 修过一处"测的是另一个量"的问题（cc 审查发现）：
        这里原本用 `rag.search(q)[0][1]`，也就是 **RRF 融合后第一名**的向量分；
        而生产代码（service.py 的 else 分支）用的是 `max(全部候选的向量分)`。
        两者不是一回事 —— RRF 可能把 BM25 命中的块顶到第一，而它向量分未必最高。
        实测差得不小（"处分有哪几种？" 0.5715 vs 0.6087），只是这 13 题恰好没翻盘。
        现在改成和生产一致的 max。

    ★ 题库只放【没有争议的题】：evalset 里的两道 boundary 题
        （q19「宿舍几点熄灯」、q20「学生证补办」）不在下面两个列表里。
        q20 向量分 0.6049 > 0.55，纯向量路会放行 —— 这是已知的离线层边界，
        由 evalset/report.md 记录（拒答准确率 5/6），产品默认走 rerank 兜底。
        把它塞进 SHOULD_REFUSE 只会让这条测试长期红着没人看，反而掩盖真信号。
    """
    for q in SHOULD_HIT:
        top = max(s for _, s in rag.search(q, k=5))       # 和生产一致：取全部候选的 max
        assert top >= svc.VEC_REJECT_THRESHOLD, (
            f"「{q}」库里有，却因向量分 {top:.4f} 低于阈值 {svc.VEC_REJECT_THRESHOLD} 被误拒")
    for q in SHOULD_REFUSE:
        top = max(s for _, s in rag.search(q, k=5))
        assert top < svc.VEC_REJECT_THRESHOLD, (
            f"「{q}」库里没有，却因向量分 {top:.4f} 高于阈值 {svc.VEC_REJECT_THRESHOLD} 被放行")


def test_guard_report_masks_pii(rag):
    """★ 补 PII 盲区：/guard-report 返回的 rejected 不得明文含手机号/身份证。
    之前 `_mask_pii` 定义了却从未调用，rejected 只做了 `[:40]` 截断，
    而那条隐私数据仅 30 字，截断无效 → 手机号/身份证原样泄露。这条守住它。

    ★★ 2026-09-25 二修：这条曾经是【摆设】，cc 审查发现、实测确认。
        换真实语料后，真实被拦的 15 条全是"内设机构/院长信箱"这类导航残留，
        **一条 PII 都没有** —— 于是把 `_mask_pii(...)` 改回 `c["text"]`（脱敏整个撤掉），
        这条照样全绿。它守不住自己 docstring 里写的那个 bug。
        原因：断言遍历的是"被拦的条目"，而被拦的条目里没有 PII，等于什么都没检查。
        改法：先把脏数据夹具（含真手机号）跑一遍把关，临时塞进 rag.rejected 再走
        guard_report() —— 保证【真的有 PII 经过脱敏函数】下面那两个 assert 才有意义。
    """
    from kb.loader import load_documents

    docs, _ = load_documents(DIRTY_DIR)
    _kept, rejected = rag._guard(docs)

    # 先自查：夹具里必须真的有手机号，否则下面又变成"遍历空列表 → 永远全绿"
    assert any("13812345678" in c["text"] for c, _ in rejected), (
        "脏数据夹具里没有含手机号的块 —— 这条测试会退回摆设状态，去补 fixtures/dirty_inbox/04")

    original = svc.rag.rejected
    try:
        svc.rag.rejected = rejected
        rep = svc.guard_report()
    finally:
        svc.rag.rejected = original          # 别污染其它用例（rag 是模块级共享的）

    for item in rep["rejected"]:
        assert "13812345678" not in item["text"], f"手机号明文泄露：{item['text']}"
        assert "440301199001011234" not in item["text"], f"身份证明文泄露：{item['text']}"
    # 反向确认：脱敏确实生效（不是"因为没走到那"才通过的）
    assert any("手机号已隐匿" in item["text"] for item in rep["rejected"]), (
        f"没看到脱敏痕迹，_mask_pii 大概没被调用：{rep['rejected']}")


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


# ==================================================================
# ⑦ 把关【覆盖度】：六道关每一道都必须真的被触发过
# ==================================================================
DIRTY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "dirty_inbox")


def test_all_gates_fire_on_dirty_corpus(rag):
    """★ 解决"关卡覆盖度"：一条断言保证 5 道关 + 版本冲突全都被触发。

    问题背景（2026-09-25 量出来的）：
        真实语料是学校公开制度，太干净 —— 入库跑完只有"太短碎屑"拦了 15 条，
        其余四关 + 版本冲突触发次数全是 0。
        这意味着"删掉隐私关"这类破坏，所有计数断言（68 / 15）依然全绿 ——
        **覆盖度是 0，但看不出来**。

    为什么不能靠真实语料证明：
        防御性代码在干净输入上天然不触发，就像 WAF 不能用正常流量证明有效。
        所以覆盖度只能用【注入的脏样本】证明 —— 见 tests/fixtures/dirty_inbox/README.md。

    这里走的是完整链路（load_documents → _guard），不是直接喂 dict，
    所以连 .md 解析、frontmatter、分块也一并被覆盖了。
    """
    from kb.loader import load_documents

    docs, errors = load_documents(DIRTY_DIR)
    assert errors == [], f"夹具文件解析失败：{errors}"
    assert len(docs) == 10, f"夹具块数不对，加载了 {len(docs)} 块"

    kept, rejected = rag._guard(docs)
    reasons = [r for _, r in rejected]

    for gate in ("太短碎屑", "完全重复", "近似重复", "个人隐私", "缺少来源", "旧版本"):
        assert any(gate in r for r in reasons), (
            f"【{gate}】关在脏数据夹具上没触发 —— 关卡被删了或改坏了。"
            f"实际拦下的原因：{reasons}")

    # ★ 关卡 5 有【两个半边】（source 空 / version 空），必须各测一次。
    #   只测半边的后果实测过：把判断改成 `if not c.get("source")`（version 那半删掉），
    #   上面那条 for 循环照样全绿 —— 因为"缺少来源"这个原因字符串还会出现一次。
    #   所以这里按条款名分别点名，删掉任意半边都会红。
    no_trace = {(c["doc"], c["clause"]) for c, r in rejected if "缺少来源" in r}
    assert ("脏-缺少来源.md", "无来源") in no_trace, f"source 为空没被拦：{no_trace}"
    assert ("脏-缺少日期.md", "无日期") in no_trace, \
        f"version 为空没被拦（关卡5的 version 半边失效了）：{no_trace}"

    # 夹具是纯脏数据，最后只该剩下 3 条「每组里活下来的那一条」：
    #   完全重复组留先到的"重复块一" + 近似重复组留先到的"近似一" + 版本冲突留新版。
    # ★ 写死具体条款而不是只写数字：数字断言在"多了/少了但总数不变"时会漏。
    kept_ids = {(c["doc"], c["clause"]) for c in kept}
    assert kept_ids == {
        ("脏-完全重复.md", "重复块一"),
        ("脏-近似重复.md", "近似一"),
        ("脏-版本冲突.md", "审批权限"),
    }, f"活下来的块不对：{[(c['doc'], c['clause']) for c in kept]}"
    # 版本冲突留下的必须是新版那条（内容"宿舍大功率电器"，不是旧版的"请假审批"）
    winner = next(c for c in kept if c["doc"] == "脏-版本冲突.md")
    assert winner["version"] == "2026-09-25", f"留的是旧版：{winner['version']}"
    assert "大功率电器" in winner["text"], f"版本冲突留错了版本：{winner['text'][:30]}"


def test_real_corpus_gate_coverage_is_documented(rag):
    """真实语料上哪些关【从不触发】—— 把这件事钉成一条会红的断言，而不是藏起来。

    用途：如果哪天真实语料脏了、某道关开始触发，这条会红，
    逼你去更新 README 里的"关卡覆盖度"表 —— 而不是让文档悄悄过期。
    """
    from collections import Counter

    counts = Counter(r for _, r in rag.rejected)
    # 当前真实语料只有"太短碎屑"触发（网页抓取残留全是短导航词）
    assert set(counts) == {"太短碎屑"}, (
        f"真实语料的拦截原因变了：{dict(counts)}。"
        f"README 里的关卡覆盖度表需要同步更新。")


# ==================================================================
# 防漂移：体检脚本里"手抄的常量"必须和真值一致
# ==================================================================
def test_health_script_min_len_is_not_drifted():
    """★ 体检脚本手抄的 MIN_LEN 必须等于 service 的真值。

    和 tests/test_loader.py::test_health_script_max_chars_is_not_drifted 是一对，
    只是真值住在不同的文件里（max_chars 在 kb/loader、MIN_LEN 在 service），
    所以拆成两条 —— 一条断言住一个真值，漂移时能直接看出是哪个数错了。

    ★ 为什么这条住在 test_guard.py 而不是 test_loader.py：
      真值 MIN_LEN 在 service.py，而 test_loader.py 刻意不 import service
      （只是想跑切片逻辑的人不该被拖着 import fastembed、慢几秒）。
      test_guard.py 本来就 import 了 service，顺手断言最省代价。
      **别为了"两个测试排在一起好看"就把 service 拖进 test_loader.py。**
    """
    import scripts.measure_chunk_health as health
    assert health.MIN_LEN_HINT == svc.MIN_LEN, (
        f"体检脚本抄的是 {health.MIN_LEN_HINT}，service 真值是 {svc.MIN_LEN} —— 该去改脚本了")
