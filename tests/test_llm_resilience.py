# -*- coding: utf-8 -*-
"""
阶段 0「安全 / 可靠硬门槛」的守门测试

这一组用例盯的是同一件事，一句话概括：
    【模型出问题 ≠ 知识库里没有】

    前者是【故障】，必须报错或降级，而且要留下日志，让人能发现；
    后者是【正常的业务结果】（库里确实没有 → 拒答）。
    两者一旦混同，服务挂了也会表现成一条平平无奇的拒答，
    调用方永远察觉不到，故障就被永久掩盖了 —— 这正是最危险的一类 bug。

跑法（仓库根目录）：
    python -m pytest tests/test_llm_resilience.py -v

★ 不需要 DEEPSEEK_API_KEY：全部用假模型，一条都不真调大模型。
"""

import json
import os
import sys
import time

# 这两行必须在 import 项目模块之前（和 test_guard.py 同样的原因）
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest                                      # noqa: E402
import service as svc                              # noqa: E402


# ---- 测试用的假模型 ----
class _FakeResp:
    """假返回：只要有 .content 属性，长得和真模型返回的对象一样"""

    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """不管问什么都返回指定字符串 —— 用来模拟"模型说了句没用的话" """

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return _FakeResp(self.reply)


class _SlowLLM:
    """模拟"模型卡住了"：一直不返回。

    注意 sleep 只有 1 秒（不是 5 秒）—— 超时后那个线程其实还在后台跑完，
    Python 退出时会等它，所以故意设短一点，免得拖慢整个测试。
    """

    def invoke(self, messages):
        time.sleep(1)
        return _FakeResp("{}")


@pytest.fixture(scope="module")
def rag():
    """建库（加载 embedding 模型要十几秒，整个文件共用一份）"""
    svc.rag.startup()
    return svc.rag


# ==================================================================
# ① 超时：模型卡住不能拖垮接口
# ==================================================================
def test_timeout_becomes_llm_call_error(monkeypatch):
    """模型卡住时必须变成 LLMCallError，而不是让调用方无限期干等。

    ★ 这条顺带守住一个隐蔽的坑：如果 _invoke_llm 的签名写成
      timeout=LLM_TIMEOUT_SEC（默认参数），那么这里 monkeypatch 模块变量
      根本不会生效（默认参数在【函数定义那一刻】就求值完了），
      测试就会假绿。所以函数体内再读一次是必须的。
    """
    monkeypatch.setattr(svc, "LLM_TIMEOUT_SEC", 0.05)
    with pytest.raises(svc.LLMCallError):
        svc._invoke_llm(_SlowLLM(), ["hi"])


# ==================================================================
# ② 模型返回垃圾：必须报错，绝不能静默给全 0 分
# ==================================================================
def test_rerank_bad_json_raises_not_silent_zero(rag, monkeypatch):
    """★ 这一组里最关键的一条。

    以前的写法是"解析失败就给所有候选 0 分" —— 这看似温和，实际最危险：
    全 0 分 → 最高分 < 5 → 触发拒答 → 界面上显示"知识库里没有"。
    于是"模型返回了垃圾"（故障）被完美伪装成"库里没有"（正常业务结果），
    日志里也什么都不留，没人会知道服务其实已经不正常了。
    现在必须抛 LLMCallError，让 /chat 去决定降级还是暴露。
    """
    monkeypatch.setattr(svc.rag, "_llm", _FakeLLM("抱歉，我这次不想输出 JSON"))
    q = "什么情况会被开除学籍？"
    cands = svc.rag.search(q, k=3)
    assert cands, "检索应该命中，否则这条用例没真正测到 rerank"

    with pytest.raises(svc.LLMCallError):
        svc.rag.rerank(q, cands)


# ==================================================================
# ③ 幻觉 id：越界不能 500，要跳过
# ==================================================================
def test_rerank_skips_hallucinated_id(rag, monkeypatch):
    """模型会幻觉出不存在的 id（超出候选数量、0 甚至负数）。

    以前直接 candidates[i - 1] 会 IndexError → 接口裸 500。
    现在应该跳过越界的那条，合法的照常保留。
    """
    payload = json.dumps({"scores": [
        {"id": 1, "score": 9, "reason": "相关"},
        {"id": 99, "score": 10, "reason": "幻觉出来的 id"},
    ]})
    monkeypatch.setattr(svc.rag, "_llm", _FakeLLM(payload))
    q = "什么情况会被开除学籍？"
    cands = svc.rag.search(q, k=3)

    out = svc.rag.rerank(q, cands)

    assert len(out) == 1, f"越界的 id=99 应被跳过，只留 id=1，实际返回 {len(out)} 条"
    assert out[0][2] == 9, "留下来的应该是 id=1 那条（精排分 9）"


# ==================================================================
# ④ 超长输入：在入口就被挡掉
# ==================================================================
def test_question_too_long_is_rejected():
    """没有 max_length 的话，别人丢一篇几万字过来会被原样拼进 prompt。

    后果有两个：烧掉大量 token（费钱又慢），极端时顶爆模型上下文窗口。
    pydantic 在入口挡掉并返回 422，不用业务代码里写 if。
    """
    from fastapi.testclient import TestClient

    client = TestClient(svc.app)
    r = client.post("/chat", json={"question": "啊" * 5000})
    assert r.status_code == 422, f"期望 422，实际 {r.status_code}"


# ==================================================================
# ⑤ 端到端降级：模型挂了要如实说，不能伪装成"库里没有"
# ==================================================================
def test_chat_degrades_instead_of_faking_rejection(rag, monkeypatch):
    """★ 端到端兜底：资料命中了、但生成那一步挂了，knowledge_hit 必须还是 True。

    如果这里返回 False，调用方会以为"这个问题库里根本没有"，
    于是故障又一次被伪装成正常业务结果 —— 和 ② 是同一个病的两种表现。
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(svc, "LLM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(svc.rag, "_llm", _SlowLLM())

    # 不用 with TestClient(...)：rag 已由 fixture 启动，避免 lifespan 再加载一次模型
    client = TestClient(svc.app)
    r = client.post("/chat", json={"question": "什么情况会被开除学籍？",
                                   "use_rerank": False})

    assert r.status_code == 200, f"期望 200，实际 {r.status_code}：{r.text[:200]}"
    d = r.json()
    assert d["knowledge_hit"] is True, "命中了就是命中了，不能因为模型挂了就说库里没有"
    assert "模型调用失败" in d["answer"], f"应如实说明是模型的问题，实际：{d['answer'][:80]}"
