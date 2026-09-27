# -*- coding: utf-8 -*-
"""测试用的"假模型"（test doubles）—— 三个测试文件共用。

★ 为什么放在 conftest.py，而不是各测试文件里各写一份：
  `_FakeLLM` 原本在 test_guard.py 和 test_llm_resilience.py 里【各抄了一份】
  （commit ab9a71c 定性的正是"同一颗哑弹抄两份"这类问题）。
  test_eval_set.py 这次也要用 —— 再抄就是第三份。
  pytest 会先加载 conftest.py，各测试文件 `from conftest import ...` 即可拿到。

★ 为什么要有假货：真模型要联网、要 API key、每次结果还可能不一样。
  测试关心的是"流程对不对"（该拒答时拒答、该命中时命中），不是"模型说了什么"，
  所以把模型换成人偶，让流程能脱离网络被反复验证。

★ 用假模型【不等于】放弃验证真实调用：
  真的 `ChatDeepSeek(...)` 构造目前【没有任何测试覆盖】（构造就要 key），
  所以 service.py 里 `llm` 属性那几个 kwarg 写错了也没人会发现 —— 这是已知的账，
  靠真机跑 / 手工验，不靠这一个假模型来兜。
"""

import time


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


class _SlowLLM:
    """模拟"模型卡住了"：一直不返回。

    注意 sleep 只有 1 秒（不是 5 秒）—— 超时后那个线程其实还在后台跑完，
    Python 退出时会等它，所以故意设短一点，免得拖慢整个测试。
    """

    def invoke(self, messages):
        time.sleep(1)
        # 睡完还是返回一个值（原来就在 test_llm_resilience.py 里，搬过来保持行为一致）。
        # 实际测试中超时（0.05 秒）先触发，这一行不会被观察到 —— 留着是为了
        # 将来有人把超时调长之后，它不至于变成"永远不返回"的另一颗哑弹。
        return _FakeResp("{}")
