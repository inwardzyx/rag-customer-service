# -*- coding: utf-8 -*-
"""
50-55 集重讲：节点防护链的「代码结构」
=========================================================
改代码之前，先想清楚该往【哪一层】加。一共四层：

   第 1 层  节点函数内部    try/except + timeout=...          50、51 集
   第 2 层  add_node 参数   retry=RetryPolicy(...)            48、49 集
                            cache_policy=CachePolicy(...)     52-54 集
   第 3 层  compile 参数    compile(cache=InMemoryCache())    55 集
   第 4 层  invoke 的 config invoke(..., config={             44 集
                                "recursion_limit": 10})

记忆口诀：
   内层管「这一次调用」，外层管「整张图怎么跑」。
   第 2 层只是「打申请」，第 3 层才是「批钱」——
   只写 CachePolicy 不给 compile(cache=...)，缓存完全不生效。

下面用同一个 call_api 节点，从第 0 步（裸奔）开始一层层加，
每一步只多几行，跑一遍就能看出差别。不需要 API key。
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy

# =========================================================
# 公共零件：一个「不稳定的外部服务」
# =========================================================

_CALLS: dict = {}   # 记录每个 query 一共被【真正调用】过几次，用来验证缓存/重试


def unstable_service(query: str, delay: float = 0.2, fail_times: int = 0) -> str:
    """模拟 LLM API / 第三方接口。
    delay      : 每次要多久才返回
    fail_times : 前几次调用必定抛 ConnectionError
    """
    n = _CALLS.get(query, 0) + 1
    _CALLS[query] = n
    time.sleep(delay)
    if n <= fail_times:
        raise ConnectionError(f"第 {n} 次调用挂了：连接被重置")
    return f"「{query}」的结果"


def call_with_timeout(fn, timeout: float):
    """真·超时：另起一个线程跑 fn，超过 timeout 秒还没回来就抛 TimeoutError。
    平时写的 requests.get(url, timeout=5) / ChatOpenAI(timeout=30) 底层就是这件事。"""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)   # 别干等那个卡住的线程，让异常立刻冒出去


class State(TypedDict):
    query: str
    result: str
    error: str


def banner(n, title):
    print("\n" + "=" * 64)
    print(f"第 {n} 步：{title}")
    print("=" * 64)


def new_input(q="广州天气"):
    return {"query": q, "result": "", "error": ""}


# =========================================================
# 第 0 步：裸奔
# =========================================================
def step_0_bare():
    banner(0, "裸节点：什么都不加（这就是 50-55 集要解决的起点）")

    def call_api(state):
        # 第 1 层：空的，直接调，出事就出事
        return {"result": unstable_service(state["query"], fail_times=1)}

    g = StateGraph(State)
    g.add_node("call_api", call_api)
    g.add_edge(START, "call_api")
    g.add_edge("call_api", END)
    app = g.compile()

    _CALLS.clear()
    try:
        print("结果：", app.invoke(new_input()))
    except Exception as e:
        print(f"整个 invoke 崩了 → {type(e).__name__}: {e}")
    print("实际调用服务次数：", _CALLS)
    print("↑ 一次失败就全盘崩溃，用户什么都拿不到")


# =========================================================
# 第 1 步：加超时（第 1 层）
# =========================================================
def step_1_timeout():
    banner(1, "第 1 层：加超时（50 集）—— 写在「调用那一行」")

    def call_api(state):
        # 新增只有一处：把调用包起来，最多等 0.5 秒
        return {"result": call_with_timeout(
            lambda: unstable_service(state["query"], delay=2),
            timeout=0.5,
        )}

    g = StateGraph(State)
    g.add_node("call_api", call_api)
    g.add_edge(START, "call_api")
    g.add_edge("call_api", END)
    app = g.compile()

    _CALLS.clear()
    t0 = time.time()
    try:
        app.invoke(new_input("慢查询"))
    except Exception as e:
        print(f"等了 {time.time() - t0:.1f} 秒就放弃 → {type(e).__name__}")
    print("↑ 服务要 2 秒，我们只等 0.5 秒。")
    print("  注意：超时之后的行为是【抛异常】，所以超时必须配下一步才有意义")


# =========================================================
# 第 2 步：错误处理 / 降级兜底（第 1 层 + 条件边）
# =========================================================
def step_2_fallback():
    banner(2, "第 1 层：try/except 把错误写成数据 + 条件边送兜底（51 集）")

    def call_api(state):
        try:                                                   # ← 新增
            r = unstable_service(state["query"], fail_times=99)  # 永远失败
            return {"result": r, "error": ""}
        except Exception as e:
            # 关键：不把异常抛出去，而是当成【数据】写进 State
            return {"error": str(e)}

    def fallback(state):
        return {"result": "（兜底）服务暂时不可用，请稍后再试"}

    def route(state):
        return "fallback" if state.get("error") else END         # ← 看 State 分流

    g = StateGraph(State)
    g.add_node("call_api", call_api)
    g.add_node("fallback", fallback)
    g.add_edge(START, "call_api")
    g.add_conditional_edges("call_api", route)                   # ← 第 1 层配的开关
    g.add_edge("fallback", END)
    app = g.compile()

    _CALLS.clear()
    out = app.invoke(new_input())
    print("最终 result：", out["result"])
    print("State 里的 error：", out["error"])
    print("↑ 图没崩，用户拿到了一个能看的回复 —— 这就是「降级」")


# =========================================================
# 第 3 步：重试（第 2 层）
# =========================================================
def step_3_retry():
    banner(3, "第 2 层：add_node 上挂 RetryPolicy（48、49 集）")

    def call_api(state):
        # 注意：这里【故意不写 try/except】——异常必须抛出去，框架才有东西可重试
        return {"result": unstable_service(state["query"], fail_times=2)}

    g = StateGraph(State)
    g.add_node("call_api", call_api, retry=RetryPolicy(     # ← 第 2 层
        max_attempts=4,                 # 最多试 4 次
        retry_on=(ConnectionError,),    # 只重试网络类偶发错误
        initial_interval=0.1,           # 第一次失败后等 0.1 秒
        backoff_factor=2.0,             # 之后翻倍：0.1 → 0.2 → 0.4
    ))
    g.add_edge(START, "call_api")
    g.add_edge("call_api", END)
    app = g.compile()

    _CALLS.clear()
    out = app.invoke(new_input())
    print("结果：", out["result"])
    print("实际调用服务次数：", _CALLS)
    print("↑ 前 2 次失败、第 3 次成功，节点自己好了，调用方完全无感")


def step_3b_pitfall():
    banner("3b", "坑：节点里吞了异常，RetryPolicy 就永远不生效")

    def call_api(state):
        try:
            return {"result": unstable_service(state["query"], fail_times=2), "error": ""}
        except Exception as e:
            return {"error": str(e)}     # 异常被吞了 → 框架看到的是「节点正常返回了」

    g = StateGraph(State)
    g.add_node("call_api", call_api,
               retry=RetryPolicy(max_attempts=4, retry_on=(ConnectionError,),
                                 initial_interval=0.1))
    g.add_edge(START, "call_api")
    g.add_edge("call_api", END)
    app = g.compile()

    _CALLS.clear()
    out = app.invoke(new_input())
    print("结果：", out)
    print("实际调用服务次数：", _CALLS, "← 只调了 1 次，retry 根本没触发")
    print("结论：重试（第 2 层）和降级（第 1 层）是二选一的两条路，别叠在同一层。")
    print("      想重试就别吞异常；想兜底就别配 retry。")


# =========================================================
# 第 4 步：缓存（第 2 层 + 第 3 层）
# =========================================================
def step_4_cache():
    banner(4, "第 2 层 + 第 3 层：CachePolicy + compile(cache=...)（52-55 集）")

    def slow(state):
        time.sleep(1)                                  # 假装很慢
        return {"result": unstable_service(state["query"])}

    g = StateGraph(State)
    g.add_node("slow", slow, cache_policy=CachePolicy(ttl=60))   # 第 2 层：打申请
    g.add_edge(START, "slow")
    g.add_edge("slow", END)

    app = g.compile(cache=InMemoryCache())             # 第 3 层：批钱（给仓库）

    _CALLS.clear()
    for i, q in enumerate(["A", "A", "B"], 1):
        t0 = time.time()
        out = app.invoke(new_input(q))
        print(f"  第 {i} 次 query={q} → {out['result']}   耗时 {time.time() - t0:.2f} 秒")
    print("实际调用服务次数：", _CALLS)
    print("↑ 第 2 次输入相同 → 命中缓存，节点根本没跑（0 秒）")
    print("  第 3 次输入变了 → 钥匙不同，重新跑")

    # 反例：只打申请不批钱
    _CALLS.clear()
    app2 = g.compile()                                 # ← 没给 cache=...
    for q in ["A", "A"]:
        t0 = time.time()
        app2.invoke(new_input(q))
        print(f"  [反例] query={q} 耗时 {time.time() - t0:.2f} 秒")
    print("  [反例] 实际调用服务次数：", _CALLS, "← 两次都真跑了，缓存没生效")


# =========================================================
# 第 5 步：四层合一
# =========================================================
def step_5_all():
    banner(5, "四层合一：超时 + 重试 + 缓存 + 步数上限（50、48、52、44 集）")

    # 用一个可变容器控制"这个服务还要失败几次"，方便同一张图演示两种情况
    fail = {"times": 2}

    def call_api(state):
        # 第 1 层：超时（真调 LLM 时写 ChatOpenAI(timeout=30)）
        # 不吞异常 → 留给第 2 层的 RetryPolicy
        return {"result": call_with_timeout(
            lambda: unstable_service(state["query"], delay=0.2, fail_times=fail["times"]),
            timeout=2,
        )}

    g = StateGraph(State)
    g.add_node("call_api", call_api,
               retry=RetryPolicy(max_attempts=3,               # 第 2 层
                                 retry_on=(ConnectionError, TimeoutError),
                                 initial_interval=0.1),
               cache_policy=CachePolicy(ttl=60))                # 第 2 层
    g.add_edge(START, "call_api")
    g.add_edge("call_api", END)

    app = g.compile(cache=InMemoryCache())                      # 第 3 层

    _CALLS.clear()
    # 第 4 层：整张图最多跑 10 步，防止转不停
    cfg = {"recursion_limit": 10}

    t0 = time.time()
    out = app.invoke(new_input(), config=cfg)
    print(f"① 重试后成功 → {out['result']}  耗时 {time.time() - t0:.2f} 秒  调用次数 {_CALLS}")

    t0 = time.time()
    out = app.invoke(new_input(), config=cfg)
    print(f"② 同样输入再来 → {out['result']}  耗时 {time.time() - t0:.2f} 秒  调用次数 {_CALLS}")
    print("   ↑ 耗时接近 0、调用次数没涨 → 命中缓存，钱省下来了")

    fail["times"] = 99          # 让这个服务彻底坏掉
    try:
        app.invoke(new_input("北京天气"), config=cfg)
    except Exception as e:
        print(f"③ 重试 3 次全失败，异常冒到最外面 → {type(e).__name__}  调用次数 {_CALLS}")
        print("   （真实项目里在这一层给用户返回一句「稍后再试」）")


# =========================================================
# 速查表
# =========================================================
CHEATSHEET = """
================================================================
速查表：想干什么 → 往哪一层加
================================================================
 想达到的效果        加在哪一层          代码长什么样
 ----------------------------------------------------------------
 限制单次调用时间     节点函数内部        call_with_timeout(fn, timeout=2)
                                        requests.get(url, timeout=5)
                                        ChatOpenAI(timeout=30)

 偶发失败自动重试     add_node(retry=)   RetryPolicy(
                                            max_attempts=3,
                                            retry_on=(ConnectionError,),
                                            initial_interval=1.0,
                                            backoff_factor=2.0)

 结果复用不重复花钱   add_node(           CachePolicy(ttl=60)
                       cache_policy=)     + compile(cache=InMemoryCache())
                     + compile(cache=)    ← 两个零件缺一不可

 出错也要给个回复     节点内 try/except   return {"error": str(e)}
                     + 条件边分流        add_conditional_edges("x", route)

 防止整图转圈不止     invoke 的 config    config={"recursion_limit": 10}
================================================================
 顺序口诀：先看缓存有没有 → 再定超时多久 → 失败先重试 →
           重试不行就降级 → 全程有步数上限兜底
================================================================
"""


if __name__ == "__main__":
    step_0_bare()
    step_1_timeout()
    step_2_fallback()
    step_3_retry()
    step_3b_pitfall()
    step_4_cache()
    step_5_all()
    print(CHEATSHEET)
