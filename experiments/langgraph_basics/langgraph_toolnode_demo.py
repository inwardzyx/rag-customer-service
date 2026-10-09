"""
101-105 集：工具节点（ToolNode）—— agent 的核心机关

    101/102  手动调用工具        → 自己写"读工单-执行-回执"的节点
    103      ToolNode 替代手动   → 两个内置件替代上面全部手写代码
    104      ToolRuntime         → 工具函数里直接拿到图的 state / tool_call_id
    105      wrap_tool_call      → 给工具执行套一层"拦截器"（这里演示自动重试）

真实场景里"决定调哪个工具"的是 LLM（bind_tools）；本文件用假 LLM 节点
返回固定的 tool_calls，把注意力全放在【执行侧】的机制上。

⚠️ 跑之前建议先关掉 LangSmith 上报（否则会联网上传 trace，慢的时候能把脚本拖住）：
    PowerShell / Git Bash 里先执行：  set LANGSMITH_TRACING=false
    Windows cmd 里：                  set LANGSMITH_TRACING=false

跑法：
    <你的虚拟环境>/Scripts/python.exe langgraph_toolnode_demo.py
"""

import operator
from typing import Annotated, TypedDict

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env_compat                                   # noqa: E402  ★ 必须在 import langchain 之前
env_compat.ensure_mmh3()                            # noqa: E402  本机 DLL 被策略拦截时的降级方案
env_compat.ensure_uuid_utils()                      # noqa: E402  同上，拦的是 uuid_utils

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition


def sep(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


class State(TypedDict):
    messages: Annotated[list, add_messages]      # 对话历史，追加式


# ==================================================================
sep("101：手动调用工具（上）—— 单个工具，自己写执行节点")
# ==================================================================

@tool
def query_leave_record(student_id: str) -> str:
    """根据学号查询请假记录的当前状态。"""
    #  ↑ 这行 docstring 不是写给人看的，是写给 LLM 看的【说明书】，
    #    LLM 就靠它决定什么时候该调这个工具
    leave_records = {"2024010101": "事假已批准，销假单已归档", "2024010202": "待销假"}
    return leave_records.get(student_id, f"没找到该学号的请假记录 {student_id}")


def fake_llm_single(state):
    """假 LLM：真实场景里这一步由 bind_tools 过的真 LLM 完成，
    它看你的问题和工具说明书，决定『我要调 query_leave_record，参数 student_id=2024010101』。"""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):      # 已经拿到工具回执了 → 该收尾，不再开工单
        return {"messages": [AIMessage(content="您的学号 2024010101 事假已批准，销假单已归档。")]}
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "query_leave_record", "args": {"student_id": "2024010101"}, "id": "call_001"}],
    )]}
#   ↑ 这三行是 ReAct 循环能停下来的关键：有回执就答，没回执才开工单。
#     少了它，图会 llm → tools → llm → tools …… 一直转到 recursion_limit 撞墙。


def manual_tool_node(state):
    """手动版工具节点：读工单 → 执行 → 写回执。这就是 101 集要手写的全部。"""
    last = state["messages"][-1]          # 最后一条消息 = LLM 的"工单"
    results = []
    for tc in last.tool_calls:            # 每张工单：name + args + id
        fn = {"query_leave_record": query_leave_record}[tc["name"]]   # 按名字找到函数
        result = fn.invoke(tc["args"])    # 执行（invoke 传参数字典）
        results.append(ToolMessage(
            content=result,               # 回执内容
            tool_call_id=tc["id"],        # ← 对号：告诉 LLM 这张回执对应哪张工单
        ))
    return {"messages": results}


def route(state):
    """手动版条件边：最后一条有 tool_calls → 去工具节点；没有 → 结束"""
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


g1 = StateGraph(State)
g1.add_node("llm", fake_llm_single)
g1.add_node("tools", manual_tool_node)
g1.add_edge(START, "llm")
g1.add_conditional_edges("llm", route, {"tools": "tools", END: END})
g1.add_edge("tools", "llm")               # 回执交给 LLM 看一眼
app1 = g1.compile()

out = app1.invoke({"messages": [HumanMessage("我的学号 2024010101 审批到哪一步了？")]})
for m in out["messages"]:
    print(f"   [{type(m).__name__}] {m.content!r}")
print("""
   一次完整循环：人提问 → LLM 开工单 → 工人执行写回执 → LLM 看回执。
   三个角色，靠 tool_calls 里的 id 和 tool_call_id 一一对号。""")

# ==================================================================
sep("102：手动调用（下）—— 一张工单列表，回执必须逐张配对")
# ==================================================================

def fake_llm_parallel(state):
    """假 LLM 一次开两张工单（真实 LLM 也经常这样并行调多个工具）"""
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [AIMessage(content="2024010101 事假已批准；2024099999 查无此记录。")]}
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "query_leave_record", "args": {"student_id": "2024010101"}, "id": "call_a"},
        {"name": "query_leave_record", "args": {"student_id": "2024099999"}, "id": "call_b"},
    ])]}

g2 = StateGraph(State)
g2.add_node("llm", fake_llm_parallel)
g2.add_node("tools", manual_tool_node)    # 同一个手写节点，天然支持多张工单
g2.add_edge(START, "llm")
g2.add_conditional_edges("llm", route, {"tools": "tools", END: END})
g2.add_edge("tools", "llm")
app2 = g2.compile()

out = app2.invoke({"messages": [HumanMessage("帮我查 2024010101 和 2024099999 两条记录")]})
print("   工具节点产出的回执：")
for m in out["messages"]:
    if isinstance(m, ToolMessage):
        print(f"   ToolMessage tool_call_id={m.tool_call_id}  内容={m.content!r}")
print("""
   关键规矩：一张 AIMessage 开了 2 张工单，就要有 2 条 ToolMessage，
   而且 tool_call_id 必须和工单 id 一一对应 —— 少一条、配错号，LLM 都会懵。""")

# ==================================================================
sep("103：ToolNode 替代手动 —— 两个内置件替代 101 的全部手写")
# ==================================================================

g3 = StateGraph(State)
g3.add_node("llm", fake_llm_single)
g3.add_node("tools", ToolNode([query_leave_record]))   # ← 手写工具节点没了
g3.add_edge(START, "llm")
g3.add_conditional_edges("llm", tools_condition)     # ← 手写条件边也没了
g3.add_edge("tools", "llm")
app3 = g3.compile()

out = app3.invoke({"messages": [HumanMessage("查一下 2024010101")]})
print("   tools_condition 判定结果：", tools_condition({"messages": out["messages"][-3:]}))
print("   最后两条消息：")
for m in out["messages"][-2:]:
    print(f"   [{type(m).__name__}] {m.content!r}")
print("""
   对比 101：manual_tool_node 整个函数 + route 整个函数，换成这两行：
       ToolNode([你的工具们])         ← 帮你读工单、执行、写回执、处理报错
       tools_condition                ← 帮你判断"还有没有工单没处理"
   功能一模一样，还白送：参数校验（args 类型不对会报出友好错误）、
   工具抛异常时自动包装成 error 回执而不是让整张图崩掉。""")

# ==================================================================
sep("104：ToolRuntime —— 工具函数里直接拿到图的 state 和 tool_call_id")
# ==================================================================

@tool
def check_permission(student_id: str, runtime: ToolRuntime) -> str:
    """查询请假记录，同时校验当前用户是否有权查看。"""
    user = runtime.state.get("user_id", "未知用户")   # ← 从图 state 里拿当前用户
    if user != "inward":
        return f"拒绝：{user} 无权查看请假记录 {student_id}"
    leave_records = {"2024010101": "已批准", "2024010202": "待销假"}
    return f"{user} 查询请假记录 {student_id}（本次调用 id={runtime.tool_call_id[:8]}…）：{leave_records.get(student_id, '不存在')}"
#   ↑ 参数名叫 runtime、类型标 ToolRuntime，框架就自动注入，不需要 Annotated、
#     也不会把它当成要 LLM 填的参数 —— LLM 只负责填 student_id。


def fake_llm_for_runtime(state):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [AIMessage(content="好的，以上是该学号的请假记录。")]}
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "check_permission", "args": {"student_id": "2024010101"}, "id": "call_r1"},
    ])]}

class State2(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str                                   # 图的状态里带一个"当前用户"

g4 = StateGraph(State2)
g4.add_node("llm", fake_llm_for_runtime)
g4.add_node("tools", ToolNode([check_permission]))
g4.add_edge(START, "llm")
g4.add_conditional_edges("llm", tools_condition)
g4.add_edge("tools", "llm")
app4 = g4.compile()

out = app4.invoke({"messages": [HumanMessage("查查 2024010101")], "user_id": "inward"})
for m in out["messages"]:
    if isinstance(m, ToolMessage):
        print("   回执：", m.content)
print("""
   104 集的价值：以前工具只能拿到 LLM 填的参数；现在还能拿到
   图的 state（当前用户）、tool_call_id、store（长期记忆）、config。
   工具从"孤立的函数"变成了"能看见整个图的函数"。""")

# ==================================================================
sep("105：wrap_tool_call —— 工具执行的拦截器（演示：失败自动重试）")
# ==================================================================

FLAKY = {"calls": 0}

@tool
def flaky_api(city: str) -> str:
    """查询城市天气（这个接口不稳定，第一次调用会失败）。"""
    FLAKY["calls"] += 1
    if FLAKY["calls"] == 1:
        raise ConnectionError("网络抖动")          # 第一次故意失败
    return f"{city} 晴，28 度"


def retry_wrapper(request, execute):
    """拦截器：每次工具执行都要路过这里。
       request  = ToolCallRequest（tool_call / tool / state / runtime 四件套）
       execute  = 真正执行的动作，可以调用【多次】← 这就是重试的关键"""
    print(f"   [拦截器] 拦到调用：{request.tool_call['name']}({request.tool_call['args']})")
    try:
        result = execute(request)                  # 第一次：里面会炸
        return result
    except Exception as e:
        print(f"   [拦截器] 失败了（{e!r}）→ 自动重试一次")
        return execute(request)                    # 第二次：成功


def fake_llm_weather(state):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [AIMessage(content="广州今天晴，28 度。")]}
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "flaky_api", "args": {"city": "广州"}, "id": "call_f1"}])]}


g5 = StateGraph(State)
g5.add_node("llm", fake_llm_weather)
g5.add_node("tools", ToolNode([flaky_api],
                              wrap_tool_call=retry_wrapper,   # ← 拦截器装在这里
                              handle_tool_errors=False))      # ← 让异常抛出来，拦截器才有机会接
g5.add_edge(START, "llm")
g5.add_conditional_edges("llm", tools_condition)
g5.add_edge("tools", "llm")
app5 = g5.compile()

out = app5.invoke({"messages": [HumanMessage("广州天气如何")]})
print("   工具实际被调了", FLAKY["calls"], "次，最终回执：",
      [m.content for m in out["messages"] if isinstance(m, ToolMessage)])
print("""
   105 集的价值：重试、限流、加日志、改写参数、直接拒绝调用……
   全都不用改工具函数本身，写一个拦截器装在 ToolNode 上就行。
   这和之前学过的 RetryPolicy 是两层东西：RetryPolicy 管节点级重试，
   wrap_tool_call 管单次工具调用的进出 —— 你在这里能拿到 request 全貌。""")

sep("总结：5 集一条线")
print("""
    101/102  手写：读 tool_calls → 执行 → ToolMessage(tool_call_id=...)
    103      ToolNode + tools_condition 两行替代（还白送参数校验和错误包装）
    104      ToolRuntime：工具函数加 runtime 参数，拿到 state/tool_call_id/store
    105      wrap_tool_call：拦截器，request + execute(可多次)，做重试/日志/改写

    一句话：103 之前是"知道原理"，103 之后是"实际生产都这么写"。""")
