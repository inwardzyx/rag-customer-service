# -*- coding: utf-8 -*-
"""
Step 1：把 fake_llm 换成【真模型】，其他部分和 101-103 集的 demo 几乎一样。

目的：亲眼看到"真模型自己决定调不调工具"——这是 fake_llm 永远给不了的东西。

工具本身仍然是"假"的（固定返回）——没关系，这一步的重点是【老板是真的】。

跑法（trace 先关掉，原因见文件底部"两个坑"）：
    set LANGSMITH_TRACING=false
    D:/Python-project/.venv/Scripts/python.exe step1_real_llm_graph.py
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============ ① 工人：和 101 集一模一样 ============
@tool
def get_order_status(order_id: str) -> str:
    """根据订单号查询物流状态。"""
    return f"订单 {order_id}：已发货，48 小时内到达。"


# ============ ② State：和以前一模一样 ============
class State(TypedDict):
    messages: Annotated[list, add_messages]   # 追加式（第 2 组讲过 reducer）


# ============ ③ 真 LLM：这是今天唯一的换血 ============
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
#    └ ChatDeepSeek 是一个【类】（大写 + 括号 = 照模具造对象）
#      它自动读环境变量 DEEPSEEK_API_KEY（你机器上已设好，不用传）
#      temperature=0 = 让模型回答尽量稳定，方便我们做实验

llm_with_tools = llm.bind_tools([get_order_status])
#    └ bind_tools 是 llm 身上的【方法】（有括号才执行）
#      作用：把"工具说明书"塞给模型，返回一个【新的】会开工单的 llm
#      注意：原 llm 没被改动 —— 这叫"返回新对象"，不是原地修改


def chat(state):
    # 真模型上场：读对话历史 + 工具说明书，自己决定"回话"还是"开工单"
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ============ ④ 搭图：和 103 集一模一样 ============
g = StateGraph(State)
g.add_node("chat", chat)                            # 老板
g.add_node("tools", ToolNode([get_order_status]))   # 工人（装筐，第 22 组）
g.add_edge(START, "chat")
g.add_conditional_edges("chat", tools_condition)    # 有工单 → tools；没有 → END
g.add_edge("tools", "chat")                         # 回执交回老板

app = g.compile()


# ============ ⑤ 实验：两条问题，一条不该调工具，一条该调 ============
QUESTIONS = [
    "你好，用一句话介绍你自己。",        # 不需要查订单 → 模型应该【不开工单】
    "帮我查一下订单 A1001 到哪了。",     # 需要查订单 → 模型应该【开工单】
]

for q in QUESTIONS:
    print("\n" + "=" * 62)
    print("用户：", q)
    out = app.invoke({"messages": [HumanMessage(content=q)]})
    for m in out["messages"]:
        kind = type(m).__name__          # __name__ = 这个消息的类名（AIMessage 等）
        if kind == "AIMessage":
            if m.tool_calls:
                print(f"  [老板] 开工单！tool_calls = {m.tool_calls}")
            else:
                print(f"  [老板] 直接回话：{m.content!r}")
        elif kind == "ToolMessage":
            print(f"  [工人] 回执（id={m.tool_call_id}）：{m.content!r}")
        else:
            print(f"  [用户] {m.content!r}")

    # token 用量：从最后一条 AI 消息的元数据里掏（上次让你留意的 usage 就是它）
    usage = out["messages"][-1].response_metadata.get("token_usage")
    if usage:
        print(f"  [账单] 输入 {usage.get('prompt_tokens')} tokens，"
              f"输出 {usage.get('completion_tokens')} tokens")

print("\n两个坑（跑之前必读）：")
print("  1. 跑之前先 set LANGSMITH_TRACING=false —— 你的机器上 trace 是永久开的，")
print("     真模型的每次调用都会往 LangSmith 上报。偶尔开一次没问题，")
print("     但练习时开着，网络一抖脚本就卡住。")
print("  2. 工人是假的（固定返回），所以答案永远是那句话 —— 这是对的，")
print("     真正干活的数据以后再换，先把『模型自己判断』这件事看清楚。")
