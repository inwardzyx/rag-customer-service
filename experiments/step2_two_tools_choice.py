# -*- coding: utf-8 -*-
"""
Step 2：拆开 tools_condition 这个黑盒 + 给模型【两个工具】让它自己选。

跑法：
    set LANGSMITH_TRACING=false
    python step2_two_tools_choice.py
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ==================== ① 两个工人（工具） ====================
@tool
def get_order_status(order_id: str) -> str:
    """根据订单号查询物流状态。用户问订单到哪了、发没发货时用这个。"""
    return f"订单 {order_id}：已发货，48 小时内到达。"


@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。用户问天气、下雨、要不要带伞时用这个。"""
    return f"{city}：晴，26 度。"


TOOLS = [get_order_status, get_weather]      # 装筐（第 22 组）


# ==================== ② State ====================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ==================== ③ 真模型，一次给它两本说明书 ====================
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
#                              ↑ 两个工具一起递给它，由它自己选


def chat(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ==================== ④ 手写条件边：拆开 tools_condition ====================
def my_tools_condition(state):
    """这就是 tools_condition 的真身（我扒了 langgraph 1.2.11 的源码）"""
    last = state["messages"][-1]
    #    └ 取【最后一条】消息：模型刚说完的话
    has_calls = hasattr(last, "tool_calls") and len(last.tool_calls) > 0
    #           └ hasattr(对象, "属性名")：问"你身上有这个格子吗？"（第 15 组学过点号取属性，
    #             这里是它的安全检查版：先问有没有，再取，避免 AttributeError）
    print(f"      [手写条件边] 最后一条 = {type(last).__name__}，"
          f"带工单吗 = {has_calls}")
    return "tools" if has_calls else END
    #      ↑ 三元表达式（第 3 组）：有工单就去 tools，没有就收工


# ==================== ⑤ 搭图 ====================
g = StateGraph(State)
g.add_node("chat", chat)
g.add_node("tools", ToolNode(TOOLS))
g.add_edge(START, "chat")
g.add_conditional_edges("chat", my_tools_condition)   # 用手写版，方便看它在想什么
g.add_edge("tools", "chat")

app = g.compile()


# ==================== ⑥ 记账：统计一共烧了多少 token ====================
BILL = {"prompt": 0, "completion": 0}


def run(question):
    print("\n" + "-" * 62)
    print("用户：", question)
    out = app.invoke({"messages": [HumanMessage(content=question)]})
    for m in out["messages"]:
        kind = type(m).__name__
        if kind == "AIMessage":
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"   → 老板开工单：{tc['name']}({tc['args']})")
            else:
                print(f"   → 老板回答：{m.content!r}")
        elif kind == "ToolMessage":
            print(f"   ← 工人回执：{m.content!r}")
    usage = out["messages"][-1].response_metadata.get("token_usage")
    if usage:
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        BILL["prompt"] += p
        BILL["completion"] += c
        print(f"   [本次] 输入 {p} / 输出 {c} tokens")
    return out


# ==================== 实验 1：不该调工具时，它调不调 ====================
print("=" * 62)
print("实验 1：闲聊 —— 手写条件边应该返回 END")
run("你好，用一句话介绍你自己。")


# ==================== 实验 2：两个工具，它选哪个 ====================
print("\n" + "=" * 62)
print("实验 2：同一张图、两个工具，模型自己选")
run("帮我查一下订单 A1001 到哪了。")
run("广州今天天气怎么样？")


# ==================== 实验 3：一句话要两个工具（并行） ====================
print("\n" + "=" * 62)
print("实验 3：一句话里两件事 —— 看它能不能一次开两张工单")
run("帮我查订单 A1001，顺便告诉我广州天气。")


# ==================== 实验 4：参数缺失时它会怎么办（真实可靠性问题） ====================
print("\n" + "=" * 62)
print("实验 4：用户没给订单号 —— 模型会开口问，还是自己编一个？")
run("帮我查一下我上次买的那个订单到哪了。")


print("\n" + "=" * 62)
print(f"总账单：输入 {BILL['prompt']} tokens，输出 {BILL['completion']} tokens")
print("=" * 62)
