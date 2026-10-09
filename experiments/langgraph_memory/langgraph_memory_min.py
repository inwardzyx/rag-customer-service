# -*- coding: utf-8 -*-
"""最小记忆 demo：只讲一件事 —— checkpointer 就是『机器人的记性』

没有 count、没有数据库、没有表结构，只有 messages。
对比两段：不配 checkpointer（每轮失忆） vs 配了（能记住）。
"""
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]   # 唯一的一栏：对话记录


def chat(state):
    seen = [m.content for m in state["messages"]]
    print(f"    [节点睁眼] 我看到 {len(seen)} 条：{seen}")
    return {"messages": [f"回复：你一共说过 {len(seen)} 句"]}


def brief(msgs):                      # 只留文字，去掉消息对象的杂七杂八
    return [m.content for m in msgs]


g = StateGraph(State)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)

# ===== 第一段：不配 checkpointer —— 每次都从零开始 =====
print("== 没有存档（g.compile() 什么都不传）==")
forgetful = g.compile()
print("  第 1 轮：", brief(forgetful.invoke({"messages": ["你好"]})["messages"]))
print("  第 2 轮：", brief(forgetful.invoke({"messages": ["我叫小明"]})["messages"]))
print("  ↑ 第 2 轮只有 1 条 —— 上一轮的『你好』没了，它失忆了")

# ===== 第二段：配 checkpointer —— 记住了 =====
print("\n== 配了存档（compile(checkpointer=InMemorySaver())）==")
app = g.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "A"}}      # 存档槽位号
print("  第 1 轮：", brief(app.invoke({"messages": ["你好"]}, cfg)["messages"]))
print("  第 2 轮：", brief(app.invoke({"messages": ["我叫小明"]}, cfg)["messages"]))
print("  第 3 轮：", brief(app.invoke({"messages": ["我叫什么名字"]}, cfg)["messages"]))
print("  ↑ 第 3 轮它看到了 5 条 —— 前面的对话全在，这就是『记忆』")

print("\n== 一句话总结 ==")
print("  checkpointer 干的事 = 每跑完一步，把整张纸条复印一份存起来")
print("  下次用同一个 thread_id 进来，就先把上次那份复印纸摊开，接着往下写")
