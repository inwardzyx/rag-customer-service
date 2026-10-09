# -*- coding: utf-8 -*-
"""
57-61 集：持久化（checkpoint）—— 给图装「游戏存档」
============================================================
一句话总纲：
  compile 时给一个 checkpointer（存档系统）
  invoke 时给一个 thread_id（存档槽位号）
  → LangGraph 每走一步自动存档，下次同槽位接着跑

对应集数：
  57  机制：每一步结束都把整个 State 存一份快照
  58  启用：compile(checkpointer=...) + invoke 时带 thread_id
  59  InMemorySaver：存档放内存，程序一关就没
  60  内存 vs 数据库：数据库能跨程序重启恢复
  61  PostgreSQL：把存档放进数据库（本文件不装，只告诉你怎么接）

语法点：config 是个「字典套字典」——
  {"configurable": {"thread_id": "A"}}
   外层键固定叫 configurable，里面放运行时参数。
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    messages: Annotated[list, operator.add]     # reducer：追加，聊天记录不丢


def chat(state):
    n = len(state["messages"])
    # n 是「轮到我说话那一刻」存档里已有的条数（你的新消息这时已经进存档了）
    return {"messages": [f"回复（轮到我说话时，记录已有 {n} 条）"]}


g = StateGraph(State)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)

# =========================================================
# 58 + 59：启用持久化 —— 就多一个参数
# =========================================================
app = g.compile(checkpointer=InMemorySaver())       # ← 存档系统：内存版

cfg_a = {"configurable": {"thread_id": "A"}}        # 存档槽位 A
cfg_b = {"configurable": {"thread_id": "B"}}        # 存档槽位 B

print("== 槽位 A 连聊三次，看记录怎么累积 ==")
print(app.invoke({"messages": ["你好"]}, cfg_a)["messages"])
print(app.invoke({"messages": ["再来一句"]}, cfg_a)["messages"])
print(app.invoke({"messages": ["最后一句"]}, cfg_a)["messages"])

print("\n== 槽位 B 是全新存档，从头开始 ==")
print(app.invoke({"messages": ["你好"]}, cfg_b)["messages"])

print("\n== 槽位 A 再来一次：还记得之前聊的（多轮记忆就是它） ==")
print(app.invoke({"messages": ["还记得我吗"]}, cfg_a)["messages"])

# =========================================================
# 57 的机制：随时翻开某个槽位的存档
# =========================================================
snapshot = app.get_state(cfg_a)
print("\n== get_state：翻开槽位 A 的存档 ==")
print("当前 State：", snapshot.values)
print("跑到第几步：", (snapshot.metadata or {}).get("step", "?"))

print("\n== 结论 ==")
print("同一个 thread_id → 接着历史跑（多轮对话记忆的实现原理）")
print("不同 thread_id   → 各存各的（一个服务同时接无数个用户）")
print("InMemorySaver 在内存里，程序一关全没 —— 想真正保存见 60、61 集")
