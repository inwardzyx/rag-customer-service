# -*- coding: utf-8 -*-
"""
60-61 集实战：把 LangGraph 的存档放进 PostgreSQL
============================================================
我自己的环境：数据库跑在 VirtualBox 的 Ubuntu 虚拟机里（Docker 容器 langgraph-pg），
这份代码在 Windows 宿主机上运行，通过网络连过去。

★ 但你不用虚拟机也能跑 —— 只要有一个 PostgreSQL：
    set LANGGRAPH_DB_URI=postgresql://用户名:密码@主机:5432/库名
不设这个环境变量时，默认值是 localhost（对应本机装的 PostgreSQL）。

和 59 集 InMemorySaver 的差别只有【一行】：
    compile(checkpointer=PostgresSaver...)
换来的能力：程序重启、甚至换一台机器，存档都还在。
"""

import operator
import os
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

# 原来这里硬编码着我自己虚拟机的 IP（192.168.56.101）和 postgres/postgres 密码。
# 那是VirtualBox 的 host-only 网段 + 默认密码，别人照抄一定连不上，
# 而且把密码写进公开仓库是不该有的习惯。改成从环境变量读：
#   set LANGGRAPH_DB_URI=postgresql://user:pass@host:5432/db
DB_URI = os.environ.get(
    "LANGGRAPH_DB_URI",
    "postgresql://postgres:postgres@localhost:5432/langgraph",
)


class State(TypedDict):
    messages: Annotated[list, operator.add]


def chat(state):
    n = len(state["messages"])
    return {"messages": [f"回复（轮到我说话时，记录已有 {n} 条）"]}


def build_graph():
    g = StateGraph(State)
    g.add_node("chat", chat)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    return g


cfg = {"configurable": {"thread_id": "pg-A"}}

# ============ 第一段：像正常程序一样跑，聊两句 ============
print("== 连上数据库，聊两句 ==")
with PostgresSaver.from_conn_string(DB_URI) as ckpt:
    ckpt.setup()                                  # 第一次跑会自动建表（幂等）
    app = build_graph().compile(checkpointer=ckpt)
    print(app.invoke({"messages": ["你好"]}, cfg)["messages"])
    print(app.invoke({"messages": ["再来一句"]}, cfg)["messages"])

# ============ 第二段：假装程序重启，重新连库直接读存档 ============
print("\n== 关掉连接，重新连（模拟程序重启），直接读存档 ==")
with PostgresSaver.from_conn_string(DB_URI) as ckpt:
    app = build_graph().compile(checkpointer=ckpt)
    snap = app.get_state(cfg)
    print("从数据库读回：", snap.values["messages"])

print("\n== 结论 ==")
print("存档真的落在数据库里了，不在内存里 —— 这就是 60 集说的『数据库持久化』")
