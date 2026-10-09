# -*- coding: utf-8 -*-
"""68-70 集：查看历史检查点 / 所有记录 / 单独一个检查点

连的就是你虚拟机里的真 PostgreSQL（66 集部署的那个库）。
跑一次 = 聊两轮 → 产生一串检查点 → 三种方式翻看它们。
每次运行用新的 thread_id，所以每次都是干净的一条时间线。
"""
import operator
import os
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 原来硬编码了我自己虚拟机的 IP 和默认密码。别人照抄连不上，
# 而且密码不该进公开仓库 —— 改成从环境变量读，见 langgraph_postgres_demo.py。
DB_URI = os.environ.get(
    "LANGGRAPH_DB_URI",
    "postgresql://postgres:postgres@localhost:5432/langgraph",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]   # 追加
    count: int                                # 覆盖


def step_a(state):
    return {"count": state["count"] + 1}


def step_b(state):
    return {"messages": [f"第 {state['count']} 步完成"]}


def build():
    g = StateGraph(State)
    g.add_node("a", step_a)
    g.add_node("b", step_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


with PostgresSaver.from_conn_string(DB_URI) as ckpt:
    ckpt.setup()
    app = build().compile(checkpointer=ckpt)

    thread = f"inspect-{int(time.time())}"
    cfg = {"configurable": {"thread_id": thread}}

    # 聊两轮：每轮 2 个节点 = 2 个 super-step = 2 个检查点，共 4 个
    app.invoke({"messages": ["开工"], "count": 0}, cfg)
    app.invoke({"messages": ["再来"], "count": 0}, cfg)

    print(f"本次 thread_id = {thread}\n")

    # ===== 68 + 69：get_state_history，把整条时间线从新到旧列出来 =====
    print("== 68/69：所有检查点（get_state_history，从新到旧）==")
    history = list(app.get_state_history(cfg))
    for snap in history:
        cid = snap.config["configurable"]["checkpoint_id"]
        msgs = snap.values.get("messages", [])
        last = msgs[-1].content if msgs else "(空)"
        print(f"  id={cid[:12]}  step={snap.metadata.get('step'):>2}  "
              f"下一步={list(snap.next) or ['(已到 END)']}  "
              f"count={snap.values.get('count')}  "
              f"共 {len(msgs)} 条，最后一条 = {last}")

    # ===== 70：get_state 不带 id = 最新一个；带 id = 指定那一个 =====
    print("\n== 70：查单独一个检查点 ==")
    latest = app.get_state(cfg)
    print("  不带 id（= 最新）: count =", latest.values["count"],
          " next =", list(latest.next))

    target = history[len(history) // 2]          # 故意挑中间的一个
    cid = target.config["configurable"]["checkpoint_id"]
    # langgraph 1.x 的写法：把 checkpoint_id 塞进 config（不是关键字参数）
    one = app.get_state({"configurable": {"thread_id": thread,
                                          "checkpoint_id": cid}})
    print(f"  指定 id {cid[:12]}: count =", one.values["count"],
          " next =", list(one.next))

    # ===== 印证：框架 API 的背后就是 SQL =====
    print("\n== 印证：直查数据库，看到的其实是同一批记录 ==")
    import psycopg
    conn = psycopg.connect(DB_URI)
    cur = conn.cursor()
    cur.execute(
        "select checkpoint_id, parent_checkpoint_id from checkpoints "
        "where thread_id = %s order by checkpoint_id",
        (thread,),
    )
    rows = cur.fetchall()
    print(f"  checkpoints 表里这个 thread 有 {len(rows)} 行：")
    for cid_, parent_ in rows:
        p = (parent_ or "None")[:12]
        print(f"    id={cid_[:12]}...  parent={p}...")
    cur.execute(
        "select channel, type from checkpoint_blobs where thread_id = %s limit 6",
        (thread,),
    )
    print("  checkpoint_blobs 表里存着每条 channel 的值：")
    for ch, t in cur.fetchall():
        print(f"    channel={ch:10s} type={t}")
    conn.close()

    print("\n== 结论 ==")
    print("  每个检查点记住『我是谁 + 我爹是谁』→ 串成一条链（67 集的持久化模式）")
    print("  get_state_history = 沿着链倒着走一遍；get_state(id) = 直接跳到链上某一点")
