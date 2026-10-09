# -*- coding: utf-8 -*-
"""67 集：快照是怎么串成链的 —— 顺着 parent 从最新一路走回开头

做法：连真库 → 跑两轮 → 直查 checkpoints 表 → 手动沿 parent 走一遍。
走完你就会发现：get_state_history 干的事，就是这十几行循环。
"""
import os
import time
from typing import Annotated, TypedDict

import psycopg
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
    messages: Annotated[list, add_messages]


def chat(state):
    return {"messages": ["回复"]}


g = StateGraph(State)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)

with PostgresSaver.from_conn_string(DB_URI) as ckpt:
    ckpt.setup()
    app = g.compile(checkpointer=ckpt)
    tid = "chain-" + str(int(time.time()))
    cfg = {"configurable": {"thread_id": tid}}
    app.invoke({"messages": ["你好"]}, cfg)
    app.invoke({"messages": ["我叫小明"]}, cfg)
    latest = app.get_state(cfg).config["configurable"]["checkpoint_id"]

conn = psycopg.connect(DB_URI)
cur = conn.cursor()
cur.execute(
    "select checkpoint_id, parent_checkpoint_id, metadata "
    "from checkpoints where thread_id = %s",
    (tid,),
)
rows = cur.fetchall()
by_id = {r[0]: r for r in rows}

print(f"两轮 invoke 之后，库里这个 thread 有 {len(rows)} 行记录\n")
print("从最新那份出发，顺着 parent 往回走：")


def tail(x):
    """UUID 前缀都一样，看尾号才分得清"""
    return x[-6:] if x else "None"


cid = latest
i = 0
while cid:
    r = by_id.get(cid)
    if not r:
        break
    step = (r[2] or {}).get("step")
    print(f"  往回第 {i} 步: id=…{tail(cid)}  step={step:>2}  "
          f"→  它的 parent 是 …{tail(r[1])}")
    cid = r[1]
    i += 1

print("  （最后那个 parent 是 None，说明走到链头了）")
print("\n再看看：有没有哪份记录『知道自己儿子是谁』？")
sons = {}
for cid_, parent_, _ in rows:
    sons.setdefault(parent_, []).append(cid_)
print("  parent 字段只能往回指；想找『下一份』得全表扫、反着查 ——")
print("  这就是为什么 get_state_history 只能从新到旧，不能从旧到新")

conn.close()
