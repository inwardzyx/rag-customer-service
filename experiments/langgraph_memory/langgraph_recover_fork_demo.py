# -*- coding: utf-8 -*-
"""
71-75 集：持久化的实战收尾 —— 崩溃恢复 / 断点续跑 / fork 分叉
连虚拟机里的真 PostgreSQL 跑。

游戏存档类比：
  71-73：打 Boss 死了 -> 存档还在 -> 修好装备 -> 从存档原地复活（不用重头打）
  74  ：回档到中间某关重打一遍（时间旅行）
  75  ：回档重打，但旧时间线不删 —— 两条线并存（fork，像 git 分叉）
"""

import os
import time
import psycopg
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 原来硬编码了我自己虚拟机的 IP 和默认密码。别人照抄连不上，
# 而且密码不该进公开仓库 —— 改成从环境变量读，见 langgraph_postgres_demo.py。
DB_URI = os.environ.get(
    "LANGGRAPH_DB_URI",
    "postgresql://postgres:postgres@localhost:5432/langgraph",
)
FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bug_fixed.flag")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def prepare(state):
    print("    [prepare] 干活，写一笔")
    return {"messages": ["prepare 完成"]}


def boom(state):
    # 模拟“有 bug 的节点”：修好（标记文件存在）之前必崩
    if not os.path.exists(FLAG):
        raise RuntimeError("boom：发现 bug，程序崩溃！（72 集的异常现场）")
    print("    [boom] bug 已修，正常干活")
    return {"messages": ["boom 完成"]}


def finish(state):
    print("    [finish] 最后一道工序")
    return {"messages": ["finish 完成"]}


def build(checkpointer):
    g = StateGraph(State)
    g.add_node("prepare", prepare)
    g.add_node("boom", boom)
    g.add_node("finish", finish)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "boom")
    g.add_edge("boom", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def fmt(msgs):
    """把消息对象列表压成 ['开始', 'prepare 完成', ...] 短格式"""
    return [getattr(m, "content", str(m)) for m in msgs]


def load_chain(tid):
    conn = psycopg.connect(DB_URI)
    cur = conn.cursor()
    cur.execute(
        "select checkpoint_id, parent_checkpoint_id, metadata "
        "from checkpoints where thread_id=%s",
        (tid,),
    )
    rows = cur.fetchall()
    conn.close()
    return {r[0]: r for r in rows}, len(rows)


def walk_back(by_id, cid, label):
    chain = []
    while cid and cid in by_id:
        r = by_id[cid]
        chain.append(f"{cid[-6:]}(step {(r[2] or {}).get('step')})")
        cid = r[1]
    print(f"    {label}: " + " <- ".join(chain))


if os.path.exists(FLAG):
    os.remove(FLAG)  # 每次运行都从“有 bug”状态开始

with PostgresSaver.from_conn_string(DB_URI) as ckpt:
    ckpt.setup()
    tid = "recover-" + str(int(time.time()))
    cfg = {"configurable": {"thread_id": tid}}

    # ---------- 71 + 72：程序中途崩溃，看存档停在哪 ----------
    print("=" * 62)
    print("【71+72】第一次运行：跑到 boom 节点崩溃")
    app = build(ckpt)
    try:
        app.invoke({"messages": ["开始"]}, cfg)
    except RuntimeError as e:
        print(f"  崩了！{e}")
    latest = app.get_state(cfg)
    print(f"  但存档还在：最新快照 step {latest.metadata.get('step')}，"
          f"下一步要跑 {list(latest.next)}")
    print(f"  已完成的内容：{fmt(latest.values['messages'])}")
    print("  → 崩掉的是‘这一步’，崩不掉‘之前所有步’")

    # ---------- 73：修好 bug，从存档原地复活 ----------
    print("=" * 62)
    print("【73】工程师修好 bug（造标记文件），用 invoke(None, cfg) 续跑")
    open(FLAG, "w").close()
    print("  None 的意思：不传新输入，从上次存档接着来")
    result = app.invoke(None, cfg)
    print(f"  续跑完成：{fmt(result['messages'])}")
    print("  → prepare 没有重跑（输出里它只出现一次）—— 这就是断点续跑")
    old_end = app.get_state(cfg)  # 记下旧线终点，防止被后面的分叉覆盖

    # ---------- 74：回档到中间某一步，重新运行 ----------
    print("=" * 62)
    print("【74】时间旅行：回档到 step 1（boom 刚跑完那一刻）重新运行")
    history = list(app.get_state_history(cfg))
    target = next(s for s in history if s.metadata.get("step") == 1)
    cid = target.config["configurable"]["checkpoint_id"]
    print(f"  挑中的快照 …{cid[-6:]}，状态：{fmt(target.values['messages'])}")
    fork_cfg = {"configurable": {"thread_id": tid, "checkpoint_id": cid}}
    app.invoke(None, fork_cfg)
    new_end = app.get_state(cfg)  # 回档重跑后的最新快照 = 新线终点
    print(f"  从那一点重跑，结果：{fmt(new_end.values['messages'])}")

    # ---------- 75：fork —— 旧时间线还在，新线分叉出去 ----------
    print("=" * 62)
    print("【75】fork：直查真库，看两条时间线怎么并存")
    by_id, total = load_chain(tid)
    print(f"    （库里这个 thread 共 {total} 份快照）")
    walk_back(by_id, old_end.config["configurable"]["checkpoint_id"], "旧线")
    walk_back(by_id, new_end.config["configurable"]["checkpoint_id"], "新线")
    print("  → 两条线共享分叉点之前的部分，之后各走各的 —— git 分支既视感")
    print("=" * 62)
    print(f"本次 thread_id：{tid}（想复查可以直接查库）")
