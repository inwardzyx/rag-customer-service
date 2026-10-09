# -*- coding: utf-8 -*-
"""两个最小故事，对应 86 集和 87 集
故事一（86 集 HITL）：写邮件 → 停下来给人类看 → 批/驳走两条不同的路
故事二（87 集并行中断）：两个人同时举手 → 回复时要"对号入座"
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    log: Annotated[list, operator.add]


def show(title, lines):
    print(f"\n--- {title} ---")
    for x in lines:
        print("   ", x)


# ==================================================================
# 故事一：邮件审批流（86 集）
# 流程：写草稿 → 【停下来问人】→ 批准就发送 / 驳回就重写
# ==================================================================
def draft(state):
    return {"log": ["草稿：『尊敬的客户，您好……』"]}


def review(state):
    # 举手：把草稿拿给人类看，等他回 yes 或 no
    decision = interrupt(f"这封邮件要发出去吗？(yes/no)\n草稿：{state['log'][-1]}")
    # 人类回复后，把他的决定写在单子上（后面靠这个决定走向）
    return {"log": [f"人类批复：{decision}"]}


def route(state):
    # 路由函数（32 集学过）：只看单子，返回"下一步去哪个节点"
    last = state["log"][-1]                       # [-1] = 列表最后一项
    return "send" if last.endswith("yes") else "rewrite"   # 三元表达式


def send(state):
    return {"log": ["已发送"]}


def rewrite(state):
    return {"log": ["打回重写"]}


g1 = StateGraph(State)
for name, fn in [("draft", draft), ("review", review),
                 ("send", send), ("rewrite", rewrite)]:
    g1.add_node(name, fn)
g1.add_edge(START, "draft")
g1.add_edge("draft", "review")
g1.add_conditional_edges("review", route)     # 条件边：看人类批复决定去哪
g1.add_edge("send", END)
g1.add_edge("rewrite", END)
app1 = g1.compile(checkpointer=InMemorySaver())

show("故事一：人类批 yes", [])
cfg_a = {"configurable": {"thread_id": "邮件-001"}}
app1.invoke({"log": []}, cfg_a)                       # 停在 review
out = app1.invoke(Command(resume="yes"), cfg_a)       # 递回复
show("批 yes 的结果", out["log"])

cfg_b = {"configurable": {"thread_id": "邮件-002"}}   # 换个槽 = 另一封邮件
app1.invoke({"log": []}, cfg_b)
out = app1.invoke(Command(resume="no"), cfg_b)
show("批 no 的结果", out["log"])


# ==================================================================
# 故事二：两个人同时举手（87 集）
# 两条并行线：财务审批 + 法务审批，两个人同时卡住等回复
# 关键点：回复时必须写清楚"这条回复是给谁的" → 用中断 id 当编号
# ==================================================================
def finance(state):
    a = interrupt("财务：这笔钱能批吗？")
    return {"log": [f"财务说：{a}"]}


def legal(state):
    b = interrupt("法务：这份合同有问题吗？")
    return {"log": [f"法务说：{b}"]}


g2 = StateGraph(State)
g2.add_node("finance", finance)
g2.add_node("legal", legal)
g2.add_edge(START, "finance")     # 同一个 START 连两条边
g2.add_edge(START, "legal")       # → 两个节点并行跑（30 集学过）
g2.add_edge("finance", END)
g2.add_edge("legal", END)
app2 = g2.compile(checkpointer=InMemorySaver())
cfg2 = {"configurable": {"thread_id": "合同-001"}}

out = app2.invoke({"log": []}, cfg2)
pending = out["__interrupt__"]        # 挂起的中断，是个【列表】
show("两个人同时举手，挂起的问题有这些", [])
for p in pending:
    print(f"    编号(id)={p.id[:12]}  问题={p.value}")
print("    ↑ 注意：谁排第一不保证（谁先跑完谁在前），所以不能按位置猜是谁")

# 恢复：字典 —— 键是中断编号，值是给那个人的回复
# 认人的正确做法：看问题内容判断是谁，别按 pending[0] / pending[1] 的位置猜
answers = {}
for p in pending:
    if "财务" in p.value:
        answers[p.id] = "批"
    else:
        answers[p.id] = "没问题"
out = app2.invoke(Command(resume=answers), cfg2)
show("对号入座之后的结果", out["log"])

print()
print("=" * 60)
print("87 集的坑：同时有多个中断时，resume 不能只给一个值（框架不知道给谁），")
print("必须写成 {中断编号: 回复} 的字典，就像给每个举手的人单独回一句话。")
