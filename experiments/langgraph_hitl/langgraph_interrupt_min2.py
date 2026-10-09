# -*- coding: utf-8 -*-
"""中断最小故事 · 第二版：一个工厂，两种停法

故事A（动态）：质检工人【举手】问主管 → 主管要回答 → Command(resume=...)
故事B（静态）：质检车间门口【装锁】   → 不用回答   → invoke(None, cfg)

除了"停的方式"，两张图的代码一模一样。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    log: Annotated[list, operator.add]      # 追加，不是覆盖


def prepare(state):
    return {"log": ["备料完成"]}


def pack(state):
    return {"log": ["装箱发货"]}


# ============ 故事A：节点函数里写 interrupt() ============
def check_raise_hand(state):
    answer = interrupt("这批货合格吗？(yes/no)")   # 举手，程序冻在这一行
    return {"log": [f"主管说：{answer}"]}


gA = StateGraph(State)
gA.add_node("prepare", prepare)
gA.add_node("check", check_raise_hand)
gA.add_node("pack", pack)
gA.add_edge(START, "prepare")
gA.add_edge("prepare", "check")
gA.add_edge("check", "pack")
gA.add_edge("pack", END)
appA = gA.compile(checkpointer=InMemorySaver())   # 只配存档器，别的都不写

cfgA = {"configurable": {"thread_id": "A"}}       # 存档槽：恢复时要用同一个
print("【故事A】动态中断 = 工人举手")
out = appA.invoke({"log": []}, cfgA)
print("  1) 停下来时：", out["log"])
snap = appA.get_state(cfgA)
print("     next =", snap.next, "← 下一站还是 check（它没跑完）")
print("     举手内容 =", snap.tasks[0].interrupts[0].value)
out = appA.invoke(Command(resume="yes"), cfgA)    # 递回答进去
print("  2) 主管回答 yes 后：", out["log"])


# ============ 故事B：compile 时指定 interrupt_before ============
def check_normal(state):
    return {"log": ["质检完成（这个函数里没有 interrupt）"]}


gB = StateGraph(State)
gB.add_node("prepare", prepare)
gB.add_node("check", check_normal)                # 注意：普通节点
gB.add_node("pack", pack)
gB.add_edge(START, "prepare")
gB.add_edge("prepare", "check")
gB.add_edge("check", "pack")
gB.add_edge("pack", END)
appB = gB.compile(checkpointer=InMemorySaver(),
                  interrupt_before=["check"])     # 锁装在 check 门口

cfgB = {"configurable": {"thread_id": "B"}}
print()
print("【故事B】静态断点 = 门口装锁")
out = appB.invoke({"log": []}, cfgB)
print("  1) 停下来时：", out["log"], "← check 一行都没跑")
snap = appB.get_state(cfgB)
print("     next =", snap.next, "← 下一站是 check")
t = snap.tasks[0]                                 # tasks 里有一条"待办"
print("     tasks[0]：name =", t.name, "  interrupts =", t.interrupts)
print("     → 有一条待办，但 interrupts 是【空的】：没人举手，只是门口有锁")
out = appB.invoke(None, cfgB)                     # 推门，不递话
print("  2) 推门后：", out["log"])

print()
print("=" * 56)
print("怎么区分？看 tasks 里那条待办的 interrupts：")
print("  举手 → tasks[0].interrupts = [('问题', id…)]  非空，要回答")
print("  门锁 → tasks[0].interrupts = ()              空的，不用答")
print("两者都必须配 checkpointer，都用同一个 thread_id 才能续上")
print("=" * 56)
