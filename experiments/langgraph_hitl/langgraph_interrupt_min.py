# -*- coding: utf-8 -*-
"""中断最简版：只有一个故事 —— 请假审批
故事：员工提交请假条 → 程序拿到天数 → 超过 3 天就停下来问老板 → 老板批/驳 → 打印结果
（对照 langgraph_interrupt_demo.py 的完整版看）
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# ---------------------------------------------------------------
# 1. 单子（State）：只放一个字段 log，用来记"已经发生了什么"
#    Annotated[list, operator.add] = "新内容钉在后面，别擦掉旧的"
# ---------------------------------------------------------------
class State(TypedDict):
    log: Annotated[list, operator.add]


# ---------------------------------------------------------------
# 2. 节点一：员工提交请假条（正常干活，不打断）
#    节点函数的第一个参数 state = 当前单子
#    返回值 = 这次要往单子上加什么（增量，不是整张单子）
# ---------------------------------------------------------------
def submit(state):
    return {"log": ["员工：我要请假 5 天"]}


# ---------------------------------------------------------------
# 3. 节点二：审批（关键点：这里会"举手"）
#    interrupt("问题") 干的事：
#      第一次跑到这 → 整张图暂停，把"问题"发给人类，等回复
#      恢复后再跑到这 → 不再停，直接【返回】人类给的回复
# ---------------------------------------------------------------
def approve(state):
    days = 5                                    # 真实场景：从 state 里读天数
    if days <= 3:
        return {"log": [f"{days} 天：假期短，自动通过"]}

    # ↓↓↓ 这一行就是"举手"：程序停在这，把问题发给人类
    answer = interrupt(f"{days} 天假期超长，老板批吗？(yes/no)")

    # ↓↓↓ 人类回复之后，程序从这继续（注意：整个 approve 是从头重跑的，
    #     但上面那行第二次执行时不再停，而是把 answer 变成人类的回复）
    return {"log": [f"老板说：{answer}"]}


# ---------------------------------------------------------------
# 4. 路由函数：看完单子，决定下一步去哪（第 32 集学过）
#    返回的是【节点名字】，必须是 add_node 注册过的
# ---------------------------------------------------------------
def route(state):
    last = state["log"][-1]                     # [-1] = 列表最后一项
    return "done" if last.endswith("yes") else "reject"


def done(state):
    return {"log": ["结果：请假批准"]}


def reject(state):
    return {"log": ["结果：请假驳回"]}


# ---------------------------------------------------------------
# 5. 把图连起来（第 28 集学过）：注册节点 → 连边 → compile
# ---------------------------------------------------------------
g = StateGraph(State)
g.add_node("submit", submit)
g.add_node("approve", approve)
g.add_node("done", done)
g.add_node("reject", reject)

g.add_edge(START, "submit")
g.add_edge("submit", "approve")
g.add_conditional_edges("approve", route)       # 条件边：看单子决定去哪
g.add_edge("done", END)
g.add_edge("reject", END)

# checkpointer 必配：中断靠存档实现，没它图停不下来也恢复不了
app = g.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------
# 6. 跑起来
# ---------------------------------------------------------------
cfg = {"configurable": {"thread_id": "请假-001"}}    # 存档槽：这���请假单的编号

print("第一次 invoke（跑到举手那行就停）：")
out = app.invoke({"log": []}, cfg)
print("  单子 =", out["log"])
print("  挂起的问题 =", out["__interrupt__"][0].value)
print("  ↑ 程序现在停着，等人类。这时候关掉程序也没事，进度在存档里。")

print()
print("老板回复 yes：")
out = app.invoke(Command(resume="yes"), cfg)    # resume = 把人的回复递回去
print("  单子 =", out["log"])

print()
print("换一个存档槽，老板回复 no：")
cfg2 = {"configurable": {"thread_id": "请假-002"}}
app.invoke({"log": []}, cfg2)
out = app.invoke(Command(resume="no"), cfg2)
print("  单子 =", out["log"])
