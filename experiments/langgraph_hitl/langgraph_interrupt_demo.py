# -*- coding: utf-8 -*-
"""84-88 集：中断（human-in-the-loop）
84 两种中断机制        -> demo_1 / demo_2
85 主动中断的具体执行  -> demo_2（重点：节点从头重跑，interrupt() 第二次返回恢复值）
86 HITL 案例演示       -> demo_3（邮件审批流）
87 并行执行多个中断    -> demo_4（必须用 {中断id: 回复} 字典恢复）
88 审批模型            -> demo_5（大额才打断问人，小额自动过）
依赖：langgraph（已装）。不需要 LLM / 数据库。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    log: Annotated[list, operator.add]
    amount: int          # 88 集审批模型用：转账金额


def step(name, msg):
    """小工具：造一个节点函数，只往 log 写一句话。"""
    def fn(state):
        return {"log": [msg]}
    fn.__name__ = name
    return fn


print("#" * 64)
print("demo_1（84 集）：静态中断 —— compile(interrupt_before=[...])")
print("#" * 64)

g = StateGraph(State)
g.add_node("a", step("a", "a 干完"))
g.add_node("b", step("b", "b 干完"))
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)

app1 = g.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["b"],          # ← 写死：每次跑到 b 之前都停
)
cfg = {"configurable": {"thread_id": "d1"}}

out = app1.invoke({"log": []}, cfg)
print("第一次 invoke，log =", out["log"], "← a 跑了，b 没跑")
snap = app1.get_state(cfg)
print("停在谁前面：next =", list(snap.next), "← 静态停顿查这里")
out = app1.invoke(None, cfg)         # None = 继续
print("invoke(None) 续跑，log =", out["log"])


print()
print("#" * 64)
print("demo_2（84/85 集）：动态中断 —— 节点里调 interrupt()")
print("#" * 64)


def ask_person(state):
    print("  [节点] 进来了，开始干活")
    answer = interrupt("请审批：要继续吗？")   # ← 第一次跑到这：暂停
    # 第二次跑进来时，上面那行不再暂停，而是【返回】人给的回复
    print(f"  [节点] 拿到人类回复：{answer!r}（注意：这行第一次没执行）")
    return {"log": [f"根据回复 {answer!r} 继续干"]}


g2 = StateGraph(State)
g2.add_node("ask", ask_person)
g2.add_edge(START, "ask")
g2.add_edge("ask", END)
app2 = g2.compile(checkpointer=InMemorySaver())
cfg2 = {"configurable": {"thread_id": "d2"}}

out = app2.invoke({"log": []}, cfg2)
print("第一次 invoke，log =", out["log"])
pending = out["__interrupt__"][0]        # ← 挂起的中断信息在这个键里
print("挂起的问题：", pending.value)

out = app2.invoke(Command(resume="批准"), cfg2)   # ← 把人的回复递回去
print("第二次 invoke，log =", out["log"], "← __interrupt__ 消失了")
print("关键现象：[节点] 的两行 print 都出现了——")
print("  恢复时节点是【从头重跑】的，interrupt() 第二次执行变成直接返回恢复值")


print()
print("#" * 64)
print("demo_3（86 集）：HITL 案例 —— 邮件审批流（批/驳两条路）")
print("#" * 64)


def draft(state):
    return {"log": ["草稿写好了：『尊敬的客户……』"]}


def review(state):
    decision = interrupt(f"待审：{state['log'][-1]}  批准吗？(yes/no)")
    return {"log": [f"人类批复：{decision}"]}


def route(state):
    return "send" if state["log"][-1].endswith("yes") else "rewrite"


def send(state):
    return {"log": ["已发送"]}


def rewrite(state):
    return {"log": ["打回重写"]}


g3 = StateGraph(State)
for n, f in [("draft", draft), ("review", review),
             ("send", send), ("rewrite", rewrite)]:
    g3.add_node(n, f)
g3.add_edge(START, "draft")
g3.add_edge("draft", "review")
g3.add_conditional_edges("review", route)
g3.add_edge("send", END)
g3.add_edge("rewrite", END)
app3 = g3.compile(checkpointer=InMemorySaver())
cfg3 = {"configurable": {"thread_id": "d3"}}

app3.invoke({"log": []}, cfg3)                       # 停在 review
out = app3.invoke(Command(resume="yes"), cfg3)       # 人批了 yes
print("批准路径：", out["log"])

cfg3b = {"configurable": {"thread_id": "d3-no"}}     # 新槽位再跑一遍
app3.invoke({"log": []}, cfg3b)
out = app3.invoke(Command(resume="no"), cfg3b)       # 人驳了 no
print("驳回路径：", out["log"])


print()
print("#" * 64)
print("demo_4（87 集）：并行节点同时中断 —— 恢复必须带中断 id")
print("#" * 64)


def w1(state):
    a = interrupt("问题1：批准 A？")
    return {"log": [f"w1 收到 {a}"]}


def w2(state):
    b = interrupt("问题2：批准 B？")
    return {"log": [f"w2 收到 {b}"]}


g4 = StateGraph(State)
g4.add_node("w1", w1)
g4.add_node("w2", w2)
g4.add_edge(START, "w1")
g4.add_edge(START, "w2")
g4.add_edge("w1", END)
g4.add_edge("w2", END)
app4 = g4.compile(checkpointer=InMemorySaver())
cfg4 = {"configurable": {"thread_id": "d4"}}

out = app4.invoke({"log": []}, cfg4)
pending = out["__interrupt__"]
print("同时挂起的中断数：", len(pending))
for p in pending:
    print(f"  id={p.id[:16]}  问题={p.value}")

out = app4.invoke(Command(resume={pending[0].id: "批A",
                                  pending[1].id: "批B"}), cfg4)
print("恢复结果：", out["log"])
print("注意：多个中断时 resume 必须是 {中断id: 回复} 字典，")
print("     直接传列表会报 RuntimeError（已实测）。")


print()
print("#" * 64)
print("demo_5（88 集）：审批模型 —— 大额才打断，小额自动过")
print("#" * 64)


def approve(state):
    amount = state["amount"]              # 金额从输入里读（运行时才知道）
    if amount < 10000:
        return {"log": [f"{amount} 元：小额，自动通过"]}
    decision = interrupt(f"{amount} 元大额转账，请审批 (yes/no)")
    verdict = "通过" if decision == "yes" else "拒绝"
    return {"log": [f"{amount} 元：人类{verdict}"]}


g5 = StateGraph(State)
g5.add_node("approve", approve)
g5.add_edge(START, "approve")
g5.add_edge("approve", END)
app5 = g5.compile(checkpointer=InMemorySaver())

print("小额（8000）自动通过：")
out = app5.invoke({"log": [], "amount": 8000},
                  {"configurable": {"thread_id": "d5s"}})
print("  ", out["log"], "← 没触发中断，一次跑完")

print("大额（80000）停下问人：")
cfg5 = {"configurable": {"thread_id": "d5b"}}
out = app5.invoke({"log": [], "amount": 80000}, cfg5)
print("  log =", out["log"], " 挂起问题 =", out["__interrupt__"][0].value)
out = app5.invoke(Command(resume="yes"), cfg5)
print("  人批 yes 后：", out["log"])

print()
print("=" * 64)
print("两句话总结：interrupt() = 节点内动态停（停几次、停不停由数据定）；")
print("interrupt_before = 流程上静态关卡（写死某节点前必停）。都要配 checkpointer。")
