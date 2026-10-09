# -*- coding: utf-8 -*-
"""90-97 集：中断进阶
90  使用规范        ：必须配 checkpointer；恢复时节点从头重跑
91  检查点信息      ：get_state() 看 .next 和 .tasks
92  并行中断检查点  ：每个并行 task 各带各的 interrupts
93  部分任务触发中断：同一超步里没举手的节点——跑完、存档、不重跑
94  静态中断介绍    ：compile(interrupt_before=[...])，不改节点代码
95  静态断点基础    ：invoke 停在门口 → invoke(None) 推门进去
96  静态中断在超步  ：锁住超步里任何一个节点 = 整个超步停在门外
97  执行设置断点    ：给每个节点都装 interrupt_before，单步调试
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    log: Annotated[list, operator.add]


def make_app(nodes, edges, before=None, after=None):
    """小工具：搭图 + compile。
    before / after 就是静态断点名单（94 集）：
      before = 在这节点【跑之前】停（门外侧的锁）
      after  = 在这节点【跑完之后】停（门内侧的锁）
    """
    g = StateGraph(State)
    for name, fn in nodes:
        g.add_node(name, fn)
    for a, b in edges:
        g.add_edge(a, b)
    return g.compile(checkpointer=InMemorySaver(),
                     interrupt_before=before, interrupt_after=after)


def sep(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def show_tasks(snap):
    """91 集的主角：检查点快照里到底记了什么"""
    print("   next（下一步要跑的节点）:", snap.next or "（没有了）")
    if not snap.tasks:
        print("   tasks: 空 —— 没有待办（图已经跑到底了）")
    for t in snap.tasks:
        if t.interrupts:                       # 这条待办里有中断在等回复
            print(f"   task {t.name}: 【举手等回复】→ {t.interrupts[0].value!r}")
        else:
            print(f"   task {t.name}: 待办，没人举手（interrupts 是空的）")


# ==================================================================
sep("90 + 91：使用规范 + 中断那一刻检查点里记了什么")
# ==================================================================
print("规范三条：")
print("  1. 必须配 checkpointer —— 不然'举到一半的手'没地方存")
print("  2. interrupt 尽量写在节点开头 —— 恢复时整个节点会从头重跑")
print("  3. 恢复：动态中断用 Command(resume=...)；静态断点用 invoke(None, cfg)")

count = {"approve": 0}          # 数一数 approve 函数一共被调用几次


def draft(state):
    return {"log": ["写好草稿"]}


def approve(state):
    count["approve"] += 1       # 每次进这个函数就 +1
    ans = interrupt("这封邮件能发吗？(yes/no)")
    return {"log": [f"批复：{ans}"]}


def send(state):
    return {"log": ["已发送"]}


app1 = make_app([("draft", draft), ("approve", approve), ("send", send)],
                [(START, "draft"), ("draft", "approve"),
                 ("approve", "send"), ("send", END)])
cfg1 = {"configurable": {"thread_id": "90"}}

out = app1.invoke({"log": []}, cfg1)
print("\n第一次 invoke 停在 approve，检查点里记着：")
show_tasks(app1.get_state(cfg1))
print("   out['log'] =", out["log"], "← approve 的结果不在（它还没跑完）")

app1.invoke(Command(resume="yes"), cfg1)
print("\n恢复之后：")
print("   approve 函数被调用了", count["approve"], "次 ← 节点被【从头重跑】了一遍")
print("   所以 interrupt 前面的代码会执行两次，副作用（发请求/写库）要小心")
print("   最终 log：", app1.get_state(cfg1).values["log"])


# ==================================================================
sep("91 加餐：StateSnapshot 每个字段都是啥（中断那一刻的解剖图）")
# ==================================================================
def prepare(state):
    return {"log": ["资料准备完"]}


def check(state):
    ans = interrupt("确认要提交吗？(yes/no)")
    return {"log": [f"确认：{ans}"]}


def finish(state):
    return {"log": ["提交完成"]}


app0 = make_app([("prepare", prepare), ("check", check), ("finish", finish)],
                [(START, "prepare"), ("prepare", "check"),
                 ("check", "finish"), ("finish", END)])
cfg0 = {"configurable": {"thread_id": "91"}}

app0.invoke({"log": []}, cfg0)          # 停在 check 里
snap = app0.get_state(cfg0)             # 91 集的主角：快照
print("values（已经存进去的状态）      :", snap.values)
print("next（下一站是谁，是个元组）    :", snap.next)
print("config（这份快照自己的地址）    :", snap.config["configurable"])
print("parent_config（它是从哪份接的） :",
      (snap.parent_config or {}).get("configurable"))
print("metadata（第几步 / 谁写的）     :",
      {k: v for k, v in snap.metadata.items() if k in ("step", "writes")})
print("created_at / id                 :", snap.created_at is not None,
      "|", snap.config["configurable"]["checkpoint_id"][:8], "…")
print()
print("tasks（这一步的活儿干到哪了）:")
for t in snap.tasks:
    print("  - name      :", t.name)
    print("    id        :", t.id[:8], "…")
    print("    interrupts:", [(i.value, i.id[:8] + "…") for i in t.interrupts])
    print("    error     :", t.error)       # 没崩就是 None
print()
print("读法口诀：next 说'下一站'，tasks 说'这一站干到哪'，")
print("         interrupts【非空】= 有人举手 → 要 Command(resume=...)")
print("         interrupts【为空】= 只是待办/门锁 → invoke(None) 推门就行")


# ==================================================================
sep("92 + 93：并行时只举起一只手 —— 部分任务中断")
# ==================================================================
runs = {"finance": 0, "legal": 0}


def finance(state):
    runs["finance"] += 1
    return {"log": ["财务：直接通过，不举手"]}


def legal(state):
    runs["legal"] += 1
    ans = interrupt("法务：这份合同有问题吗？")
    return {"log": [f"法务说：{ans}"]}


app2 = make_app([("finance", finance), ("legal", legal)],
                [(START, "finance"), (START, "legal"),
                 ("finance", END), ("legal", END)])
cfg2 = {"configurable": {"thread_id": "93"}}

out = app2.invoke({"log": []}, cfg2)
print("同一个超步：finance 跑完了，legal 举手了。检查点里：")
show_tasks(app2.get_state(cfg2))
print("   out['log'] =", out["log"])
print("   跑过的次数：", runs)

app2.invoke(Command(resume="没问题"), cfg2)
print("\n恢复之后跑过的次数：", runs)
print("   ↑ finance 是 1 不是 2 —— 同一超步里【没举手的节点跑完就存档，不重跑】")
print("   92 集说的'并行中断的检查点'：每个 task 的 interrupts 各记各的，")


# ==================================================================
sep("94 + 95：静态断点 —— compile 时说好在哪停，节点代码不用改")
# ==================================================================
def s1(state):
    return {"log": ["① 资料准备完"]}


def s2(state):
    return {"log": ["② 人工检查环节（节点里一个 interrupt 都没写）"]}


def s3(state):
    return {"log": ["③ 完成"]}


app3 = make_app([("s1", s1), ("s2", s2), ("s3", s3)],
                [(START, "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", END)],
                before=["s2"])            # ← 门锁装在 s2 门口
cfg3 = {"configurable": {"thread_id": "95"}}

out = app3.invoke({"log": []}, cfg3)
print("invoke 自己停了！但 s2 一行代码都没执行：")
print("   out['log'] =", out["log"])
snap = app3.get_state(cfg3)
print("   next =", snap.next, "   tasks 里有没有人举手：", 
      any(t.interrupts for t in snap.tasks))
print("   tasks 里有一条待办，但 interrupts 是空的 —— 不是人举手，是门口的锁")
print("   （修正：之前讲'静态断点 tasks 为空'是错的 —— tasks 非空，空的是 interrupts）")

app3.invoke(None, cfg3)          # 不递话，None = 推门继续跑
print("invoke(None) 推门后：", app3.get_state(cfg3).values["log"])
print("   注意：恢复静态断点传 None，不是 Command —— 因为没有人要回复")


# ==================================================================
sep("94 加餐：interrupt_after —— 跑完之后才停（门内侧的锁）")
# ==================================================================
def m1(state):
    return {"log": ["m1 算出来的结果"]}


def m2(state):
    return {"log": ["m2 用完 m1 的结果"]}


app6 = make_app([("m1", m1), ("m2", m2)],
                [(START, "m1"), ("m1", "m2"), ("m2", END)],
                after=["m1"])            # ← 跑完 m1 就停
cfg6 = {"configurable": {"thread_id": "94b"}}

out = app6.invoke({"log": []}, cfg6)
print("m1 已经跑完，但结果还没往下传：")
print("   out['log'] =", out["log"], "   next =", app6.get_state(cfg6).next)
print("   典型用途：让人类【先看到这一步的产出】再决定要不要继续")
app6.invoke(None, cfg6)
print("推门后：", app6.get_state(cfg6).values["log"])


# ==================================================================
sep("96：静态断点卡在超步门口 —— 没轮到的节点根本没开始")
# ==================================================================
def pa(state):
    return {"log": ["a 跑完了"]}


def pb(state):
    return {"log": ["b 跑完了"]}


app4 = make_app([("a", pa), ("b", pb)],
                [(START, "a"), (START, "b"), ("a", END), ("b", END)],
                before=["b"])
cfg4 = {"configurable": {"thread_id": "96"}}

out = app4.invoke({"log": []}, cfg4)
print("同一超步：a 和 b 本该一起跑，但 b 门口有锁 →")
print("   【整个超步都停在门外】：out['log'] =", out["log"])
print("   next =", app4.get_state(cfg4).next, "← 连 a 也还在 next 里，谁都没跑")
app4.invoke(None, cfg4)
print("推门后：", app4.get_state(cfg4).values["log"])
print("   这就是 96 集标题的意思：静态中断只能卡在【超步边界】，")
print("   粒度是整个超步不是单个节点 —— 一个节点有锁，全队一起等门")


# ==================================================================
sep("97：执行设置断点 —— 每个节点门口都装锁，单步调试")
# ==================================================================
def n1(state):
    return {"log": ["第 1 步"]}


def n2(state):
    return {"log": ["第 2 步"]}


def n3(state):
    return {"log": ["第 3 步"]}


app5 = make_app([("n1", n1), ("n2", n2), ("n3", n3)],
                [(START, "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", END)],
                before=["n1", "n2", "n3"])   # 每个节点门口都装锁
print("（注：老文档说可以写 '*' 代表全部节点，但 langgraph 1.2.11")
print("  实测不认，会报 Interrupt node `*` not found —— 老老实实列名字）")
cfg5 = {"configurable": {"thread_id": "97"}}

app5.invoke({"log": []}, cfg5)
while True:                               # 一直推门，直到没有下一站
    snap = app5.get_state(cfg5)
    if not snap.next:
        break
    print("   停在：", snap.next, " 已完成：", snap.values.get("log", []))
    app5.invoke(None, cfg5)               # 每次推一扇门，跑一个节点
print("   全部跑完：", app5.get_state(cfg5).values["log"])
print("   这就是'用执行来设置断点'：不改任何代码，把整张图变成单步模式")

print()
print("=" * 62)
print("总结：动态 interrupt = 节点自己举手（要回复）；静态 interrupt_before")
print("     = 门口装锁（只要推门 None）。两边都必须配 checkpointer。")
print("=" * 62)
