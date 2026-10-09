"""
两种"解冻"方式的最小对照文件

    Command(resume=...)   ← 有人举手，要递一张纸条进去
    invoke(None, cfg)     ← 只是门口上了锁，不用递东西，推门就行

跑法：
    <你的虚拟环境>/Scripts/python.exe langgraph_resume_two_ways.py

全文只有一张图、两个故事，外加两段"故意用错"的实测。
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


def sep(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# log 用 operator.add 当 reducer：返回的新列表是【追加】上去的，不是覆盖
class State(TypedDict):
    log: Annotated[list, operator.add]
    answer: str


def prepare(state):
    return {"log": ["① 资料准备好了"]}


# ==================================================================
sep("故事 A：节点【里面】有人举手 → 用 Command(resume=...) 递纸条")
# ==================================================================

RUNS = {"ask": 0}          # 数一下 ask 这个函数一共被跑了几遍
VERBOSE = True             # 加餐那段会跑很多次，先把调试打印关掉


def ask(state):
    RUNS["ask"] += 1
    if VERBOSE:
        print(f"   [ask 第 {RUNS['ask']} 次被调用] 走到 interrupt 这一行……")

    ans = interrupt("能提交吗？(yes / no)")
    #  ↑ 第一次跑到这里：函数"冻住"，这一行【没有返回值】，后面几行根本没执行
    #  ↑ 恢复后再跑：这一行【直接返回你 resume 进来的那个值】，然后往下走

    if VERBOSE:
        print(f"   [ask 第 {RUNS['ask']} 次] interrupt 返回了：{ans!r} ← 这就是 resume 递进来的东西")
    return {"log": [f"② 拿到答复：{ans!r}"], "answer": ans}


def submit(state):
    return {"log": ["③ 提交完成"]}


def build(before=None):
    g = StateGraph(State)
    g.add_node("prepare", prepare)
    g.add_node("ask", ask)
    g.add_node("submit", submit)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "ask")
    g.add_edge("ask", "submit")
    g.add_edge("submit", END)
    return g.compile(checkpointer=InMemorySaver(), interrupt_before=before)


appA = build()
cfgA = {"configurable": {"thread_id": "story-A"}}

out = appA.invoke({"log": [], "answer": ""}, cfgA)
print("第一次 invoke 返回：", out["log"])
print("   注意：② 不在里面 —— ask 举着手，还没跑完，自然没产出")
print("   next =", appA.get_state(cfgA).next, " ← 卡在 ask 这一站")

print("\n-- 现在递纸条：Command(resume='yes') --")
out = appA.invoke(Command(resume="yes"), cfgA)
print("恢复后结果：", out["log"])
print("   answer 字段 =", out["answer"], " ← interrupt() 的返回值原封不动到了这里")
print(f"   ask 一共被调用了 {RUNS['ask']} 次 ← 恢复时【整个节点从头重跑】")


# ==================================================================
sep("A 加餐：resume 递什么，interrupt 就返回什么（不止字符串）")
# ==================================================================

print("resume 的值没有任何限制，递什么进去就原样出来：")
VERBOSE = False                      # 下面要跑 4 轮，先把 ask 里的调试打印关掉
for value in ["yes", 42, {"reason": "预算够", "by": "张三"}, ["a", "b"]]:
    cfg = {"configurable": {"thread_id": f"payload-{type(value).__name__}"}}
    app = build()
    app.invoke({"log": [], "answer": ""}, cfg)
    o = app.invoke(Command(resume=value), cfg)
    print(f"   resume={value!r:<38} → 节点里收到 {o['answer']!r}")
VERBOSE = True                       # 加餐结束，恢复调试打印



# ==================================================================
sep("A 反例：门口有人举手，你却只喊一声 invoke(None) —— 会怎样？")
# ==================================================================

RUNS["ask"] = 0
cfgBad = {"configurable": {"thread_id": "story-A-wrong"}}
appBad = build()
appBad.invoke({"log": [], "answer": ""}, cfgBad)
print("举手状态下用 invoke(None, cfg)：")
try:
    appBad.invoke(None, cfgBad)
    print("   没报错，但 log 还是：", appBad.get_state(cfgBad).values["log"])
    print("   next =", appBad.get_state(cfgBad).next, "← 又冻在原地，白跑一趟")
except Exception as e:
    print("   报错：", type(e).__name__, e)


# ==================================================================
sep("故事 B：节点【门口】上了锁 → 用 invoke(None, cfg) 推门")
# ==================================================================


def check(state):
    # 这个函数里【一个 interrupt 都没写】
    return {"log": ["② 质检员看了一眼，通过"]}


g = StateGraph(State)
g.add_node("prepare", prepare)
g.add_node("check", check)
g.add_node("submit", submit)
g.add_edge(START, "prepare")
g.add_edge("prepare", "check")
g.add_edge("check", "submit")
g.add_edge("submit", END)
appB = g.compile(checkpointer=InMemorySaver(), interrupt_before=["check"])
#                                             ↑ 锁装在 compile 这里，节点代码没动过

cfgB = {"configurable": {"thread_id": "story-B"}}
out = appB.invoke({"log": [], "answer": ""}, cfgB)
print("第一次 invoke 返回：", out["log"])
print("   next =", appB.get_state(cfgB).next, " ← 停在 check 门口，还没进门")
snap = appB.get_state(cfgB)
print("   有没有人举手：", any(t.interrupts for t in snap.tasks), "← False：锁不是人")

print("\n-- 推门：invoke(None, cfg) --")
out = appB.invoke(None, cfgB)
print("推门后结果：", out["log"])
print("   第一个参数 None = '我不给新输入，你接着上次的存档往下跑'")


# ==================================================================
sep("B 反例：只是门口上锁，你却递纸条 Command(resume=...) —— 会怎样？")
# ==================================================================

cfgB2 = {"configurable": {"thread_id": "story-B-wrong"}}
g2 = StateGraph(State)
g2.add_node("prepare", prepare)
g2.add_node("check", check)
g2.add_node("submit", submit)
g2.add_edge(START, "prepare")
g2.add_edge("prepare", "check")
g2.add_edge("check", "submit")
g2.add_edge("submit", END)
appB2 = g2.compile(checkpointer=InMemorySaver(), interrupt_before=["check"])

appB2.invoke({"log": [], "answer": ""}, cfgB2)
print("门锁状态下用 Command(resume='yes')：")
try:
    o = appB2.invoke(Command(resume="yes"), cfgB2)
    print("   居然也跑完了：", o["log"])
    print("   ← 因为没人接这张纸条，resume 的值被丢掉了，效果等同推门")
except Exception as e:
    print("   报错：", type(e).__name__, e)


# ==================================================================
sep("一句话总结")
# ==================================================================

print("""
Command(resume=X)  ← 有人举手：把 X 递进去，X 会变成 interrupt() 的返回值
invoke(None, cfg)  ← 只是门锁：什么都不递，让图接着存档往下跑

判断用哪个，问自己一句话：
    【卡住的地方，有没有人在等一个答案？】
    有人等  → Command(resume=...)
    没人等  → invoke(None, cfg)

程序里的判断写法：
    any(t.interrupts for t in app.get_state(cfg).tasks)
        True  → Command
        False → None
""")
