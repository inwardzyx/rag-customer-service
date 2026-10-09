# -*- coding: utf-8 -*-
"""
76-81 集：fork 复盘 + 长期记忆（Store）
每一行代码后面都有【语法N】标注，语法点编号在文件末尾汇总。

核心区别（先记住这张表）：
  checkpoint（前面几集）= 会话内记忆：一个 thread 的存档，会话结束还能续
  Store（这几集）        = 跨会话记忆：所有 thread 共用的"笔记本"，长期记住用户是谁
"""

import time
from langgraph.store.memory import InMemoryStore   # 【语法1】从模块导入一个类

# ============ 78：往"长期笔记本"里存数据 ============
store = InMemoryStore()                            # 【语法2】类名() 造对象（= StateGraph(State) 同款）

# put(在哪个抽屉, 贴什么标签, 写什么内容)
store.put(("users", "inward"), "prefs", {"theme": "深色", "city": "广州"})
#      【语法3】("users", "inward") 是元组 tuple：圆括号+逗号，不可改的"列表"
#      【语法4】用元组当"抽屉路径"：第 1 层大类，第 2 层具体给谁
#      【语法5】{"theme": "深色"} 字典字面量：键: 值，逗号隔开

store.put(("users", "inward"), "memo-1", {"text": "正在学 LangGraph"})
store.put(("users", "other"), "prefs", {"theme": "浅色"})   # 换个抽屉，给别人记的

print("== 78：存好了，读出来看看 ==")
item = store.get(("users", "inward"), "prefs")     # get(抽屉, 标签) → 返回 Item 对象
print("  读到的值：", item.value)                   # 【语法6】item.value：点号取对象的属性
print("  抽屉：  ", item.namespace)                 #        Item = {value, key, namespace...} 打包好的结果
print("  标签：  ", item.key)

# ============ 79+80：怎么"用"这些记忆 ============
print("\n== 79/80：search 按抽屉前缀搜（最常用）==")
# search(抽屉前缀) → 返回 list：把这个抽屉（含子抽屉）里的东西全列出来
found = store.search(("users", "inward"))          # 【语法7】search 返回的是列表，要 for 遍历
for it in found:                                   # 【语法8】for 对象 in 列表: 逐个拿出来
    print(f"  [{it.key}] {it.value}")              # 【语法9】f-string：{} 里直接放表达式

print("\n  还能按内容过滤：")
hits = store.search(("users", "inward"), filter={"theme": "深色"})
#                              【语法10】关键字参数 filter=：只留 value 里 theme 是深色的
for it in hits:
    print("  命中：", it.key, it.value)

# ============ 81：在节点里"用"记忆（环境上下文记忆） ============
print("\n== 81：节点运行时把记忆注入提示词 ==")


class State(dict):                                 # 【语法11】class X(dict) 继承 dict（这里简化演示）
    pass


def chat_node(state: State, *, store: InMemoryStore) -> dict:
    #            【语法12】第 2 个参数 store：compile(store=...) 之后框架自动喂进来
    prefs = store.get(("users", "inward"), "prefs")     # 读长期记忆
    theme = prefs.value["theme"] if prefs else "未知"    # 【语法13】A if 条件 else B 三元表达式
    #                                               prefs 可能是 None（没存过），先判再取
    print(f"  [节点] 从长期记忆读到：用户偏好主题 = {theme}")
    return {"messages": [f"回复（已按{theme}主题个性化）"]}


# 真实 langgraph 里是 compile(store=...)，这里手喂一下演示效果
_ = chat_node({"messages": ["你好"]}, store=store)
#                  【语法14】关键字传参：store=store，左边是形参名，右边是值

# ============ 76+77：fork 复盘（上轮真库已跑过） ============
print("\n== 76/77：fork 复盘 ==")
print("  76：fork 案例代码 = 从 get_state_history 挑快照，checkpoint_id 塞进 config 回档重跑")
print("  77：fork 实现 = 新快照的 parent 指向分叉点，旧线不删 —— 真库验证过两条链并存")
print("=" * 60)
print("语法点汇总：")
print("  1 from 模块 import 类      2 类名() 造对象        3 元组 (a, b)")
print("  4 元组当抽屉路径           5 字典字面量            6 点号取属性 obj.value")
print("  7 search 返回列表          8 for 遍历              9 f-string")
print("  10 关键字参数 filter=      11 class 继承           12 store 参数被框架注入")
print("  13 三元表达式 A if x else B                        14 关键字传参 fn(x=1)")
