"""LangGraph 状态合并机制的"玩具版"
纯 Python，不需要安装任何包，直接 python 这个文件就能跑。
目的：看懂 invoke 里每个 super-step 到底对 state 做了什么。
"""


class LastValue:
    """没配 reducer 的字段：新值直接覆盖旧值"""

    def __init__(self, value=None):
        self.value = value

    def update(self, old, new):
        print(f"      [LastValue] 旧={old!r} 新={new!r} → 直接扔掉旧的，用新的")
        return new


class Aggregator:
    """配了 reducer 的字段：交给 reducer 合并"""

    def __init__(self, reducer, value=None):
        self.reducer = reducer      # reducer 就是一个普通函数 (old, new) -> merged
        self.value = value

    def update(self, old, new):
        result = self.reducer(old, new)
        print(f"      [Aggregator] 旧={old!r} 新={new!r} → 合并成 {result!r}")
        return result


def apply_update(channels: dict, updates: dict) -> dict:
    """invoke 里每个 super-step 都在干的事

    channels: 字段名 -> 一个"盒子"（每个盒子存着这个字段当前的值和合并规则）
    updates:  节点函数返回的增量 dict（只写了要改的字段）
    """
    print(f"    节点返回了增量: {updates}")
    for k, v in updates.items():
        # 一行拆成四行，逻辑完全一样：
        box = channels[k]              # 1. 拿出这个字段对应的盒子
        old = box.value                # 2. 看盒子里现在存着什么（旧值）
        merged = box.update(old, v)    # 3. 按盒子的规则，算出"新值应该是什么"
        box.value = merged             # 4. 写回盒子
    # 把所有盒子里的值倒出来，拼成一个普通 dict —— 这就是节点里看到的 state
    return {k: c.value for k, c in channels.items()}


if __name__ == "__main__":
    # reducer：就是一个普通函数，告诉框架"两份值怎么合"
    def add_item(old, new):
        return (old or []) + new

    # 定义"字段 -> 盒子"的对应关系（LangGraph 在 compile() 时干的就是这一步）
    channels = {
        "step": LastValue(0),              # 没配 reducer → 覆盖
        "messages": Aggregator(add_item, []),  # 配了 reducer → 追加
    }

    print("初始:", {k: c.value for k, c in channels.items()})

    print("\n第 1 个 super-step（节点 A 写完）:")
    print("  最终 state:", apply_update(channels, {"messages": ["你好"]}))

    print("\n第 2 个 super-step（节点 B 同时改了两个字段）:")
    print("  最终 state:", apply_update(channels, {"messages": ["再来一条"], "step": 1}))

    print("\n第 3 个 super-step（节点 C 只改 step）:")
    print("  最终 state:", apply_update(channels, {"step": 2}))

    print("\n---- 对比：如果 step 也用覆盖会怎样 ----")
    print("  step 从 0 → 1 → 2，每次都整个替换，这才是计数器的正确行为")
    print("  messages 每次都在旧列表后面追加，所以聊天历史不会丢")
