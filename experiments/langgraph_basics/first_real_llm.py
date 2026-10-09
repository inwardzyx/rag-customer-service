# -*- coding: utf-8 -*-
"""
第 0 步：真的调一次大模型（告别假 LLM）

之前 100 集里所有的 "LLM" 都是我写的假函数（fake_llm 返回固定的 tool_calls）。
这个文件让你看看【真模型】到底长什么样，以及框架在你背后替你做了什么。

跑之前：
    set LANGSMITH_TRACING=false          ← 关掉上报，免得联网拖慢
    set DEEPSEEK_API_KEY=你的key          ← 如果环境变量里没有才需要设

跑法：
    <你的虚拟环境>/Scripts/python.exe first_real_llm.py
"""

import os
import time


def hav(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


# ---------------------------------------------------------
hav("0. 先检查钥匙在不在")
# ---------------------------------------------------------

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    print("没找到 DEEPSEEK_API_KEY 环境变量。")
    print("去 platform.deepseek.com 注册 → 创建 API key → 然后在终端里：")
    print('    setx DEEPSEEK_API_KEY "sk-xxxxxx"')
    print("重开终端再跑本文件。")
    raise SystemExit(0)

print(f"   找到了：长度 {len(key)}，前缀 {key[:6]}…（完整 key 不打印，避免泄露）")

# ---------------------------------------------------------
hav("1. 用官方协议直接问一句（不经过任何框架）")
# ---------------------------------------------------------

from openai import OpenAI

client = OpenAI(
    api_key=key,
    base_url="https://api.deepseek.com",      # DeepSeek 兼容 OpenAI 的协议
)

t0 = time.time()
resp = client.chat.completions.create(
    model="deepseek-chat",                    # 模型名：要调哪个模型
    messages=[
        {"role": "system", "content": "你是一个简洁的助手，回答控制在两句话内。"},
        {"role": "user", "content": "用一句话解释什么是向量数据库。"},
    ],
    temperature=0.3,                          # 越低越稳定，越高越有创造性
)
cost = time.time() - t0

print(f"   耗时：{cost:.2f} 秒")
print(f"\n   模型回答：\n      {resp.choices[0].message.content}")

print("\n   看清返回结构（框架就是把这个盒子包装成别的样子）：")
print(f"     resp.choices[0].message.role     = {resp.choices[0].message.role!r}")
print(f"     resp.choices[0].message.content  = 上面那段话")
print(f"     resp.choices[0].message.tool_calls = {resp.choices[0].message.tool_calls!r}")
print("                                        ↑ None：这次它没想调工具，直接回答了")
print(f"     resp.model                       = {resp.model!r}")
print("                                        ↑ 我请求的是 deepseek-chat，回来的可能是")
print("                                          deepseek-flash（服务端会路由），以这个字段为准")
print(f"     resp.usage.prompt_tokens         = {resp.usage.prompt_tokens}")
print(f"     resp.usage.completion_tokens     = {resp.usage.completion_tokens}")
print("                                        ↑ 这两个数字就是算钱的地方")

# ---------------------------------------------------------
hav("2. 同样一句话，用 LangChain 封装写（你熟悉的形态）")
# ---------------------------------------------------------

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
msg = llm.invoke("用一句话解释什么是向量数据库。")

print(f"   类型：{type(msg).__name__}        ← 注意！不是字符串，是一个消息对象")
print(f"   msg.content = {msg.content!r}")
print(f"   msg.tool_calls = {msg.tool_calls!r}")
print("\n   对比：框架替你做了三件事")
print("     ① 把 ['system','user'] 那套 messages 结构藏起来了（直接传字符串就行）")
print("     ② 把 resp.choices[0].message 这一长串，简化成一个 msg 对象")
print("     ③ 各家模型的差异统一了 —— 换模型只改一行")

# ---------------------------------------------------------
hav("3. 加餐：让真模型真的开工单（这就是 101-105 集那张工单的来源）")
# ---------------------------------------------------------

def get_order_status(order_id: str) -> str:
    """根据订单号查询订单状态。"""
    orders = {"A1001": "已发货，48 小时内到达", "B2002": "待付款"}
    return orders.get(order_id, f"没找到订单 {order_id}")


tools = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "根据订单号查询订单状态",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号，如 A1001"}},
            "required": ["order_id"],
        },
    },
}]

resp2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "我的订单 A1001 到哪了？"}],
    tools=tools,
)
tc = resp2.choices[0].message.tool_calls

print(f"   用户问：'我的订单 A1001 到哪了？'")
print(f"   模型 content   = {resp2.choices[0].message.content!r}")
print("      ↑ 注意：不是空的！模型一边说话一边开工单，两者可以同时存在")
print("        （之前 fake_llm 一直写成 content=''，那是简化了，真模型不一定）")
print(f"   模型 tool_calls = ")
if tc:
    for c in tc:
        print(f"       id   = {c.id}")
        print(f"       name = {c.function.name}")
        print(f"       args = {c.function.arguments}   ← 注意是 JSON 字符串，不是字典")
else:
    print("      （这次模型没调工具，可能直接回答了）")

print("\n   ↑↑ 看到没？这就是之前 fake_llm 一直在伪造的东西。")
print("      现在它是模型自己算出来的：它读了工具的说明书(description)，")
print("      判断『这个问题该查订单』，还自己填出了 order_id='A1001'。")

if tc:
    import json
    args = json.loads(tc[0].function.arguments)
    print(f"\n   把工单交给工人执行：get_order_status(**{args})")
    print(f"   工人返回：{get_order_status(**args)}")
    print("\n   到这里，101-105 集那张图就【全部接上了真模型】——")
    print("   之前是 fake_llm 开工单，现在是真模型开工单，其余代码一行不用改。")

# ---------------------------------------------------------
hav("下一步")
# ---------------------------------------------------------
print("""
   这一小步补上了最大的短板：你终于真的调过一次大模型了。

   接下来按这个顺序走（每步都很小）：
     Step 1  把之前某个 demo 里的 fake_llm 换成真模型，跑一遍看差别
     Step 2  用 bind_tools 让真模型驱动 ToolNode（LangGraph 版，全自动）
     Step 3  做一个有真实用途的小工具（这一步开始才是"项目"）
     Step 4  加 FastAPI 接口 + Docker，让它能被别人访问
     Step 5  写 README，推 GitHub

   现在先做一件事：把这个文件跑通，然后告诉我 resp.usage 里的两个数字是多少？
   那两个数字决定你这几分钱花在哪了。
""")
