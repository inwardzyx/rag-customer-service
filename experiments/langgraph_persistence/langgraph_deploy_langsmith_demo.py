"""
98-100 集"项目部署"章节的最小对照文件

    98  本地部署对接 LangSmith   → 代码零改动，只设 4 个环境变量
    99  LangSmith 的调试使用     → 网页上能看到什么（+ 本地等价物）
    100 大模型 agent 部署        → 把图变成 HTTP 服务，别人发请求就能用

跑法：
    <你的虚拟环境>/Scripts/python.exe langgraph_deploy_langsmith_demo.py

说明：LangSmith 需要注册账号拿 key，代码里我只做"检测+引导"，
不真的开启上报（免得没 key 时傻等网络超时）。
"""

import json
import operator
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Annotated, TypedDict

# ★ 必须在 import langgraph 之前 —— 本机应用控制策略拦掉了 uuid_utils 的 DLL，
#   而 langgraph 会经langchain_core 触发它，不加这两行会直接 ImportError 起不来。
#   （主服务 service.py 里是同样的顺序，理由见那里的注释）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env_compat                                   # noqa: E402
env_compat.ensure_uuid_utils()                      # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402


def sep(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ==================================================================
sep("98：对接 LangSmith —— 本地部署，零代码改动")
# ==================================================================

print("① 先确认：langsmith 装了吗？")
try:
    import langsmith
    print("   装了，版本", langsmith.__version__, "← 装 langgraph 时自动带上的，不用单独装")
except ImportError:
    print("   没装：<你的虚拟环境>/Scripts/python.exe -m pip install langsmith")

print("""
② 对接的全部工作 = 设 4 个环境变量（代码一行都不用改）：

    LANGSMITH_TRACING   = "true"                              ← 总开关，true 才上报
    LANGSMITH_API_KEY   = "lsv2_pt_xxxxxxxx"                  ← 钥匙，去 smith.langchain.com 注册拿
    LANGSMITH_ENDPOINT  = "https://api.smith.langchain.com"     ← 上报到哪（实测过，别写错成 sm）
    LANGSMITH_PROJECT   = "my-langgraph-demo"                 ← 网页上项目叫什么名字

设完之后，你的 LangGraph 代码【一行都不用动】——
langchain / langgraph 内部埋好了上报点，它们看见
LANGSMITH_TRACING=true 就自动把每一步发出去。

③ 三个新手必须知道的坑：
   1. key 千万别写死在代码里传到 git —— 它就是密码。
      正确姿势：写在系统环境变量里，或 .env 文件（并把它加进 .gitignore）。
   2. 钥匙不对 / 网络不通时，上报会失败，但【你的程序照常跑】——
      观测挂了 ≠ 业务挂了，最多在控制台看到几条警告。
   3. LANGSMITH_PROJECT 不存在会自动创建，不用提前在网页上建。""")

print("④ 当前这台机器的实际状态（检测，不乱设）：")
for k in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"]:
    v = os.environ.get(k)                    # 从系统环境变量里找，找不到返回 None
    print(f"   {k} = {'已设置' if v else '（未设置）'}")
print("   → 现在全是未设置，所以就算跑图也不会有任何上报（安静，不报错）。")
print("   → 你拿到 key 后，把这 4 个变量设进系统环境变量，重开终端再跑图就通了。")


# ==================================================================
sep("99：调试使用 —— 网页上能看到什么（+ 本地等价物）")
# ==================================================================

print("先搭一个两节点的校园事务图，等会既给它开 trace，也把它部署出去：")


class State(TypedDict):
    text: str                                   # 用户说的话
    intent: str                                 # 识别出的意图
    reply: str                                  # 最终回复
    log: Annotated[list, operator.add]          # 流水线日志（追加）


def analyze(state):
    text = state["text"]
    # ★ 这里是全文件唯一"用到就崩"的地方：第 105-107 行 answers 的【键】
    #   必须和这里产出的 intent 完全一致，否则下一行 answers[state["intent"]]
    #   直接 KeyError。改关键词时记得同步改字典的键（Agent 第一次扫的时候漏过这处）。
    if "请假" in text or "销假" in text:
        intent = "请假"
    elif "处分" in text or "违纪" in text:
        intent = "处分"
    else:
        intent = "闲聊"
    return {"intent": intent, "log": [f"识别意图={intent}"]}


def reply(state):
    answers = {"请假": "请假需填写《请假单》并逐级审批，超过三日须院系负责人批准。",
               "处分": "处分种类有警告、严重警告、记过、留校察看、开除学籍。",
               "闲聊": "我是校园事务助手，请问有什么需要办理？"}
    return {"reply": answers[state["intent"]], "log": ["回复已生成"]}


g = StateGraph(State)
g.add_node("analyze", analyze)
g.add_node("reply", reply)
g.add_edge(START, "analyze")
g.add_edge("analyze", "reply")
g.add_edge("reply", END)
app = g.compile()

print("""
99 集在 LangSmith 网页上主要看这 5 样东西（一次 trace = 一次 invoke 的全程录像）：

    1. Run Tree（运行树）   ：整次 invoke 展开，每个节点是一根枝
    2. 每个节点的输入/输出   ：点开 analyze，能看到 {'text': '...'} 进、{'intent': ...} 出
    3. 耗时                 ：每个节点花了多少毫秒，瓶颈一眼看出来
    4. Token 数             ：LLM 节点花了多少 token = 多少钱
    5. Playground 重跑      ：改改输入直接重跑这个节点，不用动代码

这 5 样东西的本质：LangSmith 把【每一步的输入、输出、耗时】记下来给你看。
所以不出门，本地也能做一版"穷人版 LangSmith"—— 用 stream 逐步打印：""")

print("本地等价演示（stream 版调试输出）：")
cfg = {"configurable": {"thread_id": "debug-1"}}
t0 = time.perf_counter()
for chunk in app.stream({"text": "请假需要谁审批", "intent": "", "reply": "", "log": []}, cfg):
    for node, update in chunk.items():
        cost = (time.perf_counter() - t0) * 1000
        print(f"   [{cost:7.1f} ms] 节点 {node!r} 产出：{update}")
print("   ↑ 网页上 Run Tree + 耗时两栏，本地版长这样（换汤不换药）")


# ==================================================================
sep("100：大模型 agent 部署 —— 把图变成 HTTP 服务")
# ==================================================================

print("""
部署 = 让【别人的程序】不用 import 你的代码，发一个 HTTP 请求就能用你的 agent。

    不部署：你的图只活在 demo.py 里，只有你能跑
    部署后：图挂在一个网址后面，网页/小程序/同事的代码都能调

下面用 Python 标准库（不用装任何包）把刚才的校园事务图变成一个服务：
""")

# 服务要用的是【编译好的图】app，线程里直接用它


class ChatHandler(BaseHTTPRequestHandler):
    """一个类：收到 POST 请求该干什么。继承 BaseHTTPRequestHandler。"""

    def do_POST(self):
        # ① 读出请求体（客户发来的 JSON）
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # ② 跑图（和平时一模一样，只是输入来自网络）
        out = app.invoke(
            {"text": body.get("text", ""), "intent": "", "reply": "", "log": []},
            {"configurable": {"thread_id": body.get("session", "web")}},
        )
        # ③ 把结果打包成 JSON 回回去
        resp = json.dumps({"reply": out["reply"]}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args):      # 关掉控制台默认请求日志，保持输出干净
        pass


PORT = 8123
service = HTTPServer(("127.0.0.1", PORT), ChatHandler)
threading.Thread(target=service.serve_forever, daemon=True).start()
#   ↑ 起一个后台线程专门"接电话"，主线程继续往下跑演示
#   ↑ daemon=True：主程序结束时它自动跟着关，不会挂着不走

print(f"服务已挂在 http://127.0.0.1:{PORT}/chat 上，现在假装是个客户端发请求：")

for question in ["请假需要谁审批", "你好呀"]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/chat",
        data=json.dumps({"text": question, "session": "u1"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        answer = json.loads(r.read())
    print(f"   用户说：{question}")
    print(f"   agent 答：{answer['reply']}")

print("""
↑ 这就是部署成功的样子：agent 没有被 import，隔着网络被调用。

100 集讲的实际产品级做法是官方的 LangGraph Server：
    1. 写一个 langgraph.json 配置文件，声明"哪个图、什么入口"
    2. 装 langgraph-cli，跑 langgraph dev   ← 本地起一个标准服务
    3. 自动获得 REST API + 网页调试台，还自带持久化（用的就是前面学的 checkpointer）
    4. 上线时 langgraph up / 部署到 LangGraph Platform

不管包装多花哨，核心和我们刚才手写的最小版一模一样：
【HTTP 进 → 跑图 → JSON 出】。""")

service.shutdown()          # 演示完把服务关掉，端口释放
print("演示结束，本地服务已关闭。")
