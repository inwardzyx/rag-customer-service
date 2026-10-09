# -*- coding: utf-8 -*-
"""
Step 3：把"假工具"换成【真干活】的工具，并加上真实项目必须有的【参数校验】。

这一步做完，你的图就不再是对话玩具：它能真的去读磁盘上的文件。

跑法：
    set LANGSMITH_TRACING=false
    python step3_real_tools.py
"""
import os
import sys
from pathlib import Path

# ★ 这两行必须在 import langchain_* 之前 —— 本机应用控制策略拦掉了 uuid_utils 的
#   DLL，langchain_core 导入 callbacks 时就会 import 它。不加会直接 ImportError。
#   （主服务 service.py 里是同样的顺序，理由见那里的注释）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_compat                                   # noqa: E402
env_compat.ensure_uuid_utils()                      # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool                                   # noqa: E402
from dotenv import load_dotenv                                          # noqa: E402
load_dotenv()
from langchain_deepseek import ChatDeepSeek                              # noqa: E402
from langgraph.graph import END, START, StateGraph                      # noqa: E402
from langgraph.graph.message import add_messages                       # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition               # noqa: E402
from typing import Annotated, TypedDict                                 # noqa: E402

# ==================== ① 知识库：真文件，第一次跑会自动建 ====================
DOCS = Path(__file__).parent / "step3_docs"

SAMPLE_DOCS = {
    "学生请销假制度.md": """# 学生请销假制度
- 事假：须由家长与班主任联系说明情况，班主任登记在册后视情况审核，一般不得超过两周。
- 病假：须提供校医院或二级以上医院的证明，经班主任审核后报院系批准。
- 销假：返校后三个工作日内提交销假单，逾期须说明原因。
- 未经批准擅自离校，按学生纪律处分相关规定处理。
""",
    "请假审批流程.md": """# 请假审批流程
- 一日以内：由班主任审批。
- 三日以内：由班主任审核后报院系备案。
- 超过三日：须院系负责人批准，并通知家长。
- 节假日期间离校，须提前书面申请。
""",
    "违纪处分种类.md": """# 违纪处分种类
- 警告：情节轻微，违反一般管理规定。
- 严重警告：情节较重，造成不良影响。
- 记过：违反纪律，情节严重。
- 留校察看：屡次违纪或情节恶劣，察看期为一年。
- 开除学籍：严重违反法律法规或学校规章。
""",
}


def ensure_docs():
    """目录不存在就建，并把示例文档写进去（真的写到磁盘上，你可以打开看）"""
    DOCS.mkdir(exist_ok=True)
    for name, text in SAMPLE_DOCS.items():
        (DOCS / name).write_text(text, encoding="utf-8")


ensure_docs()


# ==================== ② 真工具 + 参数校验 ====================
@tool
def list_docs() -> str:
    """列出知识库里所有可查阅的文档文件名。用户问政策、规则、权益时，先调这个看看有哪些文档。"""
    names = [p.name for p in sorted(DOCS.glob("*.md"))]
    #                     └ sorted() 排序，保证每次顺序一致（不排序的话顺序随机）
    return "可查阅的文档：\n" + "\n".join(f"- {n}" for n in names)


@tool
def read_doc(filename: str) -> str:
    """按文件名读取文档全文。文件名必须是 list_docs 列出来的那个，不能带路径。"""
    # —— 校验 1：防路径穿越（安全优先）——
    if "/" in filename or "\\" in filename or ".." in filename:
        return "错误：只能传文件名，不能带路径。请改用 list_docs 列出的文件名。"

    # —— 校验 2：白名单（只许读知识库里的文件）——
    allowed = [p.name for p in DOCS.glob("*.md")]
    if filename not in allowed:
        return f"错误：找不到 {filename!r}。可选的文件名是：{allowed}"

    # —— 校验通过，真的去读磁盘 ——
    return (DOCS / filename).read_text(encoding="utf-8")
    #       └ Path 对象用 / 拼路径（比字符串拼接安全，Windows/Mac 都能用）


TOOLS = [list_docs, read_doc]


# ==================== ③ State / 模型 / 图（和 Step 2 一样） ====================
class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def chat(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


g = StateGraph(State)
g.add_node("chat", chat)
g.add_node("tools", ToolNode(TOOLS))
g.add_edge(START, "chat")
g.add_conditional_edges("chat", tools_condition)
g.add_edge("tools", "chat")
app = g.compile()


BILL = {"prompt": 0, "completion": 0}


def run(question):
    print("\n" + "-" * 62)
    print("用户：", question)
    out = app.invoke({"messages": [HumanMessage(content=question)]})
    for m in out["messages"]:
        kind = type(m).__name__
        if kind == "AIMessage":
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"   → 开工单：{tc['name']}({tc['args']})")
            else:
                print(f"   → 回答：{m.content}")
        elif kind == "ToolMessage":
            body = m.content
            short = body if len(body) <= 60 else body[:60] + "…"
            print(f"   ← 回执：{short}")
    usage = out["messages"][-1].response_metadata.get("token_usage")
    if usage:
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        BILL["prompt"] += p
        BILL["completion"] += c
        print(f"   [本次] 输入 {p} / 输出 {c} tokens")


# ==================== 实验 A：正常提问，看它真去读文件 ====================
print("=" * 62)
print("实验 A：事假最多能请多久？（工具会真的去读磁盘）")
run("我想请事假，最长能请多久？")

# ==================== 实验 B：故意给错文件名，看它会不会自己纠错 ====================
print("\n" + "=" * 62)
print("实验 B：用户在问题里说了一个【不存在】的文档名")
run("请打开《学生住宿管理办法》这份文档，告诉我宿舍有什么规定。")

# ==================== 实验 C：绕过模型，直接测工具的防线（不花钱） ====================
print("\n" + "=" * 62)
print("实验 C：直接调工具，测校验拦不拦得住（不走模型，秒出结果）")
for bad in ["不存在的文件.md", "../../../Windows/win.ini", "学生请销假制度.md"]:
    print(f"   传 {bad!r} → ", end="")
    result = read_doc.invoke({"filename": bad})
    print(result.replace("\n", " ")[:70])

print("\n" + "=" * 62)
print(f"知识库目录：{DOCS}（去打开看看，是三个真的 .md 文件）")
print(f"总账单：输入 {BILL['prompt']} tokens，输出 {BILL['completion']} tokens")
print("=" * 62)
