# -*- coding: utf-8 -*-
"""
Step 3：把"假工具"换成【真干活】的工具，并加上真实项目必须有的【参数校验】。

这一步做完，你的图就不再是对话玩具：它能真的去读磁盘上的文件。

跑法：
    set LANGSMITH_TRACING=false
    python step3_real_tools.py
"""
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

# ==================== ① 知识库：真文件，第一次跑会自动建 ====================
DOCS = Path(__file__).parent / "step3_docs"

SAMPLE_DOCS = {
    "退款政策.md": """# 退款政策
- 未发货：可全额退款，1-3 个工作日到账。
- 已发货未签收：需拒收后退款，扣除 10 元运费。
- 已签收：7 天无理由退货，商品需保持完好。
- 生鲜类商品不支持 7 天无理由。
""",
    "发货时效.md": """# 发货时效
- 普通订单：付款后 48 小时内发货。
- 预售商品：以商品页标注时间为准，通常 7-15 天。
- 定制商品：7 个工作日。
- 法定节假日顺延。
""",
    "会员权益.md": """# 会员权益
- 白银会员：满 99 元包邮。
- 黄金会员：满 59 元包邮，生日双倍积分。
- 钻石会员：全年包邮，专属客服，退货免运费。
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
print("实验 A：退款要多久到账？（工具会真的去读磁盘）")
run("我买的东西还没发货，现在退款多久能到账？")

# ==================== 实验 B：故意给错文件名，看它会不会自己纠错 ====================
print("\n" + "=" * 62)
print("实验 B：用户在问题里说了一个【不存在】的文档名")
run("请打开《退货规则》这份文档，告诉我退货有什么条件。")

# ==================== 实验 C：绕过模型，直接测工具的防线（不花钱） ====================
print("\n" + "=" * 62)
print("实验 C：直接调工具，测校验拦不拦得住（不走模型，秒出结果）")
for bad in ["不存在的文件.md", "../../../Windows/win.ini", "退款政策.md"]:
    print(f"   传 {bad!r} → ", end="")
    result = read_doc.invoke({"filename": bad})
    print(result.replace("\n", " ")[:70])

print("\n" + "=" * 62)
print(f"知识库目录：{DOCS}（去打开看看，是三个真的 .md 文件）")
print(f"总账单：输入 {BILL['prompt']} tokens，输出 {BILL['completion']} tokens")
print("=" * 62)
