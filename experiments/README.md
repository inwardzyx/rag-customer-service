# experiments/ —— 学习过程与语言练习脚本

## 这个目录是什么

主服务（`service.py`）是**产品**，这个目录是**学习过程**。

两者刻意分开：`service.py` 一行LangGraph 都不import，这个目录里全是LangGraph。
不是为了显得项目"用了 Agent 框架"，而是因为**它们解决的是不同的问题**——
主链路是一条能离线量测、能变异测试的线性管线；图编排（循环、条件边、跨会话记忆）
是另一类问题，值得单独拿一个目录放。

## ★ 来源说明（请先读这段）

这里的脚本**大部分是跟着视频课敲的**，不是原创项目。分成三类：

| 类别 | 文件 | 说明 |
|---|---|---|
| **跟课 demo** | `step1`~`step6`、`langgraph_*/`（20 个 3100 多行） | 跟着课程敲的知识点练习。**代码里保留了「和 N集一样」的注释** —— 这些注释是有意留下的，不删。 |
| **我自己写的** | `evalset/probe_ablation.py`、`evalset/probe_sensitivity.py` | 混合检索三路消融、RRF 的 k 敏感性。**不是跟课的**，是项目自己的实验。 |
| **我自己写的** | `python_syntax_for_langgraph.py`（1186 行） | Python 语法补漏速查表，为看懂 LangGraph 代码而写。 |

为什么要如实标注：跟课demo 本身**不是减分项**——大多数 AI 应用岗的候选人都是这么学Agent 框架的。
**说清楚来源才是加分项**，因为它证明了两件事：
1. 我能照着文档跑通一套完整的技术栈（自学能力的证据）；
2. 我**知道**哪些是抄的、哪些是我想的（这个区分能力才是工程能力）。

★ 但我也承认这份目录**目前还没长出我自己的东西**——
`service.py` 里的拒答短路是"在什么时候该拒绝"，本质上就是一个 guardrail 节点，
把它写成图不是没道理，但那需要先有第二类问题（多轮/改写/工具选择）值得用图。**现在还没有。**

## ★ 语料：为什么这里也是校园政策

主服务 `docs/kb/` 早就从虚构电商换成了真实校规，但这个目录一直还用电商语料。
**2026-10-09 全部换齐**（223 处），现在两个目录用同一套校园语料。

换的时候踩了三类坑，都记在 `_语料映射表.md` 里（含改完实测到的数字）：

1. **有5 处「改漏就崩」的隐性依赖** —— `CORPUS["..."]` 硬索引、答案字典的键、
   实验里"故意给错的文件名"、"合法文件名"样本。改漏是 `KeyError`，不是显示问题。
2. **有一批数字必须重测** —— 近重复阈值和它的两个实测值（0.9274→ **0.968**）、
   块数（20 → **12**）、召回率。换语料不测 = 注释里的数字全是谎话。
   ⇒ 现在这些数字都由代码动态打印（`len(EVAL)`、`len(CHUNKS)`），
   **不手写在注释里**，就是为了让"下次再换语料"不会忘了重测。
3. **★ 真实校规撑不起某些演示** —— 实测真实校规最短 22 字，而"太短碎屑"关卡
   的阈值是 15 字，所以直接搬真实语料会让那个关卡打出"拦下 0 块"。
   这类地方必须**人工造一条**，并在旁边写明为什么。

★ 顺带修掉两个真bug（改写时为了"跑一遍验证"才暴露出来的）：
7 个脚本缺 `env_compat.ensure_uuid_utils()`（本机根本起不来）；
`step4` 的 `try: import faiss` 探测是假的（被策略拦截时 import 成功但属性为空）。

## 目录结构

```
experiments/
├── step1_real_llm_graph.py        跟课│ 最小的 StateGraph + 真 LLM
├── step2_two_tools_choice.py      跟课│ 拆 tools_condition 黑盒 + 双工具选择
├── step3_real_tools.py            跟课│ 假工具换成真读文件 + 参数校验
├── step4_rag_demo.py              跟课│ RAG 完整链路（切块/向量/检索/生成）
├── step5_ingest_guard_demo.py     跟课│ 入库把关 + rerank
├── step6_fastapi_service.py       跟课│ 把 RAG 整条链包成 HTTP 服务
│
├── langgraph_basics/              ┐
│   ├── langgraph_state_toy.py          │
│   ├── langgraph_control_flow_demo.py  │ 跟课│ 控制流、条件边、子图
│   ├── langgraph_parent_chain.py       │    │ 父子图 + PostgresSaver
│   ├── langgraph_protection_layers.py  │    │ 重试/兜底/异常处理
│   ├── langgraph_toolnode_demo.py      │    │ 工具调用（★ 面试高频）
│   └── first_real_llm.py               ┘
│
├── langgraph_memory/              ┐
│   ├── langgraph_persistence_demo.py   │
│   ├── langgraph_memory_min.py         │ 跟课│ checkpoint（会话内记忆）
│   ├── langgraph_store_demo.py         │    │ Store（跨会话长期记忆）
│   ├── langgraph_recover_fork_demo.py  │    │ 时间旅行 / fork
│   └── langgraph_resume_two_ways.py    ┘
│
├── langgraph_hitl/                ┐
│   ├── langgraph_interrupt_demo.py      │
│   ├── langgraph_interrupt_min.py       │ 跟课│ Human-in-the-loop
│   ├── langgraph_interrupt_min2.py      │    │ 中断恢复的坑
│   ├── langgraph_interrupt_two_cases.py │
│   └── langgraph_interrupt_static_demo.py┘
│
├── langgraph_persistence/         ┐
│   ├── langgraph_postgres_demo.py       │ 跟课│ PostgresSaver（我真跑过）
│   ├── langgraph_checkpoint_inspect.py  │    │ 存档结构探查
│   └── langgraph_deploy_langsmith_demo.py┘
│
├── rag_concepts_demo.py           跟课│ RAG 概念从零讲（含噪声实验）
└── python_syntax_for_langgraph.py我写│ Python 语法补漏速查
```

## 跑之前

```bash
# 这个目录的依赖是单独拆出来的，因为主服务一行都不用它
#（混进 requirements.txt 会让人误以为 service.py 用了 LangGraph）
pip install -r requirements-experiments.txt
```

**大部分脚本不需要 API key**（用假模型/假工具）。要真跑通的需要：

```bash
set DEEPSEEK_API_KEY=sk-xxx
set LANGSMITH_TRACING=false    # ★ 见下方"坑"
```

### ★ 两个坑

1. **`LANGSMITH_TRACING` 必须设成 false**。默认 `true` 时，每一步都会上报到 LangSmith，
   既慢又要key，不设的话脚本会卡在等网络。
2. **连接 Postgres 的四个脚本默认连 `localhost`**（`langgraph_persistence/`、
   `langgraph_memory/langgraph_recover_fork_demo.py`、`langgraph_basics/langgraph_parent_chain.py`）。
   我自己原来是在 VirtualBox 虚拟机里跑的，但**那是我的环境，不是你的**，
   所以改成了读环境变量：

   ```bash
   set LANGGRAPH_DB_URI=postgresql://用户名:密码@主机:5432/库名
   ```

   不设就用 `localhost`。

### 语法补漏表

`python_syntax_for_langgraph.py` 是个**单文件速查表**，覆盖 LangGraph 代码里天天见到的
Python 语法（类型标注 / reducer / 字典推导 / f-string / TypedDict / Annotated / try-except），
每条都是"一句话说明 + 最小可跑例子 + 在 LangGraph 哪儿见过"。标准库就能跑，不用装包：

```bash
python python_syntax_for_langgraph.py
```

如果读 LangGraph 代码卡在语法上，先跑这个。

## ★ 还没做的（写在这里比被问到好）

- **把这些 demo 的成果用进自己的项目** —— 目前零。`service.py`是单轮问答，
  没有多轮、没有工具选择、没有 agentic 回路。
- **没做 MCP server**（把知识库包成 MCP 工具）。
- **没有多智能体编排**（supervisor / 集群调度）。
- **没有向量数据库**（68 个块用 numpy 全量点乘，理由见 README 的取舍表）。

这四条是 AI Agent 岗 JD 里高频出现的关键词，而这里全是空的。
**我不想用"我看过"冒充"我做过"，所以先如实写在这里。**