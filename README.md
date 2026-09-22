# 客服知识库问答（RAG 服务）

> **仓库地址**：<https://github.com/inwardzyx/rag-customer-service>

一个能跑起来的**检索增强生成（RAG）完整链路**：从"接大模型"一路做到"HTTP 服务 + 网页聊天框"。
核心不在"用了多少框架"，而在**每一步为什么要这么做、翻过什么车、怎么修的**。

> 数据集是 10 份虚构的电商客服文档（退款、发货、保修、税费、物流……），
> 故意混进脏数据（旧版条款、重复话术、客户手机号/身份证）来验证入库把关真的在工作。

---

## 30 秒跑起来

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配两个环境变量（Windows 用 setx，Mac/Linux 用 export）
setx DEEPSEEK_API_KEY  sk-你的key
setx HF_ENDPOINT       https://hf-mirror.com     # 国内拉 embedding 模型必须走镜像

# 3. 起服务
python step6_fastapi_service.py
```

打开 <http://127.0.0.1:8000> 就是聊天页面，
<http://127.0.0.1:8000/docs> 是自动生成的接口文档。

---

## 它长什么样

```
                    ┌─────────────── 启动时做一次 ───────────────┐
  知识库文档  ──→   切块 → embedding → 入库把关（5 道关卡）→ 向量库
                    └───────────────────────────────────────────┘
                                                      │
  用户提问  ──→  embedding ──→ 粗捞 top-5 ──→ rerank 精排 ──→ 取 top-3
                                                      │
                                                      ↓
                                              大模型看着资料作答
                                                      │
                                                      ↓
                                          答案 + 引用出处（可溯源）
```

**一句话总结这条链**：先粗捞（快、宁可捞错也别漏），再精排（慢、只挑最相关的），
最后让大模型**只看着资料回答**，资料里没有就明确说"不知道"。

---

## 六个文件，一步一步长出来的

| 文件 | 这一步解决什么 | 关键收获 |
|---|---|---|
| `step1_real_llm_graph.py` | 把真实大模型接进 LangGraph 图 | 模型调用是有延迟和失败的外部依赖，不是函数 |
| `step2_two_tools_choice.py` | 让模型自己选该调哪个工具 | `bind_tools` 返回的是**新对象**，不是原地改 |
| `step3_real_tools.py` | 真的去读文件 | 工具必须做参数校验（防路径穿越、目录白名单） |
| `step4_rag_demo.py` | 文档切块 + 向量检索 + 混合检索 | 光看字面（BM25）和光看语义（向量）都会翻车 |
| `rag_concepts_demo.py` | RAG 概念拆开演示 | recall@k 怎么算、喂假资料会怎么产生幻觉 |
| `step5_ingest_guard_demo.py` | **入库前把关**（本项目重点） | 垃圾进 = 幻觉出，必须在入口拦 |
| `step6_fastapi_service.py` | 包成 HTTP 服务 + 网页 | 模型/向量库只在启动时加载一次，绝不每请求重建 |
| `step6_guard_check.py` | 把关逻辑自检（不起服务也能跑） | 把"应该发生的事"写成断言，改坏了立刻报错 |

---

## 入库把关：五道关卡（实测结果）

10 份原始材料 → **放行 7 条，拦下 3 条**：

| 关卡 | 拦什么 | 本项目实测拦下 |
|---|---|---|
| 太短碎屑 | 少于 15 字的残片 | — |
| 完全重复 | 归一化后 md5 相同 | — |
| 近似重复 | 余弦相似度 > 0.90 | 客服话术库的同义改写版 |
| 个人隐私 | 手机号 / 身份证 / 银行卡 | 含手机号+身份证的工单导出 |
| 版本冲突 | 同一条款有新旧两版 | 2024 版"扣 30% 手续费"被 2026 版取代 |

> **修过一个真 bug**：版本冲突关一开始只按「文档名」分组，
> 结果 `退款政策.md` 里【已发货】和【未发货】两条完全不同的规定被当成"同一条的新旧版"，
> 互相挤掉，库里凭空少一条规则。
> 改成按 **(文档 + 条款)** 分组后修复 —— `step6_guard_check.py` 里有对应的回归断言。

访问 `/guard-report` 能看到"哪条被拦、为什么"，不用翻日志猜。

---

## 实测问答（真实调用 DeepSeek）

| 问题 | 回答 | 表现 |
|---|---|---|
| 已发货的订单退款要扣多少钱？ | 需扣除 10 元运费 | 命中正确条款，精排 10 分 |
| 还没发货的订单能全额退款吗？ | 可全额退，不扣费，24 小时到账 | 命中正确条款（修 bug 前答不出） |
| 支持分期付款吗？ | 资料里没有提到，我不知道 | **库里没有就直说，不瞎编** |

最后一行是关键：防幻觉不是靠"提示词写得好"，而是靠
**rerank 给出低分 → 不喂资料 → 明确拒答** 这条硬链路。

---

## 我踩过的坑（都写在代码注释里）

- **LangSmith 端点是 `api.smith.langchain.com`**，写成 `api.sm.` 连不通
- **FAISS 的 DLL 会被公司/学校的应用控制策略拦**，代码里做了自动降级到 numpy 的实现
- **embedding 模型要走 `HF_ENDPOINT=https://hf-mirror.com`**，直连 huggingface.co 超时
- **模型缓存别删**（默认 `~/.cache/fastembed`），第一次下载慢，之后秒开
- **跑练习脚本前关掉 trace**（`set LANGSMITH_TRACING=false`），否则每步都联网上报，会把脚本拖到卡死
- **服务启动时加载一次资源**：把模型写进全局对象，而不是每个请求里重建（这是服务化最容易犯的错）

---

## 技术栈

LangGraph 1.2.11 · langchain-core 1.6.3 · DeepSeek（`deepseek-chat`）·
bge-small-zh-v1.5（512 维中文 embedding）· FastAPI · uvicorn · numpy

检索是**向量检索 + 可选 BM25 混合 + RRF 融合 + 大模型 rerank**。

---

## 项目结构

```
rag-customer-service/
├── step1_real_llm_graph.py      # 接真模型
├── step2_two_tools_choice.py    # 工具选择
├── step3_real_tools.py          # 真读文件（带参数校验）
├── step4_rag_demo.py            # 切块 / 向量 / 混合检索
├── rag_concepts_demo.py         # RAG 概念演示
├── step5_ingest_guard_demo.py   # 入库五道关卡
├── step6_fastapi_service.py     # HTTP 服务 + 聊天网页
├── step6_guard_check.py         # 把关逻辑自检
├── step3_docs/ step4_docs/      # 示例知识库（含脏数据）
└── requirements.txt
```
