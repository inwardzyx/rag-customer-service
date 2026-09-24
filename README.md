# 客服知识库问答：**一个不会瞎说的 RAG 服务**

> **仓库地址**：<https://github.com/inwardzyx/rag-customer-service>

大多数 RAG demo 的目标是"答得出来"，这个项目的目标是 **"答不出来的时候敢说不知道"**。
完整链路（接模型 → 工具 → 检索 → 入库把关 → 精排 → HTTP 服务）都跑通了，
但重点在两件事：**垃圾数据不让它进库**、**库里没有就不许它编**。

> 知识库是 10 条虚构的电商客服条款（退款、发货、保修、税费、物流……），
> 其中 3 条是**故意混进去的脏数据**（旧版条款、同义改写的话术、客户手机号+身份证），
> 用来验证把关真的在工作：最终 **放行 7 条、拦下 3 条**。
>
> ⚠️ 一句实话：知识库现在**从文件读** —— `service.py` 通过 `kb/loader.py` 扫描 `docs/` 目录，
> `docs/kb/` 已审核、`docs/inbox/` 待审核；每个 `.md` 顶部 `---` 写元信息、`## 条款` 切块。
> 想加文档，往 `docs/` 扔一个 `.md` 就行，不用改代码。
> 10 条原始材料 → 放行 7 条、拦下 3 条，把关逻辑一字未改。

---

## 30 秒跑起来

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配密钥：复制模板成 .env，再填上真实值
copy .env.example .env          # Mac/Linux 用: cp .env.example .env
# 然后打开 .env，把 DEEPSEEK_API_KEY 换成你的真实 key。
# .env 已被 .gitignore 挡住，不会进 git；.env.example 里没有真密钥，可以提交。
# （为什么不用 setx？setx 是 Windows 专有命令，服务器上不存在，
#   而且设完要重开终端；.env 跟着项目走，clone 下来复制一份就能跑。）

# 3. 跑自检（不起服务，验证入库把关 + 拒答短路对不对）
python -m pytest tests/ -v
# ★ 这一步【不需要 DEEPSEEK_API_KEY】：自检只跑把关和检索，不调生成模型。
#   想顺便看"留了谁 / 拦了谁"：python -m pytest tests/ -v -s

# 4. 起服务
python service.py
```

打开 <http://127.0.0.1:8000> 是聊天页面，<http://127.0.0.1:8000/docs> 是自动生成的接口文档。

---

## 它长什么样

```
              ┌────────────── 启动时做一次 ──────────────┐
 知识库文档 ──→  入库把关（5 道关卡）→ embedding → 向量库 + BM25 索引
              └──────────────────────────────────────────┘
                                              │
 用户提问 ──→ 向量检索 ─┐
              BM25   ─┴─→ RRF 融合 → 粗捞 top-5 → 大模型精排 → 取 top-3
                                                      │
                                              ┌───────┴────────┐
                                         最高分 < 5        最高分 ≥ 5
                                              │                │
                                      直接拒答（不调模型）  看着资料作答 + 引用出处
```

**两道防线**：

| 防线 | 建在哪 | 拦什么 |
|---|---|---|
| 入库把关 | 数据**进来之前** | 脏数据（重复、隐私、旧版本）根本进不了库 |
| 拒答硬短路 | 答案**出去之前** | 库里没有相关资料时，一条都不喂给模型 |

---

## 拒答是怎么做到"硬"的

关键不在提示词写得多好，而在**低分时压根不调用生成模型**：

```python
if not ranked or ranked[0][2] < 5:        # 连最高分的候选都不到 5 分
    return ChatResponse(answer="知识库里没有能回答这个问题的资料，我不编。",
                        sources=[], knowledge_hit=False)
```

对比一下两种写法的差别：

| 写法 | 结果 |
|---|---|
| 硬塞一条低分资料进去，指望提示词让模型拒答 | 提示词会漏，**喂了假资料模型就会照着编** |
| 一条都不喂，直接返回（本项目） | 物理上不可能产生幻觉 —— 没东西可编 |

**关掉 rerank 也绕不过去。** 网页上那个「启用 rerank」复选框只是个**对比开关**，不是防线的开关。
关掉它，拒答改用**向量分**判断（余弦相似度，0~1）：

```python
# 该答的（7 题）向量分 0.6541 ~ 0.8425   ← 最低 0.6541
# 该拒的（5 题）向量分 0.2619 ~ 0.4373   ← 最高 0.4373
# 两边中间空着 0.22 的间隔，取中间值 0.55，离两边都留了余量
VEC_REJECT_THRESHOLD = 0.55
```

> 这里以前写的是"关掉 rerank 没分数可判断，只能照常生成" ——
> 那等于**取消勾选一下，整个拒答机制就没了**，而且 `knowledge_hit` 还是 `true`，
> 调用方还以为命中了。现在两条路都有拒答，`tests/` 里有断言专门盯着这个。

接口响应里有 `knowledge_hit` 字段，调用方能明确区分「答错了」和「拒绝作答」。
实测：命中时约 2.8 秒，拒答时约 1.3 秒（省掉了生成那一次调用）。

---

## 代码是怎么长出来的（`experiments/` 里保留了每一步）

| 文件 | 这一步解决什么 | 关键收获 |
|---|---|---|
| `experiments/step1_real_llm_graph.py` | 把真实大模型接进 LangGraph 图 | 模型调用是有延迟和失败的外部依赖，不是函数 |
| `experiments/step2_two_tools_choice.py` | 让模型自己选该调哪个工具 | `bind_tools` 返回的是**新对象**，不是原地改 |
| `experiments/step3_real_tools.py` | 真的去读文件 | 工具必须做参数校验（防路径穿越、目录白名单） |
| `experiments/step4_rag_demo.py` | 文档切块 + 向量检索 + 混合检索 | 光看字面（BM25）和光看语义（向量）都会翻车 |
| `experiments/rag_concepts_demo.py` | RAG 概念拆开演示 | recall@k 怎么算、喂假资料会怎么产生幻觉 |
| `experiments/step5_ingest_guard_demo.py` | **入库前把关** | 垃圾进 = 幻觉出，必须在入口拦 |
| `service.py` | 包成 HTTP 服务 + 网页 | 模型/向量库只在启动时加载一次，绝不每请求重建 |
| `tests/test_guard.py` | 把关 + 拒答的回归测试（不起服务也能跑） | 19 条 pytest 断言，改坏了立刻变红（见「变异测试」） |
| `kb/loader.py` + `docs/` | 知识库改成**从文件读**（kb/inbox 分目录） | 从"手敲 10 条常量"到"扔 md 就入库"；loader 可脱离模型独立单测 |
| `evalset/` | 20 题评测集（14 应答题 + 6 拒答题） | 拿到 Recall@5 / 漏答率 / 拒答准确率三个数字，"可度量"才成立 |
| `env_compat.py` | 绕开本机 DLL 被策略拦截的坑 | 见下方"踩过的坑" |

---

## 入库把关：五道关卡 + 版本冲突（实测结果）

10 条原始材料 → **放行 7 条，拦下 3 条**：

| 关卡 | 拦什么 | 本项目实测拦下 |
|---|---|---|
| ① 太短碎屑 | 少于 15 字的残片 | —（语料里没有，靠自造数据测） |
| ② 完全重复 | 归一化后 md5 相同 | —（语料里没有，靠自造数据测） |
| ③ 近似重复 | 余弦相似度 > 0.90 | 客服话术库的同义改写版 |
| ④ 个人隐私 | 手机号 / 身份证 / 银行卡 | 含手机号+身份证的工单导出 |
| ⑤ 缺少来源/日期 | `source` / `version` 为空 → 出事无法追溯 | —（语料里没有，靠自造数据测） |
| 版本冲突 | 同一条款有新旧两版 | 2024 版"扣 30% 手续费"被 2026 版取代 |

> **⑤ 是 2026-09-23 才补上的 —— 补的理由值得记一笔**：在此之前 `_guard()` 只有前 4 关 + 版本冲突，
> docstring 却一直写着"五道关卡"，即**文档说五关、实现只有四关**：读者以为有兜底，其实没有。
> 补它不是"和教学演示对齐" —— 知识库改成从 `docs/` 目录读之后（`kb/loader.py`），
> 入库入口被主动拓宽了，此时"能追溯来源"必须由把关自己保证，不能指望 loader 或人工。
>
> ⚠️ **别高估 ⑤**：它只保证 `source` / `version` 字段**非空**，不保证来源**可信** ——
> 自己填个 `source=官网帮助中心` 照样能过。所以 `docs/inbox/` 仍是「人工审核区」，审完才移进 `docs/kb/`。
>
> ★ **① ② ⑤ 这三关在这 10 条语料上从不触发**。也就是说：删掉它们，最终 7/3 的数字不变、
> 所有断言照样全绿 —— 等于没测。所以每一关都额外配了一条"自己造数据"的测试
> （`test_short_text_rejected` / `test_exact_duplicate_rejected` / `test_missing_source_rejected`）。

> **修过一个真 bug**：版本冲突关一开始只按「文档名」分组，
> 结果 `退款政策.md` 里【已发货】和【未发货】两条完全不同的规定被当成"同一条的新旧版"，
> 互相挤掉，库里凭空少一条规则。
> 改成按 **(文档 + 条款)** 分组后修复 —— `tests/test_guard.py` 里有对应的回归断言。

访问 `/guard-report` 能看到"哪条被拦、为什么"，不用翻日志猜。

**0.90 这个阈值不是拍脑袋定的**：同义改写块之间的余弦相似度实测是 0.9274（例如"扣 10 元运费"正版 vs 它的同义改写话术，两文本只有措辞不同），这个数字可复现。取 0.90 而不是 0.785 与 0.927 的精确中点 0.856，是故意**上浮留余量**——去重关宁严勿松，边界上的同义改写多拦一条无害，放进来一条才有害。不同义的块之间相似度明显更低（实测 0.55 起），不会误伤。

### 「写了断言」不等于「断言有用」——故意改坏给你看

验证办法叫**变异测试**（mutation testing）：故意把代码改坏，看测试会不会红。

```bash
# ① 把 DUP_THRESHOLD 从 0.90 改成 0.99（近似重复关形同虚设）
python -m pytest tests/ -q
# → FAILED tests/test_guard.py::test_rejected_duplicate - assert False
# → FAILED tests/test_guard.py::test_near_dup_after_earlier_rejection
# → 2 failed, 31 passed

# ② 把 VEC_REJECT_THRESHOLD 从 0.55 改成 0.95（该答的也被拒）
python -m pytest tests/ -q
# → FAILED tests/test_eval_set.py::test_no_answerable_missed   ← 评测集门禁也响了
# → FAILED tests/test_guard.py::test_hit_returns_sources
# → FAILED tests/test_guard.py::test_vec_threshold_separates_hit_and_miss
# → 3 failed, 30 passed

# ③ 把关卡5（缺少来源/日期）整段改成直通
python -m pytest tests/ -q
# → FAILED tests/test_guard.py::test_missing_source_rejected
# → FAILED tests/test_guard.py::test_guard_survives_missing_keys
# → 2 failed, 31 passed
#   ★ 恰好且只有这两条红 —— 说明守它的确实是这两条，没有别的测试在替它兜底
```

改坏了立刻变红，而且报错自带**具体数字**——这才是"回归断言"四个字的含义。

对比一下旧版：它是 `print("失败")`，退出码永远是 0，改坏了照样打印"全部通过 ✓"。
（`pytest` 那时甚至收集不到它，因为它一个 `test_` 函数都没有。）

---

## 实测问答（真实调用 DeepSeek）

| 问题 | 回答 | 表现 |
|---|---|---|
| 已发货的订单退款要扣多少钱？ | 需扣除 10 元运费 | 命中正确条款，精排 10 分 |
| 东西坏了能修吗？ | 保修期内非人为损坏可修，保修一年 | BM25 落空、靠向量检索命中，精排 8 分 |
| 支持分期付款吗？ | 知识库里没有能回答这个问题的资料，我不编 | `knowledge_hit=false`，**未调用生成模型** |

第二行说明为什么要混合检索，第三行说明为什么要拒答短路（宁可不答也别编）。

**第二行为什么 BM25 会落空？** 这里以前写错过，写成"文档里没有『修』字"——
其实《保修范围》那条里"修"出现了 **3 次**。真正的原因是**分词粒度**：

```python
jieba.cut_for_search("东西坏了能修吗")
# → ['东西', '坏', '了', '能', '修', '吗']          问题侧切出的是【单字】「修」「坏」

jieba.cut_for_search("商品享受一年整机保修，……人为损坏不在保修范围内。")
# → ['商品', '享受', '一年', '整机', '保修', '保修期', …, '损坏', '保修', '范围', …]
#   文档侧的「修」永远和「保」绑在一起，「坏」永远和「损」绑在一起
```

BM25 是**字面**匹配：它拿"修""坏"这两个单字去查，而文档里从来没有单独成词的"修"或"坏"，
所以一分不得。向量检索看的是语义，"坏了能修"和"保修"在向量空间里就是近的，于是命中了。
——这两路互补，谁都别丢。

---

## 我踩过的坑（都写在代码注释里）

- **LangSmith 端点是 `api.smith.langchain.com`**，写成 `api.sm.` 连不通
- **embedding 模型要走 `HF_ENDPOINT=https://hf-mirror.com`**，直连 huggingface.co 超时
- **模型缓存别删**（默认 `~/.cache/fastembed`，可用环境变量 `FASTEMBED_CACHE_PATH` 改位置），
  第一次下载慢，之后秒开
- **跑练习脚本前关掉 trace**（`set LANGSMITH_TRACING=false`），否则每步都联网上报，会把脚本拖到卡死
- **服务启动时加载一次资源**：把模型写进全局对象，而不是每个请求里重建（服务化最容易犯的错）
- **本机 DLL 会被应用控制策略拦截**：faiss 和 mmh3 都中招
  （`DLL load failed: 应用程序控制策略已阻止此文件`）。
  faiss 干脆没引入 —— 向量检索直接用 **numpy 全量点乘**，7 条语料完全够，
  上千条再考虑换 HNSW/faiss；
  mmh3 在 `env_compat.py` 里给了**纯 Python 的 murmur3 实现**兜底
  —— 它算出来的值和 C 版逐位一致（文件内有官方测试向量自检），不是凑数返回 0
- **LLM 要懒加载**：`ChatDeepSeek(...)` 在**构造时**就校验 `DEEPSEEK_API_KEY`，
  没配 key 直接抛 `ValidationError`。写在 `startup()` 里的话，
  只想跑入库把关自检（压根用不到模型）的人也会被卡住 ——
  面试官 clone 下来第一步就失败，CI 也跑不了。现在改成 `@property` 用到才建。

---

## 技术栈

langchain-core 1.6.3 · DeepSeek（`deepseek-chat`）·
bge-small-zh-v1.5（512 维中文 embedding）· FastAPI · uvicorn · jieba · rank-bm25 · numpy · pytest

> **关于 LangGraph**：它只用在 `experiments/step1-3`（学工具调用的那几步）。
> `service.py` 里**一行 LangGraph 都没有** —— 这个服务是线性的（检索 → 精排 → 生成），
> 套个图只会增加复杂度。知道什么时候**不该用**一个框架，和会用它是两回事。

检索是**向量检索 + BM25 混合 + RRF 融合 + 大模型精排**。

> **为什么用 RRF 融合而不是把两路分数相加**：向量分在 0~1 之间，BM25 分能到十几，
> 直接相加 BM25 会把向量完全淹没。不同量纲的分数不能相加 —— 这是混合检索最常见的坑。
> RRF 只看名次（`1/(60+名次)`），天然免疫量纲问题。

---

## 项目结构

```
rag-customer-service/
├── service.py                   # HTTP 服务 + 聊天网页（主交付物）
├── env_compat.py                # 环境兼容层（mmh3 的纯 Python 兜底）
├── requirements.txt
├── .github/
│   └── workflows/ci.yml         # CI：快 job（纯解析，秒级）+ 慢 job（装模型 + 质量门）
├── kb/
│   └── loader.py                # 知识库加载器：扫 docs/ 树 → 解析元信息 → 切条款
├── docs/                        # 知识库（服务真正读的目录）
│   ├── kb/                      #   已审核干净条款（6 个 .md → 7 条）
│   └── inbox/                   #   待审核脏副本（3 个 .md，会被把关拦下）
├── evalset/
│   ├── questions.json           # 20 题评测集（14 应答题 + 6 拒答题）
│   ├── run_eval.py              # 评测运行器（离线层默认不调模型）
│   └── report.md                # 实测结果（随代码提交，改阈值时 diff 可见）
├── experiments/                 # 每一步的长成过程（教学脚本，可独立运行）
│   ├── step1_real_llm_graph.py
│   ├── step2_two_tools_choice.py
│   ├── step3_real_tools.py
│   ├── step4_rag_demo.py
│   ├── rag_concepts_demo.py
│   ├── step5_ingest_guard_demo.py
│   └── step3_docs/ step4_docs/  # 示例知识库（含脏数据）
└── tests/                       # 共 33 条，全是 pytest 断言（旧版是 print 自检，退出码永远 0）
    ├── test_guard.py            # 把关 + 拒答 + PII 脱敏 + XSS 回归（19 条）
    ├── test_loader.py           # kb/loader 解析单测（9 条，不碰模型 → CI 快 job 跑它）
    └── test_eval_set.py         # 评测集自洽 + 离线质量门禁（5 条）
```

---

## 评测集：20 题，把"可度量"落到实处

`evalset/questions.json` —— 20 题 = **14 应答题**（7 条入库条款 × 2 种问法：关键词式 / 口语化）+ **6 拒答题**（4 道清晰 + 2 道**边界题**）。

```bash
# 仓库根目录，离线层：不调生成模型，不需要 API key
python evalset/run_eval.py             # 打印报告
python evalset/run_eval.py --report    # 结果写进 evalset/report.md
python evalset/run_eval.py --with-llm  # 在线层：走 rerank（需 DEEPSEEK_API_KEY）
```

实测数字（`evalset/report.md`）：

| 指标 | 数值 | 含义 |
|---|---|---|
| Recall@5 | **14/14** | 应答题的 gold 条款都进了粗捞前 5 |
| 漏答率 | **0/14** | 该答却拒答 = 0（防"把阈值调严显得准"的自证式作弊） |
| 拒答准确率 | **5/6** | 6 道该拒的有 5 道拒了 |

> ★ **两层分开报，不合并**：离线层（`use_rerank=false`）拒答只靠向量分 0.55；产品默认走 rerank（精排分 < 5 才拒答）。两层的拒答数**不是包含关系**，所以并列。
>
> ★ 唯一失手是边界题 q20「退货运费谁承担？」—— 它语义挨着退款条款、向量分 **0.6789** 越过 0.55，于是**纯向量路把它放行了**。这正是"关掉 rerank 会变糙"的实锤，也是产品默认开 rerank 的理由。**不为此调阈值**（调阈值只会让"该拒的"好看、"该答的"更危险）。

---

---

## CI：每次 push 自动跑，把"漏答 0"变成质量门

`.github/workflows/ci.yml` —— 有意拆成两个 job：

| job | 装什么 | 跑什么 | 实测耗时 |
|---|---|---|---|
| **fast** | 只装 `pytest` | `tests/test_loader.py`（9 条） | **0.03 秒** |
| **full** | `requirements.txt` + 下载 92MB 模型 | `test_guard.py` + `test_eval_set.py`（24 条） | 15 秒起（首次还要下模型） |

**为什么拆**：两类测试成本差约 500 倍（实测 0.03s vs 15s）。合成一个 job 的话，改一行加载器也要等模型下载完才知道对不对。

**fast job 为什么敢只装 pytest**：`kb/loader.py` 只 import `re` 和 `pathlib`。这不是"我觉得"，是验过的 —— 拿一个**没装 fastembed / onnxruntime / jieba** 的解释器去 import 它，连带加载的第三方模块为**零**；再用只装了 pytest 的隔离环境跑，9 条全绿。对照实验：同一个"穷"解释器跑 `test_guard.py` 会直接 `ModuleNotFoundError: No module named 'dotenv'` —— 反过来证明慢 job 确实必须装全套。

**两个真坑**：
- `HF_ENDPOINT` 必须覆盖成 `https://huggingface.co`：`service.py` 用 `os.environ.setdefault` 把默认值设成国内镜像 `hf-mirror.com`，而 GitHub 的 runner 在海外，走国内镜像大概率慢或超时。正因为源码用的是 `setdefault`，设个环境变量即可覆盖 —— **零改代码**。
- `python-version: "3.14"` 的**引号不能省**：不加引号 YAML 会把它当浮点数，变成 `3.1`。（已用 PyYAML 校验过类型确实是字符串。）

**刻意不配 `DEEPSEEK_API_KEY`**：`test_guard.py` 用 `monkeypatch` 注入假模型，一条都不调真生成模型。所以 CI 里没有任何密钥 —— 也就没有"密钥泄漏进 CI 日志"这条风险。

**顺带的收益**：慢 job 的 `pip install -r requirements.txt` 就是 Dockerfile 里 pip 层的**预演**（同为 Linux + Python 3.14）。它红 = Docker 也会红，而在 CI 里改（2 分钟）远比在容器里调（半天）便宜。

> 顺带纠正一个我们先前**双方都当真**的错误结论：一度以为"numpy / onnxruntime 在 Linux + py3.14 没有 wheel"。
> 实测是**假的**，那是被 `pip install --platform ... --abi cp314` 的**字面匹配**骗的 ——
> `--abi cp314` 会**替换**掉默认 ABI 列表，从而把 `abi3` 轮子排除在外。
> 直接读 PyPI 上真实的 wheel 文件名才对：mmh3 5.3.0 / numpy 2.5.3 / onnxruntime 1.30.0 / pydantic-core 2.49.0
> 都有 cp314 的 manylinux 轮子；tokenizers 0.23.2 没有 cp314 专用轮子，但有 `cp310-abi3`（稳定 ABI，3.14 也能装）。
> **教训**：`--platform` / `--abi` 是"清单替换"而不是"追加"，拿它做兼容性探测会造出假阴性。

---

## 还没做的（以及为什么现在不做）

主动写出来，比被面试官挖出来强。

| 缺口 | 现状 | 为什么现在不做 |
|---|---|---|
| **没部署** | 只能本地跑 | 免费平台要塞 `DEEPSEEK_API_KEY`，**别人点开就能刷你的 key**；平台一 sleep 就 502，比没链接更糟。替代方案：录 60-90 秒 GIF 放 README |
| **没有 Dockerfile** | 只能本地跑 | 顺序是有意的：先让 CI 慢 job 把「Linux + py3.14 装依赖」跑通，Dockerfile 就只剩打包这一件事 |
| **语料是虚构的** | 10 条自己写的电商条款 | 换真实语料会让"0.90 是实测的"更难解释。**可控的假数据 > 不可控的真数据**，这个项目要证明的是机制，不是数据 |

### 下一步的顺序

1. ~~服务改成读 `docs/*.md` 再切块~~ ✅ 已完成（`kb/loader.py` + `docs/`）
2. ~~20 题小评测集（含**漏答率**）~~ ✅ 已完成（`evalset/`，离线层 Recall@5 14/14、漏答 0/14）
3. ~~网页 XSS 修复（`innerHTML` → `textContent`）~~ ✅ 已完成（commit `add5b3e`）
4. ~~GitHub Actions CI~~ ✅ 已完成（`.github/workflows/ci.yml`，快慢双 job）
5. ~~补上缺失的关卡5「缺少来源/日期」~~ ✅ 已完成（顺手修的，见上面「入库把关」）
6. ~~结构化日志~~ ✅ 已完成（`service.py` / `env_compat.py` / `evalset/run_eval.py` 运行时 `print` 换成 `logging`，入口处 `basicConfig` 保住输出；`experiments/` 教学脚本保留 `print` —— 那里逐行打印是刻意的）
7. **Dockerfile** —— 92MB 模型的镜像怎么瘦身（CI 慢 job 已预先验证 Linux + py3.14 装得上）
8. 录一段 GIF 放 README 顶部 —— 30 分钟，零风险，效果接近一个在线链接
