# `_tasks/` 协议 v3 补件：断言二分 + 快照 pin + 对账器

> 2026-10-10 由 DSH 起草，基于 cc 对 v2 的评审（`D:\dsh-work\_out_collab_protocol.md`）。
> **这不是新协议**，是给现有 v2（`dispatch.sh` + `_reviews/`）补最后一块。
> v2 已经解决的：盲派、互不知情、不转述、不覆盖、字节判据。别推翻它们。

---

## 〇、这一份是被一次真实事故逼出来的

cc 在审本协议时，读 `D:\dsh-work\_out_collab_protocol.md`，报告：

> "里面只有一行 `[claude-code:unrecognized_model]`——净字节 0 = **没跑起来**"

**但那个文件当时有 178 行、19.5 KB 的完整内容。** 真实原因：它是被一个后台任务在**结束时**
才重定向落盘的，cc 读它时它正好是空的/半成品。

⇒ 由此推出一个错误结论（"cc 自动化派活路线没跑起来"）。

**这不是 cc 粗心，是协议缺一块**：`_reviews/README.md:37` 那条 `< 500 B = 没跑起来`
用的是**字节数**。而下面第三节的实测会看到：**字节数一字不差、内容已经变了**。

---

## 一、四个新字段

### 卡侧（`_tasks/TEMPLATE.md` 增）

```yaml
可跑判据:                 # ★ 缺这个字段的卡，dispatch.sh 直接拒收
  cmd: "D:\\Python-project\\.venv\\Scripts\\python.exe _tasks/probes/corpus_stats.py"
  accept: "输出含『截断丢字 8 处（丢 4195 字）』"
  run_by: dsh             # 默认 dsh（唯一稳定有执行通道的一方）
  落盘位: "_reviews/<name>.claims.json"
通道握手:                 # 每次派活**现场**探一次，不许写进户籍
  cc:  "<本会话能不能跑 git/python>"
  dsh: "dsh.sh --doctor → 期望出现 416"
```

### 审侧（`_reviews/<agent>-<slug>.claims.json`）

```yaml
claims:
  - id: C1
    type: fact            # fact | judgment  ★ 必须二选一
    anchor: "kb/loader.py:274-279"
    cmd: "..."            # fact 必填：没有命令的"事实"其实是判断
    expected: "..."
    status: unverified    # green | red | unverified —— actual 由脚本生成，禁止手抄
  - id: C2
    type: judgment
    would_flip_if: "什么证据会翻转它"     # 填不出来 ⇒ 这是偏好，降权
blind_spots:              # ★ 把"我需要看 X"从结尾客套变成提交前必填
  - claim_id: C4
    need: "..."
    obtain_by: "命令 / 谁提供"
    verdict_if_missing: hold       # confirm | retract | hold —— 必填，专治防御性堆砌
searched: "我查过哪些路径/命令"        # blind_spots 为空时必须附它
snapshot: {}              # 由 `check_claims.py pin` 写入，审稿人自己不许编
```

---

## 二、对账器：`_tasks/check_claims.py`

```bash
# ① 读之前先钉快照（审稿人 / 派活方都行）
<py> _tasks/check_claims.py pin  _reviews/<name>.claims.json <要审的文件...>

# ② 任何人（含作者）事后对账：schema + 快照漂移 + 重跑 fact 断言
<py> _tasks/check_claims.py check _reviews/<name>.claims.json
```

**退出码就是裁决**：`0` 全绿 / `1` schema 违规或有红 / `2` 有快照漂移（**结论不可用于决策**）。

它替掉的是"三方隔空争论"，换上的是 `exit code + stdout diff`。机器不投票。
但它**只保证"已想到的断言"可信**——三个同模型 agent 都想不到的坑，它管不着。
共同盲区只能靠世界暴露（真跑崩、用户不接受、时间让结论过期）。

---

## 三、fact / judgment 二分 = 递归的终止条件

"审稿人也要被审"不会无限递归，条件是**断言先分类**：

| 类 | 终止于 | 规则 |
|---|---|---|
| `fact` | **机器** | ≥2 方在**同 env**（解释器路径 + commit + 文件哈希）重跑一致即闭合 |
| `judgment` | **账本，不闭合** | 进 `open_disagreement`，记录、不阻塞、下个里程碑过期重看 |

**"谁审 meta-reviewer"的答案：没有人，因为不存在 meta 层。**
审稿人的产出结构上只有两种——可重跑的证据块、或判断。闭环终点是机器或账本。

同 env 下两方重跑不一致 ⇒ **不是评审问题，是 flaky check**，进隔离区、相关断言降级"不可判定"。
被审方的反驳义务：**必须挑 severity 最高的 fact 断言重跑**；若最高等级只有 judgment，
就交"降级请求 + 理由"，对方机械接受、不许回嘴。**这就是什么时候停。**

---

## 四、实测演示（2026-10-10，真跑通）

```bash
# 探针自身可用
$ <py> _tasks/probes/corpus_stats.py
截断丢字 8 处（丢 4195 字）
剥离网页页脚 2 处（丢 563 字）
ALREADY_COVERED 10 == gold 去重 10

# pin → check
$ <py> _tasks/check_claims.py check _reviews/dsh-collab-protocol.claims.json
【1】schema          ✔ 通过
【2】快照漂移        ✔ 与 pin 时一致
【3】fact 重跑       ✔ C1 green   ✔ C2 green
【4】judgment        · C3 记账，不判定
【5】采购单          · C4 缺「WorkBuddy 能力清单」⇒ hold
判定：✔ 全绿                                                   EXIT=0

# 触发漂移：重新生成一次底本（只变了时间戳）
$ <py> evalset/heldout/make_pack.py
$ <py> _tasks/check_claims.py check _reviews/dsh-collab-protocol.claims.json
【2】快照漂移        ✘ source_policy.md: pin 的是 af85cc16(106336B)
                       现在 f6cdffa9(106336B) ⇒ 读到的不是这一版
判定：✘ 有快照漂移 —— 这份结论不可用于决策                    EXIT=2
```

★ **字节数完全一样（106336 B），只有哈希变了。**
所以"用字节数判断有没有跑成"这条老规矩，**抓不到"读的是半成品/旧版本"这一类**。
生成物（`source_policy.md` 这种带时间戳的）尤其必须先 pin 再读。

---

## 五、仍然空着的一格：**谁跑实验**

今晚两张卡的双份评审（`cc-review-readme-table.md` / `dsh-review-readme-table.md`）：
共同命中 4 条、cc 独有 ~9 条、DSH 独有 ~3 条、互相矛盾 1 条。
**两份都自发了"我无法实证"节，都声明没跑实验——边界完全重合。**
其中 cc 还写明"`clean_corpus.py` 干跑即可、无需 key"，**依然没有人跑**。

⇒ 当前机制最大的洞不是"回声"，是**没人执行**。
补法（v3 已内置一半）：卡的 `可跑判据.run_by` 字段 + check 的【5】采购单。
**没跑 = 记 `unverified`，永远不许算"通过"。**

---

## 六、待办（我没做，留给下一轮）

1. 把 `可跑判据` 做成 `dispatch.sh` 的**入口门**（缺字段拒收）。
2. `-p` 换成带 `--resume <session-id>` 的调用（本机未验证，能省 ~21K 冷启动/轮）。
3. `_reviews/prose/` 隔离区：没有合法 `snapshot` + `claims` 的评审文件不参与决策。
4. 统一两套派活协议（`D:\dsh-work\_out_*.md` 与 `_reviews/`）—— 现在并存，
   这本身就是"两处各写一份"在**流程层**的复现。
5. `ci.yml` 加一条：`check_claims.py check` 对 `_reviews/*.claims.json` 全绿（非孤儿化，
   否则这个对账器又是一条"写在无人经过的路上的守卫"）。
