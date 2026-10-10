# -*- coding: utf-8 -*-
"""heldout/make_pack.py —— 生成 held-out 出题包

跑法：
    D:\\Python-project\\.venv\\Scripts\\python.exe evalset\\heldout\\make_pack.py

产出三个文件（都在 evalset/heldout/ 下）：
    1. source_policy.md——【给题目作者看的】全库校规原文（docs/kb 下全部现行制度），按条款排版，
       **不含任何 chunk / 向量 / 检索信息**，也不含现有 20 题的任何痕迹。
    2. questions_template.json —— 空模板，题目作者填 question / type / gold /
       why_forced_to_refuse / note 五个字段。
    3. README.md—— 给题目作者的说明书（含规则与反例）。

★ 为什么必须"只给原文、不给 chunk"：
    出题人如果看过 chunk，题面会不自觉地复用 chunk 里的措辞。
    而真实用户提问的措辞分布 ≠ chunk 措辞分布——
    这样出出来的题，测的是"检索能不能认出换了个说法的同一条款"，
    这正是这个项目唯一还没被测过的东西。

★ 为什么必须"作者没读过语料"之外还要强制写 gold：
    没有 gold 就没法自动判分，只能人工看，那这套题集就无法重复跑。
    gold 写 [文件名, 条款名] 两级，和现有 questions.json 同格式。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
KB = ROOT / "docs" / "kb"

# 出题时要避开的条款：**直接从现有 questions.json 取 gold**（单一真源）。
# ★ 这里以前是手抄的 9 条，q18 从 refuse 改成 answer 之后没跟上 ——
#   「学生资助工作实施办法.md｜第十五条」会被标成 [未覆盖]，
#   出题人于是优先给它出题，造出一道和现有题重复的题，这套 held-out 当场贬值。
#   ⇒ 判据与真源同源：改 questions.json，这里自动跟着改。
#     （这个仓库反复踩过"两处各写一份判据"的坑，见 loader 与体检脚本那对常量。）
def _already_covered() -> set[tuple[str, str]]:
    p = ROOT / "evalset" / "questions.json"
    qs = json.loads(p.read_text(encoding="utf-8"))
    got = {(q["gold"][0], q["gold"][1]) for q in qs if q.get("gold")}
    assert len(got) >= 10, f"只从 questions.json 读到 {len(got)} 条 gold，去看一眼格式"
    return got


ALREADY_COVERED = _already_covered()


def _question_stats() -> tuple[int, int, int]:
    """现有题库的 (总题数, answer 数, refuse 数) —— 同样从 questions.json 取。

    ★ 为什么要抽出来：上一版把覆盖条款数改成了 `{len(ALREADY_COVERED)}` 动态算，
      却把"20 题（15 answer + 5 refuse）"留成了手抄 —— 同一个提交里
      一半动态一半手抄，改题量时头部会跟着变、正文不会。
      这个坑与本文件顶部的 ALREADY_COVERED 是同一个，别再抄第三份。
    """
    qs = json.loads((ROOT / "evalset" / "questions.json").read_text(encoding="utf-8"))
    a = sum(1 for q in qs if q.get("type") == "answer")
    r = sum(1 for q in qs if q.get("type") == "refuse")
    return len(qs), a, r


N_Q, N_ANS, N_REF = _question_stats()


def parse_doc(path: Path):
    """把一份 kb .md 拆成 [(条款名, 正文)]，顺手把元信息丢掉。"""
    text = path.read_text(encoding="utf-8")
    # 去掉 --- 之间的元信息（doc/version/source），出题人不该看到 source URL
    text = re.sub(r"^---$.*?^---$", "", text, flags=re.S | re.M)
    parts = re.split(r"^##\s+", text, flags=re.M)
    clauses = []
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        name = head.strip()
        body = body.strip()
        if body:
            clauses.append((name, body))
    return clauses


def main():
    docs = sorted(KB.glob("*.md"))
    if not docs:
        raise SystemExit("docs/kb/ 下没有 .md，先把校规文档放进去")

    all_clauses = []
    for p in docs:
        for name, body in parse_doc(p):
            all_clauses.append((p.name, name, body))

    uncovered = [(d, n, b) for d, n, b in all_clauses if (d, n) not in ALREADY_COVERED]

    # ---------- ① 出题底本 ----------
    lines = [
        "# 校规原文（出题底本）",
        "",
        f"> ★ 生成时间 {datetime.now():%Y-%m-%d %H:%M}"
        f"；来源 docs/kb 下 {len(docs)} 份现行制度，共 {len(all_clauses)} 条"
        f"；其中 {len(uncovered)} 条现有评测集没覆盖。",
        "",
        "> 这些文件是**知识库的原始来源**。你的任务是假装你只懂学校规矩、",
        "> 从来没见过这个 RAG 系统，然后像真实学生那样提问。",
        "",
        "> ⚠️ **不要去仓库里看 `evalset/questions.json`**（现有 20 题）。",
        "> 看了你的题会不自觉地贴那边措辞，这套题就废了。",
        "",
        f"> 合计 {len(all_clauses)} 条，其中 {len(uncovered)} 条现有评测集**没覆盖到**——",
        "> **优先从没覆盖的条款里出题**（下面每条都标了 `[已覆盖]` / `[未覆盖]`）。",
        "",
        "---",
        "",
    ]
    cur = None
    for doc, name, body in all_clauses:
        if doc != cur:
            cur = doc
            lines += [f"## 📄 {doc}", ""]
        flag = "已覆盖" if (doc, name) in ALREADY_COVERED else "未覆盖"
        lines += [f"### {name}  `[{flag}]`", "", body, ""]
    (OUT / "source_policy.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- ② 空模板 ----------
    template = {
        "_说明": [
            "把这个文件填完交给出题人所在项目的人（填questions 里那三个字段）。",
            "type 只有两个取值：answer（库里答得出来）/ refuse（库里答不出来）。",
            "gold 填 [文件名, 条款名]；refuse 题填 null。",
            "why_forced_to_refuse 只在 refuse 题填，写清'为什么这道题库里答不出来'。",
            "★ 写不出来就别硬写——一道说清理由的 refuse 题比十道含糊的强。",
        ],
        "questions": [
            {
                "id": "h1",
                "type": "answer",
                "question": "",
                "gold": ["", ""],
                "note": "",
            }
        ],
    }
    (OUT / "questions_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- ③给题目作者的说明书 ----------
    readme = f"""# held-out 出题包 —— 给出题人看的

## 你是谁，为什么要你出题

这是一个 RAG 问答系统的评测题集。**现在的 20 道题是项目作者自己出的**，
所以那套题有个结构性毛病：**同一批题既用来定阈值、又用来报准确率**，
这在方法论上叫 tuning set，不算独立验证。

**你出的这批题是held-out（独立验证集）**：作者没看过、你也没看过他的chunk，
所以这批题的答案对双方都是"新的"。

## 三条铁规矩

### 1. 只看 `source_policy.md`，不要去看仓库

`source_policy.md` 是全库 {len(docs)} 份现行制度的原文。**这是你唯一该看的材料。**

-❌ 不要打开 `evalset/questions.json`（现有 20 题）
- ❌ 不要打开 `docs/` 或 `kb/` 目录（那是切好的 chunk，看了会泄漏措辞）
- ❌ 不要打开 README（它举的例子会污染你的出题）

**为什么**：如果你的题面和 chunk 里的措辞重合度高，
那这道题测的就是"检索认不认得原话"，而不是"认不认得换个说法"。
真实用户不会照着条款原文提问，他们只会用自己的话。

### 2. 用真实学生的口气提问

- ✅「要请一个礼拜的假得谁批啊」—— 口语、有缩写、有语气词
- ✅「学长说要写检讨才能拿处分 真的吗」
- ❌「请假一周的审批权限是什么」—— 这是条款的标题，不是学生的问法

**刻意制造用词差异**：条款写"请假"，你写"请个假"；
条款写"开除学籍"，你写"直接开除"；条款写"擅自"，你写"没打招呼就"。
真实用户和条款原文的措辞差距，就是这个系统要过的关。

### 3. 每道 refuse 题必须写清"为什么库里答不出来"

光标`type: refuse` 没用——库里有 {len(all_clauses)} 条条款，什么都能沾上边。

★ **口径说明（别被数字绕晕）**：这里说 {len(all_clauses)} 条，是**从文件切出来的条款数**；
而检索时能召回的**不是这个数** —— 入库关卡还会拦掉一批
（太短碎屑 / 废止声明 / 完全重复 / 近似重复）。
★ 所以出题时按本文件的条款数走（底本里全是原文，够你选），
但判"库里到底有没有"要按**实际入库的块数**走 ——
两者差的那几条正是被关卡拦掉的，在底本里能看见、系统却召不回。
分不清就会出一道"底本里有、库里其实没有"的题，
那题会被系统正确拒答，而你会误以为它拒答错了。
（实时的入库块数看 `python evalset/run_eval.py` 启动时打的"入库 N 块"。）

你要写的是**"库里哪一条都没提这件事"**，例如：
- 「图书馆几点关门」——全部校规里没有校历/场馆开放信息
- 「学费多少钱」——校规里没有"学费标准"这个值。★ 注意：资助类条款里有
  "减免一学年学费 / 减免一半"这类**额度**，沾边 ≠ 答得出来
- 「宿舍晚上几点熄灯」——有条款要求"遵守学校作息时间"，但不给具体时刻

★ **反例（别学这个）**：「怎么申请助学贷款」**已经不算 refuse 题了**。
  语料扩容后库里真有答案（`学生资助工作实施办法`第十五条：生源地信用助学贷款、
  不超过 20000 元/生/年），它已被改成 answer 题。
  ⇒ **拒答题的边界会随语料变**。出题前请对着 `source_policy.md` 逐条确认
    "库里真的没有用户要的那个值"，而不是"感觉沾不上边"。

如果一条你写不出"为什么库里没有"，那它很可能不是 refuse 题。

## 出几道、出什么

**你要出的目标：20 道**（14 道 answer + 6 道 refuse —— ★ 这是**目标配比**，
**不是**现有题库的配比，别拿它去对下面那句"现有"），
再加 **5–8 道 multi-hop 边界题**（见下）。

**优先从标`[未覆盖]` 的条款出题**——现有题库共 **{N_Q} 题**
（{N_ANS} answer + {N_REF} refuse，实时取自 `evalset/questions.json`），
只覆盖了 **{len(ALREADY_COVERED)} 个**条款，剩下
**{len(all_clauses) - len(ALREADY_COVERED)} 条**从来没被问过。

### answer 题（14 道 —— 目标，非现有）

覆盖尽量多的不同条款。`gold` 填 `[文件名, 条款名]`，照这个样子写：

```json
PLACEHOLDER_ANSWER_EXAMPLE
```

> ★ `gold` 要填到**条款级**，不是文档级。填错了这题就没法自动判分。

### multi-hop 边界题（5–8 道）—— 这类最有价值，也最容易出

这类题**必须结合两份文档、或同一文档里跨条款才能答完整**，
纯靠 top-1 单块答不全。而当前系统的 gate 只看 top-1 最高分，
所以这类题**很可能被误拒** —— 那正是我们要测出来的东西。

例子（不用照抄，换你自己的）：
- 「请假没批就回来了，会怎么处理？」—— 请销假制度（假该销没销）
  ＋ 纪律处分（旷课/擅自离校怎么处分）
- 「我在外面租房住， 学校会怎么管？」—— 纪律处分第十八条（三）擅自校外租房

这类题如果你写不出确定的 gold，就写进 note 说明"这题我认为是跨条款的，
但具体该引哪条我不确定"—— **这种诚实标注比硬填一个错 gold 有用得多**。

### refuse 题（6 道 —— 目标，非现有）

选那些**校规压根没提**的事：场馆开放、食堂、宿舍熄灯、缴费、办证、补办、校历。
`gold` 填 `null`，并填 `why_forced_to_refuse`。

## 交回什么

填好 `questions_template.json`（另存为 `questions.json`），连同你的
**出题说明**（在文件顶部加几行，写你从什么角度出的题、有没有拿不准的地方）。

## 判分口径（出题人不需要关心，但你知道一下）

- answer 题：看 `gold` 条款有没有进检索 top-5（Recall@5），**不看答案文字**。
  为什么：答案由大模型生成，逐字比对没有意义；要评生成质量需要人工打分，
  那是另一件事。
- refuse 题：看 `knowledge_hit` 是不是 `false`。是 `false` 算拒对。
- ★ 本轮**只看这两个检索层指标**，不评"答案写得对不对"——
  当前系统还没做答案级判分，这一点在 README 里是明说的，别假装有。
"""
    # f-string 里不能直接写 JSON 花括号（会被当格式符），用占位符替换
    answer_example = json.dumps(
        {
            "id": "h1",
            "type": "answer",
            "question": "要是在考场上跟别人换卷子抄，会怎么处理？",
            "gold": ["学生纪律处分管理规定.md", "第三章-第十条（十）"],
        },
        ensure_ascii=False,
        indent=2,
    )
    readme = readme.replace("PLACEHOLDER_ANSWER_EXAMPLE", answer_example)
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"条款总数 {len(all_clauses)}，其中未覆盖 {len(uncovered)}")
    print(f"出好了：\n  {OUT/'source_policy.md'}\n  {OUT/'questions_template.json'}\n  {OUT/'README.md'}")

    # ---------- 自检：底本不能泄漏检索信息，也不能含现有题目 ----------
    # 为什么必须查：出题底本一旦混进 chunk 措辞或现有 20 题，
    # 这套 held-out 就退化成第二份 tuning set，白做。
    src = (OUT / "source_policy.md").read_text(encoding="utf-8")
    banned = ["chunk", "向量", "BM25", "RRF", "精排", "Recall", "rerank",
              "0.55", "source:", "version:", "embedding", "top-5"]
    leaked = [w for w in banned if w in src]
    if leaked:
        raise SystemExit(f"底本泄漏了检索信息：{leaked}")

    old = ROOT / "evalset" / "questions.json"
    if old.exists():
        qs = json.loads(old.read_text(encoding="utf-8"))
        dup = [q["id"] for q in qs if q.get("question") and q["question"] in src]
        if dup:
            raise SystemExit(f"底本里混进了现有题目：{dup}")
    print("自检通过：底本无检索信息泄漏，未混入现有题目")


if __name__ == "__main__":
    main()