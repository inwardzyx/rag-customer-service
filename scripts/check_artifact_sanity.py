"""产物合理性下界检查：挡住"坏环境下跑出来的结果"覆盖好产物。

为什么需要它（2026-10-11，一起真实事故）
------------------------------------------------
`evalset/report-vector.md` 被一次**坏环境下跑的**评测覆盖成：

    应答题数：15  Recall@5：0/15
    阈值余量：应答题最低向量分 0.4240／拒答题最高向量分 0.4240

两个信号摆在那儿：**全盘召不回**（Recall 0/15），以及
**应答题最低分 == 拒答题最高分 == 0.4240**（两个不同分布的极值撞成同一个常数）。

⇒ 这个文件是**别人/别轮次据此写文档的基准**，被悄悄改掉的后果是
  README 的数字跟着一起变成错的（这次是对账器抓到才发现的）。

★ 病根不是"忘了检查"，是**没有任何东西在说"这次跑出来的数不合理"**：
  run_eval.py 自己不会报错 —— 它忠实报告了它算出来的东西。

## 为什么这个检查不派 agent、也不看 git

它必须满足三条才能挂进pre-push：
  1. **零 token、毫秒级** ⇒ 纯读文件 + 正则
  2. **不依赖 git**（坏产物可能在任何一次运行后产生，与提交无关）
  3. **判据能现算** ⇒ 下界来自"数学上不可能"，不是"我觉得太低"

## ★ 判据怎么来的（每条都必须能解释为什么是这个数）

| 判据 | 下界 | 依据 |
|---|---|---|
| **清单反查**（判据 0） | 清单 ⊇ 磁盘实际产物 | 漏登记的那份**永远不会被检查**。实测踩过：上一版清单少列了 `report-rerank-main-norw.md` |
| Recall@5 > 0 | > 0 | 15/44 道题里有 gold 与问题字面重合（见 README 自评第 2 条）。全 0 意味着**不是检索差，是 embedding 没加载/语料为空** |
| 应答题最低向量分 ≠ 拒答题最高向量分 | 不相等即可 | 两个是**不同题目集合**上的极值，正常必不相等。相等 ⇒ 分数没被逐题区分（常量化 / 取错字段 / 没跑） |
| 分母= 题目集合大小 | 相等 | 产物里的"应答题数"必须等于它自己 Recall 的分母，不等就是字段错位 |
| 拒答准确率分母 = 拒答题数 | 相等 | 同上 |

⚠️ **已知抓不到的坏产物**（写下来，免得假装能抓）：
  - **Recall@5 掉了但不为 0** —— 判据 1 只挡"全 0"。真实退化（15/15 → 9/15）
    属于**要允许发生并被记录**的东西，不该拦。
  - **rerank 层静默失效** —— 若 rerank 分数没算出来，报告可能退回成vector 层那套数字，
    两份报告会长得一模一样。这要靠"两份产物的数字不该相同"来判，
    **本脚本没做**（需要先确认哪些报告本就应该相同，`-norw` 与非 `-norw` 的关系就不简单）。
  - **语料换了但没重跑** —— 产物里没记语料指纹/sha，比对不出来。

⚠️ **刻意不做的检查**：
  - 不设"Recall@5 至少 60%"这种**性能下界** —— 那会把"真实退化"也拦下来，
    而真实退化是**要允许发生并被记录**的（这就是本项目的负结果传统）。
    这里只拦"**明显不是跑出来的**"。
  - 不校验具体数值是否与 README 一致 —— 那是 `check_readme_numbers.py` 的活，
    两边职责不重叠。

用法：
    python scripts/check_artifact_sanity.py         # 检查全部产物
    python scripts/check_artifact_sanity.py --show   # 顺便打印实测值

退出码：0 = 全过；1 = 有产物不合理（拦提交）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ★ 产物文件清单。**必须显式列出**（不用 glob 去逐个检查）——
#   但显式清单必须配一道反查，否则"新报告忘了登记"会静默漏检。
#   详见下面 ARTIFACT_GLOB 处的说明。
ARTIFACTS = [
    "evalset/report-vector.md",
    "evalset/report-vector-heldout.md",
    "evalset/report-rerank.md",
    "evalset/report-rerank-main.md",
    "evalset/report-rerank-main-norw.md",
    "evalset/report-rerank-heldout.md",
    "evalset/report-rerank-heldout-norw.md",
]

# ★★ 清单的反查基准：**磁盘上有哪些产物**，用来发现"新报告忘了登记"。
#   为什么必须反查（2026-10-11 实测踩到）：
#     上一版 ARTIFACTS 只列了 6 份，磁盘上其实有 7 份——
#     `evalset/report-rerank-main-norw.md` 从头到尾**没被任何检查碰过**。
#     而"漏检比误报危险得多"这条正是我自己在上面 docstring 里写的，
#     结果它当场就发生在我自己身上。⇒ 光有"显式清单"不够，
#     必须再有一道"清单 ⊇ 磁盘实际产物"的反查，否则清单本身就成漏斗口。
ARTIFACT_GLOB = "evalset/report*.md"


def _on_disk() -> list[str]:
    """磁盘上实际存在的产物（相对 REPO_ROOT，正斜杠）。"""
    return sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in REPO_ROOT.glob(ARTIFACT_GLOB)
        if p.is_file()
    )



def _extract(text: str) -> dict:
    """从一份报告里抠出关键字段。抽不到就留 None（由调用方判）。"""
    out: dict = {}

    m = re.search(r"应答题数：(\d+)\s*Recall@5：(\d+)/(\d+)", text)
    if m:
        out["ans_total"] = int(m.group(1))
        out["recall_ok"] = int(m.group(2))
        out["recall_den"] = int(m.group(3))

    m = re.search(r"拒答题数：(\d+)\s*拒答准确率：(\d+)/(\d+)", text)
    if m:
        out["refuse_total"] = int(m.group(1))
        out["refuse_ok"] = int(m.group(2))
        out["refuse_den"] = int(m.group(3))

    m = re.search(r"漏答率：(\d+)/(\d+)", text)
    if m:
        out["miss_ok"] = int(m.group(1))
        out["miss_den"] = int(m.group(2))

    m = re.search(r"应答题最低向量分\s*([0-9.]+)", text)
    if m:
        out["ans_min_score"] = float(m.group(1))
    m = re.search(r"拒答题最高向量分\s*([0-9.]+)", text)
    if m:
        out["refuse_max_score"] = float(m.group(1))

    return out


def check_one(rel: str) -> list[str]:
    """检查一份产物，返回问题清单（空 = 通过）。"""
    p = REPO_ROOT / rel
    if not p.exists():
        return [f"✗ {rel}：文件不存在 —— 清单里登记了但磁盘上没有。"
                f"要么它被删了，要么该从 ARTIFACTS 里去掉"]
    text = p.read_text(encoding="utf-8")
    if len(text) < 100:
        return [f"✗ {rel}：只有 {len(text)} 字节 —— 评测跑到一半崩了？"]
    d = _extract(text)
    errs: list[str] = []

    # ---- 字段抽取失败 =产物格式变了，要人看一眼 ----
    if "recall_den" not in d:
        errs.append(f"✗ {rel}：抽不到「应答题数/Recall@5」—— 产物格式可能变了，"
                    f"这个脚本没跟上。⚠️ 不要直接改脚本，先确认新格式是什么")
    if "refuse_den" not in d:
        errs.append(f"✗ {rel}：抽不到「拒答题数/拒答准确率」—— 同上")

    # ---- 判据 1：Recall@5 不能为 0 ----
    if "recall_ok" in d and d["recall_ok"] == 0:
        errs.append(
            f"✗ {rel}：Recall@5 是 0/{d['recall_den']} —— 全盘一条都没召回。"
            f"★这不是「检索效果差」，是 embedding 没加载 / 语料为空 / 模型路径错了。"
            f"**别把这份结果当基线写进任何文档**"
        )

    # ---- 判据 2：Recall 分子不能超过分母 ----
    if "recall_ok" in d and "recall_den" in d and d["recall_ok"] > d["recall_den"]:
        errs.append(f"✗ {rel}：Recall 分子 {d['recall_ok']} > 分母 {d['recall_den']}"
                    f" —— 字段读串位了")

    # ---- 判据 3：分母 = 题目集合大小 ----
    if "ans_total" in d and "recall_den" in d and d["ans_total"] != d["recall_den"]:
        errs.append(
            f"✗ {rel}：应答题数 {d['ans_total']} ≠ Recall 分母 {d['recall_den']}"
            f" —— 两个字段说的不是同一件事"
        )
    if "refuse_total" in d and "refuse_den" in d and d["refuse_total"] != d["refuse_den"]:
        errs.append(
            f"✗ {rel}：拒答题数 {d['refuse_total']} ≠ 拒答准确率分母 {d['refuse_den']}"
            f" —— 两个字段说的不是同一件事"
        )

    # ---- 判据 4：漏答数不应超过 Recall 未召回数（口径一致）----
    if "miss_ok" in d and "recall_ok" in d and "recall_den" in d:
        miss_expected = d["recall_den"] - d["recall_ok"]
        if d["miss_ok"] != miss_expected:
            warns = (
                f"· {rel}：漏答率 {d['miss_ok']}/{d['miss_den']}，"
                f"但按 Recall 反推应为 {miss_expected}。"
                f"⇒ 两者口径不同（漏答算的是 knowledge_hit，而 Recall 算 gold 命中）—— "
                f"**这是已知的口径差，不是 bug**，但写文档时别混用"
            )
            # 只提示不报错：口径差是真实存在的，不该拦
            _sanity_warns.append(warns)

    # ---- 判据 5：★ 两个不同题集的极值不该撞成同一个常数 ----
    #   这是 2026-10-11 那次事故最强的信号：0.4240 == 0.4240
    if "ans_min_score" in d and "refuse_max_score" in d:
        if abs(d["ans_min_score"] - d["refuse_max_score"]) < 1e-9:
            errs.append(
                f"✗ {rel}：应答题最低向量分与拒答题最高向量分**完全相等**"
                f"（都是 {d['ans_min_score']}）—— "
                f"这两个是不同题目集合上的极值，正常必不相等。"
                f"★ 相等说明分数被常量化/取错字段/根本没跑。**这正是 2026-10-11 那次事故的信号**"
            )

    # ---- 判据 6：漏答分母应等于应答题数 ----
    if "miss_den" in d and "ans_total" in d and d["miss_den"] != d["ans_total"]:
        errs.append(
            f"✗ {rel}：漏答率分母 {d['miss_den']} ≠ 应答题数 {d['ans_total']}"
        )

    return errs


# 模块级累积的 warns（口径差这类"不是错但要知道"的事）
_sanity_warns: list[str] = []


def check_all() -> tuple[list[str], list[str]]:
    _sanity_warns.clear()
    errs: list[str] = []
    for rel in ARTIFACTS:
        errs += check_one(rel)

    # ★ 探头自检：如果一份都没检查到，等于没检查
    checked = [r for r in ARTIFACTS if (REPO_ROOT / r).exists()]
    if not checked:
        errs.append(
            "⚠⚠【探头失灵】ARTIFACTS 里的文件一个都不存在 —— "
            "本轮没有检查到任何产物。①是不是路径写错了？"
            "②是不是产物被移到别处、清单没跟着改？"
        )

    # ---- ★ 判据 0：清单反查 —— 磁盘上的产物有没有漏登记的 ----
    #   这不是"多检查一个文件"，是**堵住漏斗口**：
    #   漏登记 ⇒ 那份产物从此不被任何检查碰过 ⇒ 坏了也没人喊。
    registered = set(ARTIFACTS)
    missing = [r for r in _on_disk() if r not in registered]
    if missing:
        errs.append(
            f"✗ 有{len(missing)} 份产物在磁盘上，但**没登记进 ARTIFACTS** —— "
            f"它们不会被本脚本检查："
            + "".join(f"\n      · {m}" for m in missing)
            + "\n★ 要么加进 ARTIFACTS（如果它该被检查），"
              "要么说明它为什么可以豁免（在代码旁写理由，别默默留着）。"
        )
    # 反向：清单里登记了、磁盘上却没有 —— check_one 已经会报，这里不重复

    return errs, list(_sanity_warns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="顺便打印各产物的实测值")
    args = ap.parse_args()

    errs, warns = check_all()

    if args.show:
        print("各产物实测值：")
        for rel in ARTIFACTS:
            p = REPO_ROOT / rel
            if not p.exists():
                print(f"  {rel:44s} ⇒ ★文件不存在")
                continue
            d = _extract(p.read_text(encoding="utf-8"))
            print(
                f"  {rel:44s} ⇒ Recall {d.get('recall_ok','?')}/{d.get('recall_den','?')}"
                f" ·拒答 {d.get('refuse_ok','?')}/{d.get('refuse_den','?')}"
                f" · 最低分 {d.get('ans_min_score','?')}"
                f" / 最高分 {d.get('refuse_max_score','?')}"
            )
        print()

    for w in warns:
        print(w)
    if warns:
        print()

    if errs:
        print("产物合理性检查失败：")
        for e in errs:
            print(f"  {e}")
        print()
        print("★ 一份骗人的评测报告比没有评测更坏 —— 它会把坏环境的产物")
        print("  当成基线，然后README 跟着一起变成错的。")
        print("  先查环境（embedding 加载了吗 / 语料空不空 / API 通不通），再重跑。")
        return 1

    print(f"✓ 产物合理性检查通过（{len(ARTIFACTS)} 份）")
    return 0


if __name__ == "__main__":
    sys.exit(main())