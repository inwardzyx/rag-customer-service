"""产物合理性检查的变异测试：注入"像坏环境跑出来的"产物，验证脚本会响。

为什么需要这个
----------------
`check_artifact_sanity.py` 的存在理由就是"2026-10-11 那次坏环境产物被提交"。
★ 所以它的第一条变异必须用**那次事故的真实数据**（Recall 0/15、两个极值都是 0.4240），
  而不是我自己编的数 —— 编的数可能恰好不符合当时真正的失效机制。

★ 另一个更一般的理由：这个脚本里的判据全是"我推导出来的下界"，
  而我推导的时候**没想过它什么时候不该报**。负向变异（应当保持绿的那些）
  是唯一能测出"假阳性"的方向，而假阳性会把检查变成噪音。

用法：
    python scripts/mutate_check_artifact.py

判据：
  - 每条变异都必须被检出（漏一个 = 这个判据抓不到它声称能抓的东西）
  - 不注入时必须全绿（否则满屏假阳性，迟早被当噪音关掉）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK = REPO_ROOT / "scripts" / "check_artifact_sanity.py"
PY = sys.executable

TARGET = "evalset/report-vector-heldout.md"

# ★ 事故原始数据：Recall 0/15 + 应答题最低分 == 拒答题最高分 == 0.4240
#   两处都在同一份产物里 —— 一次坏环境跑出来的
ACCIDENT = """# 评测报告（离线层(向量分0.55)）

- 应答题数：44　Recall@5：0/44
- 漏答率：0/44（knowledge_hit==false 才算漏答）
- 拒答题数：61　拒答准确率：9/61
- 阈值余量：应答题最低向量分 0.4240（⚠ 已低于阈值）／ 拒答题最高向量分 0.4240（安全（<0.55））

> 注：这一份是注入的假数据，用于验证检查脚本。
"""

# (说明, 要替换的原文, 替换成什么, 期望被哪条判据抓到)
MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "★ 事故复现：Recall@5 全 0",
        "- 应答题数：44　Recall@5：35/44",
        "- 应答题数：44　Recall@5：0/44",
        "判据1（Recall 不能为 0）",
    ),
    (
        "分母与应答题数不一致",
        "- 应答题数：44　Recall@5：35/44",
        "- 应答题数：45　Recall@5：35/44",
        "判据3（分母 = 题目集合大小）",
    ),
    (
        "分子超过分母（字段读串位）",
        "- 应答题数：44　Recall@5：35/44",
        "- 应答题数：44　Recall@5：99/44",
        "判据2（分子 ≤ 分母）",
    ),
    (
        "拒答准确率分母 ≠ 拒答题数",
        "- 拒答题数：61　拒答准确率：9/61",
        "- 拒答题数：60　拒答准确率：9/61",
        "判据3",
    ),
    (
        "漏答分母 ≠ 应答题数",
        "- 漏答率：2/44",
        "- 漏答率：2/45",
        "判据6（漏答分母 = 应答题数）",
    ),
]

# ★★★ 成对变异：两条判据要**同时**注入才构成事故现场。
#   判据 5 是"两个不同题集的极值相等" —— 单独改最低分不会让它触发
#   （另一个极值没变，仍然不等）。我第一版把它当单条变异写，
#   脚本报"漏检"，**其实是我的变异设计错了**，不是检查有洞。
#   ⇒ 这类判据必须成对注入。这本身是个元教训：
#     **"漏检"有两种可能 —— 检查有洞，或变异设计错了。判之前先分清。**
PAIR_MUTATIONS: list[tuple[str, tuple[tuple[str, str], ...], str]] = [
    (
        "★ 事故复现（成对）：应答题最低分与拒答题最高分撞成同一个常数",
        (
            ("应答题最低向量分 0.5224", "应答题最低向量分 0.4240"),
            ("拒答题最高向量分 0.782", "拒答题最高向量分 0.4240"),
        ),
        "判据5（两个极值不该相等）—— 这是 2026-10-11 事故最强的信号",
    ),
]


# ★★★ 负向变异：这些**必须保持绿**。它们是测假阳性的唯一方向。
# 实测踩过的假阳性：「漏答率与 Recall 反推值不同」—— 我一度想把它报成错，
#     但那是【真实的口径差】（漏答算 knowledge_hit，Recall 算 gold 命中），
#     报出来就是假阳性。现在它只提示。
NEGATIVE_CASES: list[tuple[str, str, str]] = [
    (
        "漏答率与 Recall 反推值不同（已知口径差，不是 bug）",
        "- 漏答率：2/44",
        "- 漏答率：9/44",
    ),
]


# ★★★ 判据 0（清单反查）的变异：改的是**检查器自己的清单**，不是产物。
#   为什么单独一类：前七条全都在动产物，只有这一条动的是"检查器管不管这个文件"。
#   如果不测它，就等于默认"清单没漏"—— 而上一版清单**真的漏了**
#   （report-rerank-main-norw.md 从未被检查），且那个洞是在我写完检查器之后才被发现的。
#
#   ⚠️ 这条变异**不注入任何坏产物**：如果只有"清单漏登记 + 产物本身健康"，
#     期望的行为是**必须报错** —— 因为"没人检查它"本身就是缺陷，
#     不管它现在看起来干不干净。
LIST_MUTATIONS: list[tuple[str, str, str]] = [
    (
        "★ 清单漏登记：把 report-rerank-main-norw.md 从 ARTIFACTS 里去掉"
        "（产物本身健康，但从此不再被任何检查碰过）",
        '"evalset/report-rerank-main-norw.md",\n    ',
        "",
        "判据0（清单反查）",
    ),
]



def run_check() -> tuple[int, str]:
    r = subprocess.run(
        [PY, str(CHECK)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _restore(path: Path, original: str) -> None:
    path.write_text(original, encoding="utf-8", newline="")


def main() -> int:
    target = REPO_ROOT / TARGET
    if not target.exists():
        print(f"✗ 找不到 {TARGET}")
        return 2
    original = target.read_text(encoding="utf-8")

    # ---- 0. 基线必须绿 ----
    code, out = run_check()
    if code != 0:
        print("✗ 未注入变异时检查就报错了 —— 先修好再测：")
        print(out)
        return 2
    print("✓ 基线：未注入变异时报绿")

    undetected: list[str] = []
    not_applied: list[str] = []
    applied = 0

    # ---- 正向变异：每一条都必须被检出 ----
    for desc, old, new, expect in MUTATIONS:
        if old not in original:
            not_applied.append(f"{desc}（原文不存在：{old!r}）")
            print(f"? 跳过：{desc}")
            continue
        target.write_text(original.replace(old, new), encoding="utf-8", newline="")
        try:
            code, out = run_check()
            if code == 0:
                undetected.append(desc)
                print(f"✗ 漏检：{desc}（期望由{expect} 抓到）")
            else:
                first = next(
                    (ln.strip() for ln in out.splitlines() if ln.strip().startswith("✗")),
                    "(没打印错误行)",
                )
                print(f"✓ 检出：{desc}")
                print(f"    {first}")
                applied += 1
        finally:
            _restore(target, original)

    # ---- 成对变异：两处一起改，才构成事故现场 ----
    for desc, edits, expect in PAIR_MUTATIONS:
        missing = [old for old, _ in edits if old not in original]
        if missing:
            not_applied.append(f"{desc}（原文不存在：{missing}）")
            print(f"? 跳过：{desc}")
            continue
        mutated = original
        for old, new in edits:
            mutated = mutated.replace(old, new)
        target.write_text(mutated, encoding="utf-8", newline="")
        try:
            code, out = run_check()
            if code == 0:
                undetected.append(desc)
                print(f"✗ 漏检：{desc}（期望由{expect} 抓到）")
            else:
                first = next(
                    (ln.strip() for ln in out.splitlines() if ln.strip().startswith("✗")),
                    "(没打印错误行)",
                )
                print(f"✓ 检出：{desc}")
                print(f"    {first}")
                applied += 1
        finally:
            _restore(target, original)

    # ---- 负向变异：必须保持绿 ----
    false_positives: list[str] = []
    for desc, old, new in NEGATIVE_CASES:
        if old not in original:
            not_applied.append(f"{desc}（原文不存在）")
            continue
        target.write_text(original.replace(old, new), encoding="utf-8", newline="")
        try:
            code, out = run_check()
            if code != 0:
                false_positives.append(desc)
                print(f"✗ 假阳性：{desc} —— {desc}**应当保持绿**，却报错了")
                first = next(
                    (ln.strip() for ln in out.splitlines() if ln.strip().startswith("✗")),
                    "",
                )
                print(f"    {first}")
            else:
                print(f"✓ 保持绿（负向用例）：{desc}")
                applied += 1
        finally:
            _restore(target, original)

    # ---- 清单变异：改的是检查器自己的 ARTIFACTS ----
    check_src = CHECK.read_text(encoding="utf-8")
    for desc, old, new, expect in LIST_MUTATIONS:
        if old not in check_src:
            not_applied.append(f"{desc}（检查器源码里找不到：{old!r}）")
            print(f"? 跳过：{desc}")
            continue
        CHECK.write_text(check_src.replace(old, new, 1), encoding="utf-8", newline="")
        try:
            code, out = run_check()
            if code == 0:
                undetected.append(desc)
                print(f"✗ 漏检：{desc}（期望由{expect} 抓到）")
            else:
                first = next(
                    (ln.strip() for ln in out.splitlines() if ln.strip().startswith("✗")),
                    "(没打印错误行)",
                )
                print(f"✓ 检出：{desc}")
                print(f"    {first}")
                applied += 1
        finally:
            CHECK.write_text(check_src, encoding="utf-8", newline="")

    # ---- 还原后必须还绿 ----
    code, out = run_check()
    if code != 0:
        print("✗ 全部恢复后仍报错 —— 产物文件可能没还原干净")
        print(out)
        return 2

    print()
    if not_applied:
        print(f"⚠ {len(not_applied)} 条变异没注入成功，这些方向没测到：")
        for d in not_applied:
            print(f"    - {d}")
    if undetected:
        print(f"✗ {len(undetected)} 条正向变异漏检 —— 这些判据抓不到它们声称能抓的东西：")
        for d in undetected:
            print(f"    - {d}")
    if false_positives:
        print(f"✗ {len(false_positives)} 条负向变异被误报 —— 假阳性会把检查变成噪音：")
        for d in false_positives:
            print(f"    - {d}")
    if undetected or false_positives:
        return 1
    if not_applied:
        return 2
    print(f"✓ {applied} 条变异全部符合预期（正向全检出 + 负向保持绿），恢复后仍报绿")
    print("  ⇒ 这一版产物检查【真的会响，且不会乱响】")
    return 0


if __name__ == "__main__":
    sys.exit(main())
