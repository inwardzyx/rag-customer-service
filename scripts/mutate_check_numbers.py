"""对账脚本的变异测试：注入已知错误，验证 check_readme_numbers 真的会报红。

为什么需要这个（2026-10-10）
--------------------------------
`check_readme_numbers.py` 第一版跑出来是 "✓ 通过"，
但变异测试把 README 的 0.5968 改成 0.9999 后，它**照样报通过** ——
因为第一版只查产物侧、没查 README 侧。

★ 这就是今天 loader 清洗账塌成 0 的同一个模式：
  **探头只照了一面，却把"没发现问题"读成"没问题"。**
  而"跑通了"这种自述型证据，恰恰是最不可信的那种。

⇒ 所以这个脚本自己必须有变异测试。规则：
  **任何自检类脚本，都要能证明它在坏情况下会响。**

用法：
    python scripts/mutate_check_numbers.py

判据（与 mutate_check.py 保持一致的写法）：
    - 全部变异都必须被检出（漏一个 = 自检有洞）
    -不注入变异时必须报绿（否则是假阳性满天飞）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CHECK = REPO_ROOT / "scripts" / "check_readme_numbers.py"
PY = sys.executable

# ★ 每个变异：(说明, 要替换的原文, 替换成什么)
#   ⚠️ 原文必须真的存在于 README 里 —— 否则"变异没生效"会被误读成"检出成功"
#   ★ 用 str.replace(old, new) 【不带次数】= 把全部出现处都改掉。
#     第一版是 replace(..., 1)（只改第一处），这本身就是假通过的来源：
#     同一句话在 README 里出现两次时，改第二处既不会被 checker 发现、
#     也不会被变异测试发现（cc 指出的洞2）。
MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "把阈值余量 0.5968 改错（README 侧）",
        "README.md",
        "应答最低 0.5968 vs拒答最高 0.6321",
        "应答最低 0.9999 vs拒答最高 0.6321",
    ),
    (
        "把拒答准确率 3/5 改成 4/5（README 侧）",
        "README.md",
        "拒答准确率掉到 3/5",
        "拒答准确率掉到 4/5",
    ),
    (
        "把拒答题最高向量分 0.6321 改错（README 侧）",
        "README.md",
        "应答最低 0.5968 vs拒答最高 0.6321",
        "应答最低 0.5968 vs拒答最高 0.7123",
    ),
    (
        "把 gold 中位排名 1→2→4 改错（README 侧）",
        "README.md",
        "gold中位排名 **1 → 2 → 4**",
        "gold中位排名 **1 → 2 → 9**",
    ),
    (
        "把源语料丢字数 5241 改错（README 侧）",
        "README.md",
        "合计要丢 5241 字",
        "合计要丢 9999 字",
    ),
    # ★★ 这几条是【2026-10-10 盲审抓到后补的】—— 第一版清单里没有，
    #   所以「5/5 全检出」从来没照到这些洞。它们验证的是：
    #   修复之后，再注入这些错误，脚本报不报错。
    #   ⚠️ 要把【全部】出现处都改掉：_check_recall 的判据是
    #   「README 里能找到至少一个等于 15/15 的 x/15」，
    #   只改一处的话别处还留着 15/15，它照样通过
    #   —— 这是我第一版变异设计的错（第二个自己造的假通过）。
    (
        "把应答题 Recall@5 从 15/15 改成 12/15（cc+DSH 独立抓到的洞3，全部出现处）",
        "README.md",
        "15/15",
        "12/15",
    ),
    (
        "把实验记录里所有三段排名都改掉（验证 DSH 指出的静默降级）",
        "experiments/_语料规模与多查价值实验.md",
        "中位排名",
        "排名中位数",
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


def main() -> int:
    if not README.exists():
        print("✗ 找不到 README.md")
        return 2
    original = README.read_text(encoding="utf-8")

    # ---- 0. 干净时必须绿 ----
    code, out = run_check()
    if code != 0:
        print("✗ 未注入变异时对账就报错了 —— 先修好再测：")
        print(out)
        return 2
    print("✓ 基线：未注入变异时报绿")

    # ---- 逐个变异，每轮恢复原文 ----
    undetected: list[str] = []
    not_applied: list[str] = []
    applied: list[str] = []

    for desc, target_rel, old, new in MUTATIONS:
        target = REPO_ROOT / target_rel
        base = original if target_rel == "README.md" else target.read_text(encoding="utf-8")
        if old not in base:
            not_applied.append(desc)
            print(f"? 跳过（原文不存在，变异没生效）：{desc}")
            print(f"    找的是：{old!r}")
            continue

        mutated = base.replace(old, new)  # ★ 全量替换，不带次数
        assert mutated != base, "替换没生效"
        # ★ 换行必须原样写回。
        #   实测踩过（DSH 指出的读码推断，我确认了机制）：
        #   read_text 会把 CRLF 统一成 \n，write_text 又按 os.linesep 翻回去。
        #   ⇒ 若README 原本是 LF，跑一次这个脚本整份文件就变 CRLF，
        #     产生全文件级 diff，而且 git 会把"每一行都改了"。
        #
        # ⚠️ DSH 指出的原文说法："两个脚本纯读文件 + 正则"对 mutate 不成立 ——
        #   它一轮要写 6 次（5 注入 + 1 复原），跑的时候是在【改工作区】。
        #   这条已写进 CLAUDE.md 的协作黑板一节，别再宣称它是只读的。
        target.write_text(mutated, encoding="utf-8", newline="")
        applied.append(desc)
        try:
            code, out = run_check()
            if code == 0:
                undetected.append(desc)
                print(f"✗ 漏检：{desc}")
                print("    ⇒ 对账脚本对这一处没反应，说明它没在照 README")
            else:
                first = next(
                    (ln.strip() for ln in out.splitlines() if ln.strip().startswith(("✗", "⚠"))),
                    "(没打印错误行)",
                )
                print(f"✓ 检出：{desc}")
                print(f"    {first}")
        finally:
            target.write_text(base, encoding="utf-8", newline="")

    # ---- 复原后必须还绿 ----
    code, out = run_check()
    if code != 0:
        print("✗ 全部变异恢复后仍然报错 —— README 可能没还原干净")
        print(out)
        return 2

    print()
    # ★★ 洞1（cc 与 DSH 独立同时抓到）：注入失败必须算失败。
    #   原来 not_applied 只打印不 return，最后照样打
    #   "✓ 全部 {len(MUTATIONS)} 个变异都被检出" —— 那个 len 是【清单长度】，
    #   不是实际跑了几条。于是「2 个没注入 + 3 个检出」和「5 个都检出」
    #   输出同一句话、同一退出码。⇒ 现在非空即失败。
    if not_applied:
        print(f"✗ {len(not_applied)}/{len(MUTATIONS)} 个变异没注入成功（原文已变），"
              "这些位置没测到 —— 【不能算通过】：")
        for d in not_applied:
            print(f"    - {d}")
        print()
        print("  ⚠常见原因：README 措辞/空格变了（这里 old 是【逐字节精确】匹配，")
        print("    而 check 侧正则都是 \\s* 宽容型 —— 一处排版调整就能让变异全静默跳过）。")
        print("  ⇒ 修法：把 MUTATIONS 里的 old 改成跟着 README 一起更新，或改用正则。")
        return 1

    if undetected:
        print(f"✗ {len(undetected)}/{len(applied)} 个变异漏检 —— 自检有洞，别信它：")
        for d in undetected:
            print(f"    - {d}")
        return 1

    #★ 打实际跑出来的数，不打 len(MUTATIONS)
    print(f"✓ 实际注入 {len(applied)} 个变异，{len(applied) - len(undetected)} 个被检出，"
          "且恢复后仍报绿")
    print("  ⇒ 这一版对账脚本【真的会响】，可以挂pre-commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())