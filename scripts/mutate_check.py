# -*- coding: utf-8 -*-
"""变异测试：故意把代码改坏，看测试会不会红 —— 证明「测试不是摆设」。

★ 为什么需要它（本项目的一条纪律）

    测试全绿有两种可能：① 代码是对的；② 测试根本测不到那个点。
    光看「绿」区分不了这两者。把代码改坏一次再跑：

        改坏 → 测试变红 = 抓住（符合预期）
        改坏 → 测试仍绿 = **漏网**（说明有断言在装样子，必须补）

    这类「写了但没人发现它没用」的断言，本项目已经抓到过 3 次。
    最近一次是页脚剥离的日志被降级成 `info`（= 静默清洗），**48 条测试全绿、毫无反应**，
    补了 `test_strip_is_not_silent` 才抓住 —— 那条变异就在下面的清单里（第 5 条）。

★ 为什么改完不用 git 恢复，而是自己按字节恢复

    `read_text` / `write_text` 在 Windows 上会把 LF 悄悄转成 CRLF，
    恢复完文件内容看着没变、换行全变了，下次提交就是一大片假 diff。
    所以这里一律用 `read_bytes` / `write_bytes`，并在最后校验哈希。

★ 跑法（仓库根目录）

    python scripts/mutate_check.py                 # 跑全部变异
    python scripts/mutate_check.py --list          # 只列清单，不动代码
    python scripts/mutate_check.py --only 页脚      # 只跑名字里含「页脚」的

★ 退出码

    0 = 全部被抓住；1 = 有漏网 / 锚点失效 / 恢复失败。
    所以它可以直接当门禁用（不过 CI 没接它，理由见 README）。
"""

import hashlib
import subprocess
import sys
from pathlib import Path

# Windows 控制台默认是 GBK，直接 print 中文会炸。
# reconfigure 把标准输出换成 UTF-8（老 Python 没有这个方法，所以先 hasattr 问一下）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent


# ==================================================================
# 变异清单
#   name   : 这次改坏的是什么（也是 --only 匹配的对象）
#   file   : 改哪个文件（相对仓库根目录）
#   anchor : 原文里要被替换的那段（必须**逐字**匹配，包括缩进）
#   mutant : 替换成什么 —— 也就是「一个真实可能犯的错」
#   test   : 改坏之后必须变红的那条测试
#   why    : 这个变异模拟的真实风险（一句话，写给人看）
# ==================================================================
MUTATIONS = [
    {
        "name": "版本比较退回字符串比较",
        "file": "service.py",
        "anchor": "    return tuple(int(p) for p in parts)",
        "mutant": "    return v.strip()  # 变异：退回字符串比较",
        "test": "tests/test_guard.py::test_version_compare_survives_unpadded_date",
        "why": "模拟「图省事直接比字符串」—— 那样 '2026-9-5' 会被当成比 '2026-09-25' 更新",
    },
    {
        "name": "页脚剥离句中命中（去掉行首锚定）",
        "file": "kb/loader.py",
        "anchor": "    m = _CHROME_RE.search(text)",
        "mutant": '    m = re.search("|".join(CHROME_STRONG + CHROME_WEAK), text)  # 变异',
        "test": "tests/test_loader.py::test_strip_footer_ignores_marker_mid_sentence",
        "why": "模拟「全串查找」—— 会把正文里那句'如有疑问请联系我们'后面的真条文削掉",
    },
    {
        "name": "分块不报错（去掉无标题保护）",
        "file": "kb/loader.py",
        "anchor": "        if len(sections) == 1:",
        "mutant": "        if False:  # 变异：去掉无标题保护",
        "test": "tests/test_loader.py::test_no_heading_is_an_error_not_silence",
        "why": "模拟「标题漏空格就整篇消失」—— 文档产出 0 块却没有任何报错",
    },
    {
        "name": "导语丢弃不吭声（warning 降级）",
        "file": "kb/loader.py",
        "anchor": '            logger.warning("「%s」第一个 ## 之前有 %d 字正文',
        "mutant": '            logger.debug("「%s」第一个 ## 之前有 %d 字正文',
        "test": "tests/test_loader.py::test_preamble_before_first_heading_is_reported",
        "why": "模拟「悄悄丢内容」—— 和 content[:400] 截断是同一个病",
    },
    {
        "name": "页脚剥离不吭声（warning 降级）",
        "file": "kb/loader.py",
        "anchor": '                logger.warning("「%s｜%s」剥离网页页脚 %d 字',
        "mutant": '                logger.debug("「%s｜%s」剥离网页页脚 %d 字',
        "test": "tests/test_loader.py::test_strip_is_not_silent",
        "why": "★ 上一次真实漏网过的那个 —— 改完 48 条测试全绿，等于又造了一个静默清洗",
    },
    {
        "name": "截断不吭声（warning 降级）",
        "file": "kb/loader.py",
        "anchor": '                logger.warning("「%s｜%s」长 %d 字，超过 max_chars=%d，已截断（丢 %d 字）"',
        "mutant": '                logger.debug("「%s｜%s」长 %d 字，超过 max_chars=%d，已截断（丢 %d 字）"',
        "test": "tests/test_loader.py::test_truncation_is_not_silent",
        "why": "模拟最初的病：丢字不报，谁也不知道库里少了几百字",
    },
    {
        "name": "切多块次判据消失（粒度变了却只说记账）",
        "file": "scripts/measure_chunk_health.py",
        "anchor": "    if len(lost) >= SPLIT_TRIGGER_COUNT:",
        "mutant": "    if False:  # 变异：删掉次判据",
        "test": "tests/test_loader.py::test_split_trigger_grading",
        "why": "模拟「次判据被顺手删掉」—— 3 块以上被截断本说明粒度从条变章，删了就只剩「记账」，动作线永远不触发",
    },
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(m: dict) -> tuple[bool, str]:
    """跑一个变异，返回 (是否被抓住, detail)。

    ★ 锚点找不到时【报错】，不是跳过。理由和变异测试本身一样：
      测试要是可以被静默跳过，它就退化成摆设了 —— 锚点失效必须吵出来。
    """
    path = ROOT / m["file"]
    original = path.read_bytes()
    text = original.decode("utf-8")

    if m["anchor"] not in text:
        return False, "锚点没找到 —— 代码改过了，请更新 scripts/mutate_check.py 里这条"

    try:
        path.write_bytes(text.replace(m["anchor"], m["mutant"], 1).encode("utf-8"))
        r = subprocess.run(
            [sys.executable, "-m", "pytest", m["test"], "-q"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    finally:
        path.write_bytes(original)          # 无论如何都要还原

    if path.read_bytes() != original:
        return False, "★ 恢复失败！文件和工作区不一致，请用 git status 检查"

    last = ""
    for line in r.stdout.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            last = line.strip()

    if r.returncode != 0:
        return True, last                   # 变红 = 抓住
    return False, last                      # 还是绿的 = 漏网


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print("变异清单（%d 条）：\n" % len(MUTATIONS))
        for i, m in enumerate(MUTATIONS, 1):
            print("%d. %s" % (i, m["name"]))
            print("   改 %s → 期望 %s 变红" % (m["file"], m["test"].split("::")[-1]))
            print("   为什么：%s\n" % m["why"])
        return 0

    picked = MUTATIONS
    if "--only" in argv:
        kw = argv[argv.index("--only") + 1]
        picked = [m for m in MUTATIONS if kw in m["name"]]
        if not picked:
            print("没有名字含「%s」的变异。用 --list 看全部。" % kw)
            return 1

    print("=" * 66)
    print("变异测试：故意改坏代码，看测试会不会红")
    print("判据：变红 = 抓住（好）；仍绿 = 漏网（说明那条测试是摆设）")
    print("=" * 66)

    before = {m["file"]: sha(ROOT / m["file"]) for m in picked}
    caught = 0

    for i, m in enumerate(picked, 1):
        print("\n[%d/%d] %s" % (i, len(picked), m["name"]))
        print("       改坏：%s" % m["why"])
        ok, detail = run_one(m)
        caught += ok
        print("       → %s   %s" % ("变红 ✔ 抓住" if ok else "★★ 仍绿，漏网！", detail))

    print()
    print("-" * 66)
    print("结果：抓住 %d / %d" % (caught, len(picked)))

    # 再校验一遍：所有涉及的文件必须和开始时字节一致
    bad = [f for f, h in before.items() if sha(ROOT / f) != h]
    if bad:
        print("★ 恢复校验失败：%s" % bad)
        return 1
    print("恢复校验：%d 个文件哈希一致 ✔" % len(before))

    return 0 if caught == len(picked) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
