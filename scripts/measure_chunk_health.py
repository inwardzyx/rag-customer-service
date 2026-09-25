# -*- coding: utf-8 -*-
"""scripts/measure_chunk_health.py —— 数据层体检（只读，秒级）

回答一个问题：**现在这份知识库里，块健不健康？**

为什么单独要有这个脚本：

    检索层有一堆指标（Recall@5、拒答率、阈值间隔……），数据层却一直没有。
    「数据层怎么样」不该靠印象回答 —— 它得能像 `evalset/run_eval.py` 一样被量出来，
    否则每次改完切片规则，你只能说"感觉干净多了"。

★ 刻意不 import service：那会连带加载 embedding 模型（十几秒）。
  体检脚本必须是**秒级**的，否则你不会想跑它，它就废了。
  （代价是 max_chars / MIN_LEN 这两个阈值只能按约定写在这里，下面标了出处。
    防漂移的办法不是改这个脚本，而是去测试里断言"约定 == 真值"——
    两条各守一个真值，漂移时能直接看出是哪个数抄错了：
      max_chars → tests/test_loader.py::test_health_script_max_chars_is_not_drifted
      MIN_LEN   → tests/test_guard.py::test_health_script_min_len_is_not_drifted）

跑法（仓库根目录）：

    python scripts/measure_chunk_health.py
    python scripts/measure_chunk_health.py --samples 5    # 极端样本多看几条
"""

import logging
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.loader import (                                   # noqa: E402
    CLEAN_OP_FOOTER, CLEAN_OP_PREAMBLE, CLEAN_OP_TRUNCATE,
    _CHROME_RE, load_documents,
)

DOCS_DIR = ROOT / "docs"

# 这两个阈值是「约定」不是「引用」—— 原因见文件头（不想为了拿两个数去加载模型）。
MAX_CHARS_HINT = 400      # 对应 kb/loader.load_documents(max_chars=400)
MIN_LEN_HINT = 15         # 对应 service.MIN_LEN，低于它会被当"太短碎屑"拦掉

# version 必须是 YYYY-MM-DD —— 否则版本冲突关比不出新旧（见 service._version_key）
DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 正文里不可能合法出现"行首 ##" —— 真出现了，就说明有一个标题没被切开：
# 漏写 ## 后的空格时 `^##\s+` 匹配不上，那一段会**并进上一块**（块数/长度/身份键全都正常，
# 但上一块里混进了下一条的标题和条文 → 检索到了会张冠李戴）。
STRAY_HEAD = re.compile(r"(?m)^##")


class _CleanLogCollector(logging.Handler):
    """收集 kb/loader 加载过程中"动过的刀"。

    ★ 为什么需要这一节：上面那些检查项**全是终态快照** —— 只问"现在的块长什么样"，
      看不见"加载时动了几刀"。而剥离页脚 / 截断 / 丢导语这三刀都会**永久改内容**。

      最坏的情况正是第二轮修掉的那个 bug 的形状：一条真条文被从行中切断、削掉 53 字，
      剩下的文本长度正常、页脚标记也随正文一起没了 ——
      **快照 6 项全绿，一个字都不会说。** 这一节补的就是这个盲区。

    ★ 为什么读 record.clean_op，而不是解析日志文案：
      解析文案 = 又一处"判据和被测对象不同源"，改个措辞就静默失效。
      loader 把动作写进 extra、体检 import 同一批常量 —— 两边同源。
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.ops = {}                      # op -> (次数, 总字数)

    def emit(self, record):
        op = getattr(record, "clean_op", None)
        if op is None:
            return                         # 不是清洗动作，不记账
        n, chars = self.ops.get(op, (0, 0))
        self.ops[op] = (n + 1, chars + getattr(record, "clean_chars", 0))


def pct(values, q):
    """分位数（q 取 0-1）。不引 numpy，手算就行 —— 体检脚本要零依赖。"""
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def main(argv):
    samples = 5
    if "--samples" in argv:
        samples = int(argv[argv.index("--samples") + 1])

    print("=" * 70)
    print("数据层体检（只读）—— docs/ 里的块健不健康")
    print("=" * 70)

    collector = _CleanLogCollector()
    loader_log = logging.getLogger("kb.loader")
    loader_log.addHandler(collector)
    try:
        docs, errors = load_documents(DOCS_DIR)
    finally:
        loader_log.removeHandler(collector)

    # ★ 空库要早退：min()/max() 对空序列是**抛异常**（不是返回 None —— 和 sum([]) 返回 0
    #   不一样，这俩常被记混）。一个专门回答"数据层健康吗"的脚本，恰好在答案是
    #   "库里什么都没有"的时候崩掉，那就太讽刺了 —— 而那是**最该被报出来的状态**。
    if not docs:
        print("\n【规模】块数 0 —— 库是空的（docs/ 下一个可解析的 .md 都没有）")
        if errors:
            print("        解析失败的文件 %d 个：%s" % (len(errors), errors))
        print("\n结论：★ 数据层是空的。这比任何单项指标都严重，先让它有内容。")
        return 1

    lens = [len(d["text"]) for d in docs]
    checks = []          # (检查项, 是否通过, 数量) —— 结论里数的是这一份，和 ✔ 行一一对应

    # ---------- 规模 ----------
    by_doc = Counter(d["doc"] for d in docs)
    print("\n【规模】")
    print("  文档数            %d" % len(by_doc))
    print("  块数              %d" % len(docs))

    # ---------- 长度分布 ----------
    print("\n【长度分布】单位：字")
    print("  最短 %-5d 中位 %-5d p90 %-5d 最长 %d"
          % (min(lens), int(statistics.median(lens)), pct(lens, 0.9), max(lens)))
    over = [d for d in docs if len(d["text"]) >= MAX_CHARS_HINT]
    short = [d for d in docs if len(d["text"]) < MIN_LEN_HINT]
    checks.append(("超长块（≥%d 会被截断丢字）" % MAX_CHARS_HINT, not over, len(over)))
    print("  ≥%d（会被截断）    %d  %s"
          % (MAX_CHARS_HINT, len(over), "✔ 没有" if not over else "★ 该考虑切多块了"))
    print("  <%d（当碎屑拦）    %d  （不算问题：把关本来就会拦掉它们）"
          % (MIN_LEN_HINT, len(short)))

    # ---------- 脏数据 ----------
    # ★ 判据必须和 loader 同源：这里以前写的是"全串查找"（any(m in text ...)），
    #   而 loader 第二轮已经收紧成**只认行首**。两者方向相反 ——
    #   正文里合法的一句"学生如有疑问请联系我们"在 loader 眼里是正确的，
    #   在旧判据眼里却是"★ 有残留"，还可能诱导出一个**错误的修复**（去把 loader 改松）。
    #   现在直接复用 `_CHROME_RE`，判据只有一处。
    dirty = [d for d in docs if _CHROME_RE.search(d["text"])]
    blank = [d for d in docs if not d["text"].strip()]
    cr = [d for d in docs if "\r" in d["text"]]
    checks.append(("页脚残留块", not dirty, len(dirty)))
    checks.append(("空块", not blank, len(blank)))
    print("\n【脏数据】")
    print("  页脚残留块        %d  %s" % (len(dirty), "✔ 干净" if not dirty else "★ 有残留"))
    print("  空块              %d  %s" % (len(blank), "✔ 没有" if not blank else "★ 有"))
    # 提示项，不计入结论：read_text 的 universal newlines 已经把它保证成 0
    print("  含 \\r 的块        %d  （提示项，不计入结论：换行已被 read_text 归一）" % len(cr))

    # ---------- 结构 ----------
    stray = [d for d in docs if STRAY_HEAD.search(d["text"])]
    checks.append(("块内出现行首 ##（标题没切开）", not stray, len(stray)))
    print("\n【结构】")
    print("  块内行首 ## 残留  %d  %s"
          % (len(stray), "✔ 没有" if not stray else "★ 有条标题漏了 ## 后的空格，并进了上一块"))

    # ---------- 元数据 ----------
    bad_date = [d for d in docs if not DATE_OK.match(str(d.get("version", "")))]
    key = Counter((d["doc"], d["clause"]) for d in docs)
    dup_clause = {k: v for k, v in key.items() if v > 1}
    checks.append(("version 非 YYYY-MM-DD", not bad_date, len(bad_date)))
    checks.append(("同文档内 clause 重名", not dup_clause, len(dup_clause)))
    print("\n【元数据 / 身份键】")
    print("  version 非 YYYY-MM-DD   %d  %s"
          % (len(bad_date), "✔ 都能比较新旧" if not bad_date else "★ 会让版本比较失效"))
    print("  同文档内 clause 重名    %d  %s"
          % (len(dup_clause),
             "✔ 身份键唯一" if not dup_clause else "★ 会踩版本冲突关（同 key 只留一个）"))

    # ---------- 解析失败 ----------
    checks.append(("解析失败文件", not errors, len(errors)))
    print("\n【解析】")
    print("  解析失败文件      %d  %s"
          % (len(errors), "✔ 没有" if not errors else "★ 有（见 errors 列表）"))

    # ---------- 清洗账 ----------
    print("\n【清洗账】加载时动过的刀（不可逆 —— 这几个数应该长期稳定）")
    if not collector.ops:
        print("  （没动过刀）")
    for op in (CLEAN_OP_FOOTER, CLEAN_OP_TRUNCATE, CLEAN_OP_PREAMBLE):
        n, chars = collector.ops.get(op, (0, 0))
        print("  %s：%d 块，共 %d 字" % (op, n, chars))
    print("  ★ 哪天这里的数字变了，就是清洗规则又被动了 —— 这是全绿快照给不了的信息。")

    # ---------- 每个文档 ----------
    print("\n【按文档】")
    for name, n in sorted(by_doc.items()):
        sub = [len(d["text"]) for d in docs if d["doc"] == name]
        print("  %-28s %3d 块   最短 %-4d 中位 %-4d 最长 %d"
              % (name, n, min(sub), int(statistics.median(sub)), max(sub)))

    # ---------- 极端样本 ----------
    print("\n【最短的 %d 块】（最容易是碎屑/标题党）" % samples)
    for d in sorted(docs, key=lambda x: len(x["text"]))[:samples]:
        print("  %-4d字  %s｜%s" % (len(d["text"]), d["doc"], d["clause"]))

    print("\n【最长的 %d 块】（最可能混进无关内容）" % samples)
    for d in sorted(docs, key=lambda x: -len(x["text"]))[:samples]:
        print("  %-4d字  %s｜%s" % (len(d["text"]), d["doc"], d["clause"]))

    # ---------- 结论 ----------
    bad = [(name, n) for name, ok, n in checks if not ok]
    print("\n" + "=" * 70)
    if bad:
        print("结论：%d 项没通过 —— %s"
              % (len(bad), "；".join("%s（%d）" % (name, n) for name, n in bad)))
    else:
        print("结论：以上 %d 项全部通过，没查到已知类型的数据层问题。" % len(checks))
        print("      （注意这只说明「没有已知类型的病」，不等于「语料够用」——")
        print("        语料规模的天花板见 README 的「🔒 数据边界」一节）")
    print("=" * 70)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
