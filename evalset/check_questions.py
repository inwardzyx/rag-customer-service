# -*- coding: utf-8 -*-
"""evalset/check_questions.py —— 校验一套评测题自不自洽。

为什么需要它：held-out 那套题是**盲出题人**写的（它只看到校规底本，没看过仓库），
所以它一定会写出**在库里不存在**的条款名 —— 不是它不认真，是它抄底本时可能
抄错一个字。而 gold 里有一个字不对，这道题就永远判为"没召回"，
于是整套 held-out 静默失真：看起来是"模型不行"，其实是标注错了。

★ 这条判据与 `tests/test_eval_set.py::test_answerable_gold_exists_in_kb` 同源：
  对主评测集用测试守，对 held-out 用这个脚本守 —— 两边查的是同一件事。

跑法：
    <py> evalset/check_questions.py --questions evalset/questions.json
    <py> evalset/check_questions.py --questions evalset/heldout/questions.json \\
                                   --against evalset/questions.json
退出码：0 = 全通过；1 = 有问题
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kb.loader import load_documents  # noqa: E402


def norm(s: str) -> str:
    """去掉标点/空白后比较 —— 用来抓"换个标点就算新题"的重复。"""
    return re.sub(r"[\s，。？！、,.?!：:；;\"'“”‘’（）()《》]", "", s or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--against", default=None,
                    help="另一套题（通常是主评测集）：查问题是否重复")
    A = ap.parse_args()

    qp = Path(A.questions)
    errs: list[str] = []
    warns: list[str] = []

    if not qp.exists():
        print(f"✗ 找不到 {qp}")
        return 1
    try:
        data = json.loads(qp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"✗ {qp} 不是合法 JSON：{e}")
        return 1

    qs = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(qs, list) or not qs:
        print("✗ 题目列表为空或格式不对（应为 {\"questions\": [...]} 或直接是数组）")
        return 1

    # ---- 库里真实存在的 (doc, clause)，以 loader 为准（不是以文件名猜）----
    docs, _ = load_documents(REPO / "docs")
    exist = {(d["doc"], d["clause"]) for d in docs}
    docnames = {d["doc"] for d in docs}

    seen_ids: set[str] = set()
    seen_q: dict[str, str] = {}
    kinds = Counter()
    for i, q in enumerate(qs, 1):
        tag = q.get("id") or f"#{i}"
        t = q.get("type")
        question = (q.get("question") or "").strip()

        if not q.get("id"):
            errs.append(f"{tag}: 缺 id")
        elif q["id"] in seen_ids:
            errs.append(f"{tag}: id 重复")
        else:
            seen_ids.add(q["id"])

        if t not in ("answer", "refuse"):
            errs.append(f"{tag}: type 必须是 answer|refuse，现在是 {t!r}")
        if not question:
            errs.append(f"{tag}: question 为空")
        else:
            k = norm(question)
            if k in seen_q:
                errs.append(f"{tag}: 与 {seen_q[k]} 是同一道题（去标点后相同）")
            else:
                seen_q[k] = tag
            if len(question) < 4:
                warns.append(f"{tag}: 问题过短（{len(question)} 字），可能不像真实提问")

        if t == "answer":
            kinds["answer"] += 1
            gold = q.get("gold")
            if not (isinstance(gold, list) and len(gold) == 2 and all(isinstance(x, str) for x in gold)):
                errs.append(f"{tag}: answer 题的 gold 必须是 [文件名, 条款名] 两个字符串，现在是 {gold!r}")
            else:
                doc, clause = gold
                if doc not in docnames:
                    errs.append(f"{tag}: gold 的文件名 {doc!r} 不在库里（库里有 {len(docnames)} 份）")
                elif (doc, clause) not in exist:
                    errs.append(f"{tag}: gold 的条款名 {clause!r} 在 {doc} 里不存在"
                                f"（★ 盲出题最常犯这个：抄错一个字，整道题永远判为没召回）")
            if q.get("multi_hop"):
                kinds["multi_hop"] += 1
        elif t == "refuse":
            kinds["refuse"] += 1
            if q.get("gold") not in (None, [], ()):
                errs.append(f"{tag}: refuse 题的 gold 必须是 null")
            why = (q.get("why_forced_to_refuse") or "").strip()
            if not why:
                errs.append(f"{tag}: refuse 题必须写 why_forced_to_refuse（说不出为什么，它可能不是 refuse 题）")
            elif len(why) < 8:
                warns.append(f"{tag}: why_forced_to_refuse 太短，举证不足：{why!r}")

    # ---- 和另一套题查重 ----
    if A.against:
        ap_ = Path(A.against)
        if ap_.exists():
            other = json.loads(ap_.read_text(encoding="utf-8"))
            oqs = other.get("questions") if isinstance(other, dict) else other
            oset = {norm(q.get("question") or ""): q.get("id") for q in oqs}
            for q in qs:
                k = norm(q.get("question") or "")
                if k in oset:
                    errs.append(f"{q.get('id')}: 与主评测集 {oset[k]} 的问题完全重合 "
                                f"—— held-out 的意义就是「没被问过」")
        else:
            warns.append(f"--against 指定的 {ap_} 不存在，跳过查重")

    # ---- 报告 ----
    print("=" * 70)
    print(f"题目文件 {qp}")
    print(f"  总数 {len(qs)}　answer {kinds['answer']}（其中 multi-hop {kinds['multi_hop']}）"
          f"　refuse {kinds['refuse']}")
    print("=" * 70)
    for w in warns:
        print(f"  ⚠ {w}")
    if errs:
        print(f"\n✗ {len(errs)} 个错误：")
        for e in errs:
            print(f"  · {e}")
        return 1
    print(f"\n✓ 全通过（{len(warns)} 条提示）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
