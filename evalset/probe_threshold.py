# -*- coding: utf-8 -*-
"""evalset/probe_threshold.py —— 拒答阈值 VEC_REJECT_THRESHOLD 的扫描与选择

**这个实验要回答的问题：0.55 这个数还成立吗？**

★ 为什么现在才有这个脚本（它一直缺着，是个真缺口）：
  `README` 写着「0.55 是在旧的 7 条玩具语料上量的」，后来又说「阈值余量本就薄」。
  但 `probe_sensitivity.py` 只扫了 RRF 常数和粗捞池，**从来没扫过阈值本身** ——
  也就是说 0.55 是【拍出来的】，从来没被任何实验检验过。
  语料从 68 块涨到 263 块之后，这个数更不可能自动还成立。

★ 为什么阈值不能"看着数据调"（这是本脚本存在的意义）：
  拒答阈值是在【应答题要放行】和【拒答题要拦住】之间划一刀。
  同一个数要同时满足两边的要求，所以它不是"取一个最优值"，
  而是"看有没有一个区间，两边都满足"。
  → 输出不是一个"推荐值"，是【可行区间 + 每个阈值的具体错法】。
  没有可行区间，就是"纯向量路这条路已经走不通了"——
  那是个结论，不是个数字。

跑法（仓库根目录，零 API 成本、不需要 key）：
    D:\\Python-project\\.venv\\Scripts\\python.exe evalset/probe_threshold.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evalset" / "questions.json"
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("LANGSMITH_TRACING", "false")

import service as svc                                # noqa: E402


# 扫哪些候选阈值。★ 用 0.01 为步长而不是 0.05：
#   步长太粗会漏掉"刚好卡在可行区间边界上"的那个数，
#   而阈值的实际含义就是边界 —— 粗步长等于把结论算糊。
STEP = 0.01
LO, HI = 0.30, 0.95


def main() -> int:
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        qs = json.load(f)
    answerable = [q for q in qs if q["type"] == "answer"]
    refuse = [q for q in qs if q["type"] == "refuse"]

    svc.rag.startup()
    print(f"[前置] 入库 {len(svc.rag.chunks)} 块，拦下 {len(svc.rag.rejected)} 块")
    print(f"[前置] 应答题 {len(answerable)} 道 / 拒答题 {len(refuse)} 道\n")

    # 每道题的 top-1 向量分（拒答判定只看 top-1，所以只取这一路）
    a_scores = {q["id"]: max(s for _, s in svc.rag.search(q["question"], k=5))
                for q in answerable}
    r_scores = {q["id"]: max(s for _, s in svc.rag.search(q["question"], k=5))
                for q in refuse}
    id2q = {q["id"]: q for q in qs}

    print("=" * 74)
    print("一、两组分数分布（阈值要划在这两组之间）")
    print("=" * 74)
    for name, d in (("应答题（要放行，分应 ≥ 阈值）", a_scores),
                    ("拒答题（要拦住，分应 < 阈值）", r_scores)):
        vals = sorted(d.values())
        print(f"\n{name}：")
        for qid in sorted(d, key=lambda k: d[k]):
            print(f"   {qid}  {d[qid]:.4f}  {id2q[qid]['question']}")
        print(f"   → 最低 {vals[0]:.4f}／最高 {vals[-1]:.4f}　中位 {vals[len(vals)//2]:.4f}")

    lo_ans = min(a_scores.values())
    hi_ref = max(r_scores.values())
    print("\n" + "=" * 74)
    print("二、可行的阈值区间")
    print("=" * 74)
    print(f"  要放行全部应答题：阈值 ≤ {lo_ans:.4f}")
    print(f"  要拦住全部拒答题：阈值 > {hi_ref:.4f}")
    if lo_ans <= hi_ref:
        print("\n  ★★ 两组【交叉】：没有任何阈值能同时满足两边。")
        print("     → 纯向量路（use_rerank=false）已经走不通，不是调阈值能救的。")
        print("     → 要么把拒答判定交给 rerank（产品默认路径），要么补语料/改判据。")
    else:
        print(f"\n  → 可行区间：({hi_ref:.4f}, {lo_ans:.4f}]　"
              f"宽度 {lo_ans - hi_ref:.4f}")

    print("\n" + "=" * 74)
    print("三、逐阈值扫描（看每档具体错在哪几题）")
    print("=" * 74)
    print(f"{'阈值':>6}  {'应答题放行':>9}  {'拒答拦住':>8}  {'错法'}")
    feasible = []
    th = LO
    while th <= HI + 1e-9:
        t = round(th, 4)
        passed = sum(1 for v in a_scores.values() if v >= t)
        blocked = sum(1 for v in r_scores.values() if v < t)
        errs = []
        for qid, v in a_scores.items():
            if v < t:
                errs.append(f"{qid}误拒")
        for qid, v in r_scores.items():
            if v >= t:
                errs.append(f"{qid}误放")
        perfect = (passed == len(a_scores) and blocked == len(r_scores))
        if perfect:
            feasible.append(t)
        mark = " ★全对" if perfect else ""
        print(f"{t:>6.2f}  {passed:>4}/{len(a_scores):<4}  {blocked:>3}/{len(r_scores):<3}  "
              f"{','.join(errs[:4]) if errs else '—'}{mark}")
        th += STEP

    print("\n" + "=" * 74)
    print("四、结论")
    print("=" * 74)
    cur = svc.VEC_REJECT_THRESHOLD
    print(f"  当前 VEC_REJECT_THRESHOLD = {cur}")
    if feasible:
        print(f"  ★ 全对区间：{feasible[0]:.2f} ~ {feasible[-1]:.2f}"
              f"（{len(feasible)} 个候选）")
        if lo_ans <= cur <= hi_ref:
            print("  → 当前阈值【不在】全对区间内。")
        elif cur < feasible[0]:
            print(f"  → 当前阈值【偏低】，会误放行；建议 ≥ {feasible[0]:.2f}")
        else:
            print(f"  → 当前阈值【偏高】，会误拒；建议 ≤ {feasible[-1]:.2f}")
    else:
        print("  ★★ 不存在全对阈值 —— 纯向量路在这 20 题上【无解】。")
        print("     这不是阈值没调好，是【判据本身不够】。")
        print("     产品默认路径（use_rerank=true，精排分<5 拒答）不受此影响，")
        print("     所以线上不一定要改阈值 —— 但 README 里 0.55 的论证方式必须改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
