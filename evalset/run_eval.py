# -*- coding: utf-8 -*-
"""evalset/run_eval.py —— 评测集运行器（CLI）

测的是「效果」，不是「代码不崩」：
  · 应答题：Recall@5（gold 条款是否进粗捞前 5）+ 漏答率（knowledge_hit 是否为 false）
  · 拒答题：拒答准确率（knowledge_hit 是否为 false）

两层：
  · 离线层（默认）：use_rerank=false，拒答靠 VEC_REJECT_THRESHOLD=0.55，零 API 成本，可进 CI
  · 在线层（--with-llm）：use_rerank=true（产品默认），拒答靠精排分 < 5，需 DEEPSEEK_API_KEY

★ recalled 直接调 rag.search() 拿，不拿 /chat 的 sources ——
  否则「因拒答 sources 为空」会被算成「没召回」，检索问题和阈值问题搅在一起。

跑法（仓库根目录）：
  离线（快，不要 key）：
    python evalset/run_eval.py
  在线（需 DEEPSEEK_API_KEY）：
    python evalset/run_eval.py --with-llm
  把结果落盘到 evalset/report.md：
    python evalset/run_eval.py --report
"""
import argparse
import logging
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "evalset" / "questions.json"
sys.path.insert(0, str(REPO_ROOT))

import service as svc                                  # noqa: E402

logger = logging.getLogger(__name__)


def load_questions():
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_gold(q, k=5):
    """gold 条款是否进粗捞前 k。直接用 rag.search()，不拿 /chat 的 sources。"""
    cands = svc.rag.search(q["question"], k=k)
    recalled = {(c["doc"], c["clause"]) for c, _ in cands}
    return tuple(q.get("gold") or []) in recalled


def top_vec_score(q, k=5):
    cands = svc.rag.search(q["question"], k=k)
    return max((s for _, s in cands), default=0.0)


def knowledge_hit(q, with_llm):
    resp = svc.chat(svc.ChatRequest(question=q["question"], use_rerank=with_llm))
    return resp.knowledge_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="在线层：用 rerank（产品默认），需 DEEPSEEK_API_KEY")
    ap.add_argument("--report", action="store_true",
                    help="结果写入 evalset/report.md")
    args = ap.parse_args()

    os.environ.setdefault("LANGSMITH_TRACING", "false")

    qs = load_questions()
    logger.info("加载模型 + 建库……")
    svc.rag.startup()
    logger.info(f"入库 {len(svc.rag.chunks)} 块，拦下 {len(svc.rag.rejected)} 块")

    answerable = [q for q in qs if q["type"] == "answer"]
    refuse = [q for q in qs if q["type"] == "refuse"]

    # 检索层：Recall@5
    recalled = [(q["id"], recall_gold(q)) for q in answerable]
    recall_count = sum(1 for _, r in recalled)

    # 应答层：漏答率（knowledge_hit 是否为 false）
    miss = [(q["id"], knowledge_hit(q, args.with_llm)) for q in answerable]
    miss_count = sum(1 for _, hit in miss if not hit)

    # 拒答层：拒答准确率
    refused = [(q["id"], knowledge_hit(q, args.with_llm)) for q in refuse]
    refused_count = sum(1 for _, hit in refused if not hit)

    # 阈值余量（只看离线层的向量分）
    if not args.with_llm:
        min_ans = min(top_vec_score(q) for q in answerable)
        max_ref = max(top_vec_score(q) for q in refuse)
    else:
        min_ans = max_ref = None

    layer = "在线层(rerank)" if args.with_llm else "离线层(向量分0.55)"
    lines = [f"# 评测报告（{layer}）", ""]
    lines.append(f"- 应答题数：{len(answerable)}　Recall@5：{recall_count}/{len(answerable)}")
    lines.append(f"- 漏答率：{miss_count}/{len(answerable)}（knowledge_hit==false 才算漏答）")
    lines.append(f"- 拒答题数：{len(refuse)}　拒答准确率：{refused_count}/{len(refuse)}")
    if not args.with_llm:
        note_ans = "安全（≥0.55）" if min_ans >= 0.55 else "⚠ 已低于阈值"
        if max_ref < 0.55:
            note_ref = "安全（<0.55）"
        else:
            note_ref = (f"⚠ 有边界题向量分 {max_ref:.4f} 越过 0.55 —— "
                        f"纯向量路已知边界，产品默认 rerank 路径会拦住它")
        lines.append(
            f"- 阈值余量：应答题最低向量分 {min_ans:.4f}（{note_ans}）／ "
            f"拒答题最高向量分 {max_ref:.4f}（{note_ref}）")
    lines.append("")
    lines.append("> 注：离线层（use_rerank=false）拒答只靠向量分 0.55，余量本就薄；"
                "产品默认走 rerank，精排分 < 5 才拒答，边界题在那里会被拦住。"
                "两层数字不是包含关系，请并列看。")
    lines.append("")
    lines.append("## 应答题 Recall@5 明细")
    for qid, r in recalled:
        qt = next(q["question"] for q in answerable if q["id"] == qid)
        lines.append(f"- {qid}: {'OK' if r else 'MISS'}  {qt}")
    lines.append("")
    lines.append("## 拒答题 拒答明细")
    for qid, hit in refused:
        qt = next(q["question"] for q in refuse if q["id"] == qid)
        flag = "boundary" if next(q.get("boundary") for q in refuse if q["id"] == qid) else "clear"
        lines.append(f"- {qid}: {'拒答' if not hit else '误答'} [{flag}]  {qt}")
    report = "\n".join(lines)

    logger.info(report)
    if args.report:
        out = REPO_ROOT / "evalset" / "report.md"
        out.write_text(report, encoding="utf-8")
        logger.info(f"已写入 {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    t0 = time.time()
    main()
    logger.info(f"耗时 {time.time() - t0:.1f}s")
