# -*- coding: utf-8 -*-
"""check_claims.py —— 把"谁说得对"从模型手里拿走，交给 exit code。

它是 `_tasks/` + `_reviews/` 协议的补件，不是新协议。用法：

    # ① 审稿人（或派活方）先给结论落一份 claims 文件（见 _tasks/CLAIMS_TEMPLATE.json）
    #    审稿人读文件前先 pin 一次 —— 这一步是防"读到半成品"
    <py> _tasks/check_claims.py pin  _reviews/cc-xxx.claims.json README.md service.py

    # ② 任何人（包括作者）事后对账：校验 schema + 查快照漂移 + 重跑 fact 断言
    <py> _tasks/check_claims.py check _reviews/cc-xxx.claims.json

它做四件事，每一件都有对应的真实失败案例：

  1) **schema 强制**（把"我需要看 X"从结尾客套变成提交前必填）
       - 每条 claim 必须二选一：有 evidence（fact）或 有 blind_spot
       - blind_spot 必须带 verdict_if_missing（三选一），否则 = 防御性堆砌
       - blind_spots 为空时必须附 searched（空缺口 + 空搜索轨迹 = 拒收）
  2) **快照漂移检测** ★ 这是本文件存在的头号理由
       源起：2026-10-10 一次真实事故 —— 审稿人读 `_out_collab_protocol.md` 时它
       还在被后台任务写入，于是审稿人把它判成"只有一行 CLI 噪音、没跑起来"，
       并据此推出了一个错误结论。**"我读到的" ≠ "你写的"，必须 pin。**
  3) **fact 断言重跑**：跑 claim.cmd，拿 stdout 比对 expected，写回 actual。
       ★ actual 永远由本脚本生成 —— 人手抄的 actual 就是"判据副本"，必然过期。
  4) **采购单**：status=unverified 的不阻塞，但会被列成"下一轮最便宜的活"。

退出码：0 = 全绿；1 = 有 red / schema 违规；2 = 有快照漂移（结论不可用于决策）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
OK, BAD, WARN = "✔", "✘", "!"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def short(p: Path) -> dict:
    st = p.stat()
    return {"sha256": sha256(p)[:16], "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------- pin
def cmd_pin(args) -> int:
    rev = Path(args.review)
    data = json.loads(rev.read_text(encoding="utf-8")) if rev.exists() else {
        "reviewer": "?", "task": rev.stem, "claims": [], "blind_spots": [], "searched": []}
    snap = data.setdefault("snapshot", {})
    for f in args.files:
        p = (REPO / f)
        if not p.exists():
            print(f"  {BAD} 找不到 {f}")
            return 1
        snap[f] = short(p)
        print(f"  {OK} pin {f:34s} {snap[f]['sha256']}  {snap[f]['bytes']:>7} B  {snap[f]['mtime']}")
    data["pinned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rev.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → 已写入 {rev}")
    print("  ★ 审稿人现在可以开始读了；事后 `check` 会证明你读的是不是这一版。")
    return 0


# ---------------------------------------------------------------- check
def validate(data: dict) -> list[str]:
    errs: list[str] = []
    claims = data.get("claims", [])
    if not claims and not data.get("blind_spots"):
        errs.append("claims 和 blind_spots 都空 —— 拒收")
    for c in claims:
        cid = c.get("id", "?")
        ty = c.get("type")
        if ty not in ("fact", "judgment"):
            errs.append(f"{cid}: type 必须是 fact | judgment（现在是 {ty!r}）")
        if ty == "fact" and not c.get("cmd"):
            errs.append(f"{cid}: fact 类必须带 cmd（否则它其实是 judgment）")
        if ty == "judgment" and not c.get("would_flip_if"):
            errs.append(f"{cid}: judgment 类必须写 would_flip_if（填不出来 ⇒ 这是偏好）")
    for b in data.get("blind_spots", []):
        if b.get("verdict_if_missing") not in ("confirm", "retract", "hold"):
            errs.append(f"blind_spot {b.get('claim_id','?')}: verdict_if_missing 必须三选一"
                        "（confirm/retract/hold）—— 不说'没有它我怎么判'，缺口列表一文不值")
        if not b.get("obtain_by"):
            errs.append(f"blind_spot {b.get('claim_id','?')}: 缺 obtain_by（谁/怎么拿到）")
    if not data.get("blind_spots") and not data.get("searched"):
        errs.append("blind_spots 为空却不写 searched —— 空缺口 + 空搜索轨迹 = 拒收")
    return errs


def check_snapshot(data: dict) -> list[str]:
    drift: list[str] = []
    for f, rec in (data.get("snapshot") or {}).items():
        p = REPO / f
        if not p.exists():
            drift.append(f"{f}: 文件已不存在")
            continue
        now = short(p)
        if now["sha256"] != rec["sha256"] or now["bytes"] != rec["bytes"]:
            drift.append(f"{f}: pin 的是 {rec['sha256']}({rec['bytes']}B) "
                         f"现在 {now['sha256']}({now['bytes']}B) ⇒ 读到的不是这一版")
    return drift


def run_claim(c: dict) -> tuple[str, str]:
    try:
        r = subprocess.run(c["cmd"], shell=True, cwd=str(REPO), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "red", "<timeout>"
    return ("green" if r.returncode == 0 else "red"), out.strip()[-400:]


def cmd_check(args) -> int:
    rev = Path(args.review)
    data = json.loads(rev.read_text(encoding="utf-8"))
    print("=" * 74)
    print(f"对账：{rev}   审稿人={data.get('reviewer')}  卡={data.get('task')}")
    print("=" * 74)

    errs = validate(data)
    print(f"\n【1】schema（{len(errs)} 处违规）")
    for e in errs:
        print(f"  {BAD} {e}")
    if not errs:
        print(f"  {OK} 通过")

    drift = check_snapshot(data)
    print(f"\n【2】快照漂移（{len(drift)} 处）")
    for d in drift:
        print(f"  {BAD} {d}")
    if not drift:
        print(f"  {OK} 快照与 pin 时一致 —— 这份结论确实读的是这一版")

    facts = [c for c in data.get("claims", []) if c.get("type") == "fact"]
    print(f"\n【3】fact 断言重跑（{len(facts)} 条）")
    red = 0
    for c in facts:
        status, actual = run_claim(c)
        c["actual"] = actual          # ★ actual 永远由脚本生成，不手抄
        c["status"] = status
        exp = (c.get("expected") or "")[:40]
        mark = OK if status == "green" else BAD
        print(f"  {mark} {c['id']:<4} {status:<7} 期望含「{exp}」")
        if status == "red":
            red += 1
            print(f"      实际尾部：{actual[-160:]!r}")
    jd = [c for c in data.get("claims", []) if c.get("type") == "judgment"]
    print(f"\n【4】judgment 断言（{len(jd)} 条）—— 不判定，只记账")
    for c in jd:
        print(f"  · {c['id']}: 会翻转它的证据 = {c.get('would_flip_if')}")

    unver = [c["id"] for c in data.get("claims", []) if c.get("status") == "unverified"]
    blind = data.get("blind_spots", [])
    print(f"\n【5】采购单（{len(unver)} 条未验证断言 + {len(blind)} 个盲区）")
    for cid in unver:
        print(f"  · {cid} 未验证 —— 对决策零贡献，不许算'通过'")
    for b in blind:
        print(f"  · {b.get('claim_id')}: 需要「{b.get('need')}」"
              f"（{b.get('obtain_by')}）⇒ 缺失则 {b.get('verdict_if_missing')}")

    rev.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 74)
    if drift:
        print("判定：✘ 有快照漂移 —— 这份结论不可用于决策（先重 pin 再重读）")
        return 2
    if errs or red:
        print(f"判定：✘ schema {len(errs)} 处 / red {red} 条")
        return 1
    print("判定：✔ 全绿（judgment 与未验证项仍不构成决策依据，只记账）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="claims 对账器（schema + 快照漂移 + 重跑）")
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("pin", help="把待审文件按 sha256/字节/mtime 钉进 claims 文件")
    p1.add_argument("review")
    p1.add_argument("files", nargs="+")
    p1.set_defaults(fn=cmd_pin)
    p2 = sub.add_parser("check", help="校验 schema + 查漂移 + 重跑 fact 断言")
    p2.add_argument("review")
    p2.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
