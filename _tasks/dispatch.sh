#!/usr/bin/env bash
# ============================================================
# dispatch.sh —— 把一张任务卡【同时】派给 cc 和 DSH
# ============================================================
# 为什么要有这个脚本（而不是在对话里口头派活）：
#   1. 任务书和判据落在_tasks/<slug>.md，不活在我的上下文里
#      —— 我一停，别人接不上；我复述漏一句，审的就不是同一件事。
#   2. 两份结论各自落盘到_reviews/，不经过我转述
#      —— 你能直接看到原话，包括我说错的部分。
#   3. 两个 agent 同时起，不互相等待。
#
# 用法：
#   bash _tasks/dispatch.sh <slug>            派给 cc + DSH
#   bash _tasks/dispatch.sh <slug> --cc       只派 cc
#   bash _tasks/dispatch.sh <slug> --dsh      只派 DSH
#
# 前提：先按 _tasks/TEMPLATE.md 写好 _tasks/<slug>.md
# ============================================================

set -uo pipefail

REPO="D:/_review_rag-cs"
CARD_SLUG="${1:?用法: dispatch.sh <slug> [--cc|--dsh]}"
ONLY="${2:-}"
CARD="$REPO/_tasks/$CARD_SLUG.md"
OUT_DIR="$REPO/_reviews"
WORK="D:/dsh-work"

# ---------- 自检：卡在不在 ----------
if [ ! -f "$CARD" ]; then
  echo "任务卡不存在: $CARD" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

CC_OUT="$OUT_DIR/cc-$CARD_SLUG.md"
DSH_OUT="$OUT_DIR/dsh-$CARD_SLUG.md"
CC_LOG="$OUT_DIR/_cc-$CARD_SLUG.log"
DSH_LOG="$OUT_DIR/_dsh-$CARD_SLUG.log"

# ---------- 卡内容 ----------
# 附一段"身份说明"：谁在读、写到哪、不许改代码。
# ★ 用 cat 让它自己读卡，不把正文塞进命令行 —— 这是防截断的关键
#   （命令行传多行会被 claude.cmd 在第一个 \n 处截断，见 cc-collab-playbook）
FOOTER_CC="

---
以上是任务卡全文，你只读不写：
- 不要修改/创建/删除任何仓库文件，不要 git add / git commit
- 把结论【全文打印在你的回复里】，不要写入任何文件（外层 shell 会负责落盘）
- 动手之前先说出上面「一、要回答的问题」那一节的原句，证明你确实读到了卡
- 结论要短：给结论 + 该改哪几行，不要复述卡的内容、不要贴完整文件；但发现的问题照报
- 每条问题带 文件:行号，每条数字带「怎么算的」"

FOOTER_DSH="

---
以上是任务卡全文，你只读不写：
- 不要修改/创建/删除任何仓库文件，不要 git add / git commit
- 把结论【全文打印在你的回复里】，不要写入任何文件（外层 shell 会负责落盘）
- 注意：你的工作目录是 $WORK，任务卡里所有路径都是绝对路径，直接用
- 结论要短：给结论 + 该改哪几行，不要复述卡的内容、不要贴完整文件；但发现的问题照报
- 每条问题带 文件:行号，每条数字带「怎么算的」"

# ---------- 组装 prompt（保持单行，防截断） ----------
CC_PROMPT="用 Bash 工具执行: cat \"$CARD\" —— 读到的就是你的任务卡全文，照它执行。$FOOTER_CC"
DSH_PROMPT="$(cat "$CARD")$FOOTER_DSH"

CC_EXE="/c/Users/zyx16/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe"

# ---------- 起 cc ----------
launch_cc() {
  # ★ 直接调 claude.exe 绕开 claude.cmd 的多行截断
  # ★ --allowedTools 必须同时给 Bash 和 PowerShell（它会自己挑，漏一个废一半）
  timeout 900 "$CC_EXE" -p "$CC_PROMPT" \
      --allowedTools "Bash,PowerShell,Read,Grep,Glob" \
      --permission-prompts none \
      > "$CC_LOG" 2>/dev/null < /dev/null
  echo "exit=$?" >> "$CC_LOG"
}

# ---------- 起 DSH ----------
launch_dsh() {
  # dsh.sh 自带 DSH_PERMISSION_MODE=danger-full-access 和 600s 超时
  bash /d/dsh-work/dsh.sh -f "$CARD" -o "$DSH_OUT" --raw \
      > "$DSH_LOG" 2>&1
  echo "exit=$?" >> "$DSH_LOG"
}

# ---------- 并行 ----------
echo "▸ 任务卡: $CARD"
case "$ONLY" in
  --cc)  echo "▸ 只派 cc";  launch_cc ;;
  --dsh) echo "▸ 只派 DSH"; launch_dsh ;;
  *)     echo "▸ 同时派 cc + DSH（互不知情）"
         launch_cc &
         CC_PID=$!
         launch_dsh &
         DSH_PID=$!
         wait $CC_PID;  wait $DSH_PID ;;
esac
echo "▸ 两个都结束了"

# ---------- 落盘校验：别信 AI 自报，只看字节数 ----------
# ★ ★量错对象是我自己犯过的 bug：DSH 用 -o 把结论写进产出文件、
#   log 里只有进度行（170 字节），按 log 判会误报"没跑起来"。
#   ⇒ 两个 agent 的产出位置不同，必须各量各的：
#     cc  → stdout 重定向进 _cc-<slug>.log，再净成正式文件
#     DSH → dsh.sh -o 直接写产出文件，log 只是进度
#判据：
#     < 500 B = 只有噪音，进程没起来
#     800B~2KB = 多半只收到开头（prompt 被截断）
#     > 2 KB  = 真拿到产出了
report_cc() {
  local log="$1" out="$2"
  [ -f "$log" ] || { echo "✗ cc 没有日志: $log"; return 1; }
  local bytes; bytes=$(wc -c < "$log" | tr -d ' ')
  if [ "$bytes" -lt 500 ]; then
    echo "✗ cc 日志只有 $bytes 字节 —— 没跑起来（< 500B）。别改prompt 写法，先看日志"
    return 1
  fi
  # 剔掉开头两行 CLI 噪音（按前缀匹配，不按行号）
  grep -v -e '^Warning: no stdin data received' \
          -e '^\[claude-code:unrecognized_model\]' \
          -e '^exit=' "$log" > "$out" 2>/dev/null
  local clean; clean=$(wc -c < "$out" | tr -d ' ')
  echo "✓ cc → $out（净 $clean 字节 / 原始 $bytes）"
  [ "$clean" -lt 300 ] && echo "  ⚠️ 净输出偏小，可能只拿到了开头，建议看一眼内容"
  return 0
}

report_dsh() {
  local out="$1" log="$2"
  # ★量产出文件，不是 log
  [ -f "$out" ] || { echo "✗ DSH 没有产出文件: $out"; return 1; }
  local bytes; bytes=$(wc -c < "$out" | tr -d ' ')
  if [ "$bytes" -lt 500 ]; then
    echo "✗ DSH 产出只有 $bytes 字节 —— 没跑起来（< 500B）"
    [ -f "$log" ] && echo "  日志：$log"
    return 1
  fi
  echo "✓ DSH → $out（$bytes 字节）"
  [ "$bytes" -lt 1000 ] && echo "  ⚠️ 产出偏小，可能只拿到了开头，建议看一眼内容"
  return 0
}

report_cc  "$CC_LOG"  "$CC_OUT"
report_dsh "$DSH_OUT" "$DSH_LOG"

echo "--------------------------------------------------"
echo "两份结论独立产出、互不看对方。下一步：自己对比它们"
echo "在哪一条上分歧—— 分歧项才值得占用注意力，一致项直接落地。"
echo "（别让我转述结论：打开 $OUT_DIR 直接看原话。）"