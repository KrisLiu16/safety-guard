#!/usr/bin/env bash
# 执行方（Mac）一侧：轮询 origin/main，设计方推送 [design] 提交后启动一次执行方 agent。
#
# 用法（在仓库根目录）：
#   EXECUTOR_CMD='claude -p' collab/watch_main.sh      # 或 EXECUTOR_CMD='codex exec'
# 可选环境变量：
#   INTERVAL   轮询间隔秒数，默认 60
#   HEARTBEAT  没有新设计时，每隔多少秒也启动一次执行方，用来跟进“进行中”的长作业；默认 1800，0 表示关闭
# 暂停：main 上存在 collab/PAUSE 文件时只轮询、不启动 agent。
# 状态、锁和日志都放在 .git/ 下，不会进入提交。兼容 macOS 自带的 bash 3.2。
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${INTERVAL:-60}"
HEARTBEAT="${HEARTBEAT:-1800}"
STATE="$REPO/.git/collab_last_design"
LOCK="$REPO/.git/collab_watch.lock"
LOG="$REPO/.git/collab_watch.log"
: "${EXECUTOR_CMD:?请设置 EXECUTOR_CMD，例如 EXECUTOR_CMD='claude -p'}"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "已有一个 watch_main.sh 在运行（锁 $LOCK）。确认没有后删除该目录再启动。" >&2
  exit 1
fi
trap 'rmdir "$LOCK"' EXIT
cd "$REPO" || exit 1

run_executor() {
  # $1 = 触发原因
  if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
    log "当前不在 main 分支，跳过（$1）"; return 1
  fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    log "工作区有未提交改动，跳过（$1）"; return 1
  fi
  if ! git pull -q --ff-only origin main; then
    log "git pull --ff-only 失败，跳过（$1）"; return 1
  fi
  log "启动执行方：$1"
  $EXECUTOR_CMD "$(cat collab/EXECUTOR_PROMPT.md)" >>"$LOG" 2>&1
  log "执行方结束，退出码 $?"
}

last_run=$(date +%s)
if [ ! -f "$STATE" ]; then
  # 第一次启动：先跑一次，处理已经在排队的任务单。
  git fetch -q origin main && git rev-parse origin/main >"$STATE"
  run_executor "首次启动" && last_run=$(date +%s)
fi

while true; do
  sleep "$INTERVAL"
  if ! git fetch -q origin main; then log "fetch 失败，稍后重试"; continue; fi
  if git cat-file -e origin/main:collab/PAUSE 2>/dev/null; then continue; fi
  head=$(git rev-parse origin/main)
  seen=$(cat "$STATE")
  if git merge-base --is-ancestor "$seen" "$head" 2>/dev/null; then
    designs=$(git log --format=%s "$seen..$head" | grep -c '^\[design\]')
  else
    designs=1  # 历史被改写过，保守地当作有新设计
  fi
  now=$(date +%s)
  if [ "$designs" -gt 0 ]; then
    run_executor "main 上有 $designs 个新的 [design] 提交" && echo "$head" >"$STATE" && last_run=$(date +%s)
  elif [ "$HEARTBEAT" -gt 0 ] && [ $((now - last_run)) -ge "$HEARTBEAT" ]; then
    run_executor "定时跟进" ; last_run=$(date +%s)
  fi
done
