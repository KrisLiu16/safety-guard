#!/usr/bin/env bash
# 设计方（云端）一侧：轮询 origin/main，出现任何不是 [design] 开头的新提交（执行方反馈、用户提交）就退出，
# 退出会唤醒设计方会话。设计方每次处理完都重新启动它。
set -u
cd "$(dirname "$0")/.." || exit 1
INTERVAL="${INTERVAL:-60}"
git fetch -q origin main
base=$(git rev-parse origin/main)
while true; do
  sleep "$INTERVAL"
  git fetch -q origin main 2>/dev/null || continue
  others=$(git log --format='%h %s' "$base..origin/main" | grep -v '^[0-9a-f]* \[design\]')
  if [ -n "$others" ]; then
    echo "main 上有新提交："
    echo "$others"
    exit 0
  fi
done
