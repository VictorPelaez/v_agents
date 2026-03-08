#!/bin/bash
# live-forward-v4.2 launcher
# Usage: ./launch.sh [--strategy STRATEGY_FILE] [--label LABEL]

set -e
SKILL_DIR=$(dirname "$0")
WORK=/root/.openclaw/workspace/agents/trader
# DEFAULT_STRATEGY="$WORK/STRATEGY_V4.2.md"
LABEL=${LABEL:-V4.2}
# STRATEGY_FILE=${1:-$DEFAULT_STRATEGY}

# Defaults
DEFAULTS=(
  "LABEL=$LABEL"
  "LIVE_MODE=1"
)

echo "Skill: live-forward-v4.2"
echo "Defaults to be used:"
for v in "${DEFAULTS[@]}"; do echo "  - $v"; done

read -p "Proceed with these settings? (y/N): " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
  echo "Aborting."; exit 1
fi

LOG_DIR="$WORK/logs/forwardtest_${LABEL}"
mkdir -p "$LOG_DIR"
PIDFILE_CHECK="$LOG_DIR/ad.pid"
# If a live PIDfile exists and the process is alive, do not start another instance
if [[ -f "$PIDFILE_CHECK" ]]; then
  pid_running=$(cat "$PIDFILE_CHECK" 2>/dev/null || echo "")
  if [[ -n "$pid_running" ]] && kill -0 "$pid_running" 2>/dev/null; then
    echo "Another instance (pid $pid_running) is already running for label=$LABEL. Aborting start to avoid duplicates." 
    exit 0
  else
    # stale pidfile
    rm -f "$PIDFILE_CHECK" || true
  fi
fi

echo "Starting live-forward loop (label=$LABEL). Logs -> $LOG_DIR"

while true; do
  TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
  LOG_STD="$LOG_DIR/ad_${TIMESTAMP}.out"
  PIDFILE="$LOG_DIR/ad.pid"

  export LABEL="$LABEL"
  
  echo "Launching trading-bot.ad.cjs at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # run with automatic rotation after 3 hours (10800s)
  nohup node $WORK/trading-bot.live.service.cjs --mode=paper --live=true --label=$LABEL > "$LOG_STD" 2>&1 &
  # nohup node $WORK/trading-bot.live.cjs --mode=paper --live=true --label=$LABEL > "$LOG_STD" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Started PID $(cat $PIDFILE), logging to $LOG_STD"

  # wait for process to exit
  wait $(cat $PIDFILE) || true
  echo "Process $(cat $PIDFILE) exited at $(date -u +%Y-%m-%dT%H:%M:%SZ). Rotating logs and restarting in 10s..."
  sleep 10
done
