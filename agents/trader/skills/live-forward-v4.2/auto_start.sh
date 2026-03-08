#!/bin/bash
# Auto-start helper for live-forward-v4.2
# Use as: /root/.openclaw/workspace/agents/trader/skills/live-forward-v4.2/auto_start.sh

WORK=/root/.openclaw/workspace/agents/trader/skills/live-forward-v4.2
cd "$WORK"

# Start launcher (auto-confirm)
nohup bash -c 'echo y | ./launch.sh' >/root/.openclaw/workspace/agents/trader/skills/live-forward-v4.2/launch_nohup.out 2>&1 &

echo "started auto-launch (check launch_nohup.out)"
