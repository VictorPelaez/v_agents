#!/usr/bin/env bash
# Lanzar trading-bot.live.service.sa.cjs (SA: Stabilized & Adaptive)
# + Arrancar dashboard/app a continuación (sin interrumpir el bot).

set -euo pipefail

WORK=/root/.openclaw/workspace/agents/trader
LABEL=${LABEL:-V4.2}

LOG_DIR="$WORK/logs/forwardtest_${LABEL}"
mkdir -p "$LOG_DIR"

# ---- BOT (SA) ----
# Usamos el MISMO PIDFILE (ad.pid) para evitar que corran 2 servicios a la vez
BOT_PIDFILE="$LOG_DIR/ad.pid"

start_bot_sa() {
  if [[ -f "$BOT_PIDFILE" ]]; then
    local pid
    pid=$(cat "$BOT_PIDFILE" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Bot ya está corriendo (PID $pid) en $LOG_DIR"
      return 0
    else
      rm -f "$BOT_PIDFILE" || true
    fi
  fi

  local ts log
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$LOG_DIR/ad_sa_${ts}.out"

  # Nota: el servicio SA lee SYMBOL/EXCHANGE/LABEL desde env o config.json.
  nohup env \
    LABEL="$LABEL" \
    EXCHANGE="${EXCHANGE:-}" \
    SYMBOL="${SYMBOL:-}" \
    ENABLE_LIVE="${ENABLE_LIVE:-0}" \
    REQUIRE_KEYS="${REQUIRE_KEYS:-0}" \
    node "$WORK/trading-bot.live.service.sa.cjs" \
    > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$BOT_PIDFILE"

  echo "Bot SA iniciado con PID $pid, log -> $log"
}

# ---- DASHBOARD ----
APP_DIR="$WORK/skills/live-forward-v4.2/app"
DASH_PIDFILE="$LOG_DIR/dashboard.pid"

start_dashboard_fast_bg() {
  # Evitar duplicados dashboard
  if [[ -f "$DASH_PIDFILE" ]]; then
    local pid
    pid=$(cat "$DASH_PIDFILE" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Dashboard ya está corriendo (PID $pid)"
      return 0
    else
      rm -f "$DASH_PIDFILE" || true
    fi
  fi

  local port host ts log
  port=${PORT:-18790}
  host=${HOST:-127.0.0.1}
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$LOG_DIR/dashboard_${ts}.out"

  # Modo rápido:
  # - Si faltan node_modules o build, los prepara.
  # - Luego arranca backend en background.
  if [[ ! -d "$APP_DIR/backend/node_modules" ]]; then
    echo "[dashboard] npm install (backend)..."
    (cd "$APP_DIR/backend" && npm install)
  fi

  if [[ ! -d "$APP_DIR/frontend/node_modules" ]]; then
    echo "[dashboard] npm install (frontend)..."
    (cd "$APP_DIR/frontend" && npm install)
  fi

  if [[ ! -d "$APP_DIR/frontend/dist" ]]; then
    echo "[dashboard] build frontend..."
    (cd "$APP_DIR/frontend" && npm run build)
  fi

  echo "[dashboard] starting backend on http://${host}:${port} (bg)"
  nohup env PORT="$port" HOST="$host" \
    node "$APP_DIR/backend/src/server.js" \
    > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$DASH_PIDFILE"

  echo "Dashboard iniciado con PID $pid, log -> $log"
}

start_bot_sa

# Por defecto NO arrancar dashboard para ahorrar recursos.
# Para levantarlo explícitamente:
#   DASHBOARD=1 ./launch_live_service_sa.sh
if [[ "${DASHBOARD:-0}" == "1" ]]; then
  start_dashboard_fast_bg
  echo "OK: bot + dashboard lanzados (sin bloquear)."
else
  echo "OK: bot lanzado. Dashboard skipped (set DASHBOARD=1 to enable)."
fi
