#!/usr/bin/env node

/**
 * trading-bot.live.service.sa.cjs
 *
 * "SA" = Stabilized & Adaptive service version.
 *
 * Goals (per Victor):
 * - Maintain compatibility with current system (paths, config keys, journal format)
 * - Be more robust (no crash on missing data, stale lock recovery)
 * - Clear & well-commented logic
 * - Include volatility / regime detection (simple, stable heuristics)
 * - Optimize execution (less IO churn, aligned caching)
 *
 * Notes:
 * - This service runs in paper mode by default (journal-based simulation).
 * - It consumes Binance-style public endpoints (Binance/MEXC provide /api/v3 for spot).
 */

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
require('dotenv').config();

const {
  getClosedCloses,
  computeSmaPair,
  computeAtr,
  computeRealizedVol,
  detectMarketRegime
} = require('./market-regime.sa.cjs');

/* -----------------------------
 * PATHS & CONFIG
 * ----------------------------- */

const LABEL = process.env.LABEL || 'V4.2';
const BASE_DIR = path.join(__dirname, 'skills', `live-forward-${LABEL.toLowerCase()}`);
const CONFIG_PATH = path.join(BASE_DIR, 'config.json');
const LOCK_PATH = path.join(BASE_DIR, 'bot.lock');
const OPEN_POSITIONS_PATH = path.join(BASE_DIR, 'open_positions.json');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    console.error('readJsonFile err:', filePath, e.message);
    return fallback;
  }
}

function loadSkillConfig() {
  return readJsonFile(CONFIG_PATH, {});
}

const cfg = loadSkillConfig();
const EXCHANGE = process.env.EXCHANGE || cfg.EXCHANGE || 'binance';

/* -----------------------------
 * STATE
 * ----------------------------- */

let shutdownRequested = false;
let lockFd = null;
let snapshotTimer = null;
let snapshotDirty = false;
let lastSnapshotHash = '';
let lastTradeCandle = null;

const state = {
  openTradesById: new Map(),
  openTradeIdBySignalKey: new Map(),
  seenEventKeys: new Set(),
  recentSignalSeenAt: new Map(),
  lastOpenBySymbol: {},
  lastDecision: {},
  lastTickerCache: { ts: 0, price: null }
};

/* -----------------------------
 * EXCHANGE / API
 * ----------------------------- */

const API_BASES = {
  binance: 'https://api.binance.com',
  mexc: 'https://api.mexc.com'
};

function getApiBase() {
  return API_BASES[EXCHANGE] || API_BASES.binance;
}

const API_KEYS = {
  binance: {
    key: process.env.BINANCE_API_KEY || '',
    secret: process.env.BINANCE_API_SECRET || ''
  },
  mexc: {
    key: process.env.MEXC_API_KEY || '',
    secret: process.env.MEXC_API_SECRET || ''
  }
};

const ACTIVE_API_KEY = API_KEYS[EXCHANGE]?.key || '';
const ACTIVE_API_SECRET = API_KEYS[EXCHANGE]?.secret || '';

// Keys are not required for public data endpoints. We keep compatibility but avoid hard-exit.
const REQUIRE_KEYS = String(process.env.REQUIRE_KEYS || cfg.REQUIRE_KEYS || '0') === '1';
if (REQUIRE_KEYS && (!ACTIVE_API_KEY || !ACTIVE_API_SECRET)) {
  console.error(`API keys for ${EXCHANGE} not set in .env (REQUIRE_KEYS=1)`);
  process.exit(1);
}

console.log('Service:', 'trading-bot.live.service.sa.cjs');
console.log('Exchange:', EXCHANGE);
console.log('API Base:', getApiBase());

/* -----------------------------
 * HELPERS
 * ----------------------------- */

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isValidNumber(n) {
  return typeof n === 'number' && Number.isFinite(n);
}

function writeJsonFileAtomic(filePath, value) {
  try {
    ensureDir(path.dirname(filePath));
    const tmpPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(value, null, 2), 'utf8');
    fs.renameSync(tmpPath, filePath);
    return true;
  } catch (e) {
    console.error('writeJsonFileAtomic err:', filePath, e.message);
    return false;
  }
}

function appendJsonl(filePath, record) {
  try {
    ensureDir(path.dirname(filePath));
    fs.appendFileSync(filePath, JSON.stringify(record) + '\n', 'utf8');
    return true;
  } catch (e) {
    console.error('appendJsonl err:', filePath, e.message);
    return false;
  }
}

function loadJsonl(filePath) {
  try {
    if (!fs.existsSync(filePath)) return [];
    const raw = fs.readFileSync(filePath, 'utf8');
    if (!raw.trim()) return [];
    return raw
      .split(/\r?\n/)
      .filter(Boolean)
      .map(line => {
        try {
          return JSON.parse(line);
        } catch (e) {
          console.error('loadJsonl parse err:', e.message);
          return null;
        }
      })
      .filter(Boolean);
  } catch (e) {
    console.error('loadJsonl err:', filePath, e.message);
    return [];
  }
}

function getYmd(dateValue) {
  const dt = new Date(dateValue || Date.now());
  return String(dt.getUTCFullYear()) +
    String(dt.getUTCMonth() + 1).padStart(2, '0') +
    String(dt.getUTCDate()).padStart(2, '0');
}

function getJournalPath(dateValue) {
  return path.join(BASE_DIR, `trade_journal_${getYmd(dateValue)}.jsonl`);
}

function fmtDateUtc1(d) {
  // UTC+1 formatting (used in logs & iteration summaries)
  const dt = new Date(d.getTime() + 60 * 60 * 1000);
  const Y = dt.getUTCFullYear();
  const M = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const D = String(dt.getUTCDate()).padStart(2, '0');
  const h = String(dt.getUTCHours()).padStart(2, '0');
  const m = String(dt.getUTCMinutes()).padStart(2, '0');
  const s = String(dt.getUTCSeconds()).padStart(2, '0');
  return `${Y}-${M}-${D} ${h}:${m}:${s}`;
}

function formatHms(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const h = String(Math.floor(s / 3600)).padStart(2, '0');
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${h}:${m}:${ss}`;
}

/* -----------------------------
 * LOCK (robust)
 * ----------------------------- */

function isPidAlive(pid) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    // EPERM => exists but no permissions; treat as alive.
    return e && e.code === 'EPERM';
  }
}

function acquireLock() {
  ensureDir(BASE_DIR);

  // If lock exists, check staleness.
  if (fs.existsSync(LOCK_PATH)) {
    try {
      const raw = fs.readFileSync(LOCK_PATH, 'utf8');
      const info = JSON.parse(raw || '{}');
      const pid = Number(info.pid || 0);
      if (pid && isPidAlive(pid)) {
        console.error('lock exists, process seems alive:', { lock: LOCK_PATH, pid });
        return false;
      }
      // stale lock
      console.warn('stale lock detected, removing:', { lock: LOCK_PATH, pid });
      try { fs.unlinkSync(LOCK_PATH); } catch (_) {}
    } catch (e) {
      console.warn('lock exists but unreadable, removing:', { lock: LOCK_PATH, err: e.message });
      try { fs.unlinkSync(LOCK_PATH); } catch (_) {}
    }
  }

  try {
    lockFd = fs.openSync(LOCK_PATH, 'wx');
    fs.writeFileSync(lockFd, JSON.stringify({ pid: process.pid, started_at: new Date().toISOString() }), 'utf8');
    return true;
  } catch (e) {
    if (e && e.code === 'EEXIST') {
      console.error('lock exists, another instance may be running:', LOCK_PATH);
      return false;
    }
    console.error('acquireLock err:', e.message);
    return false;
  }
}

function releaseLock() {
  try {
    if (lockFd !== null) fs.closeSync(lockFd);
  } catch (_) {}
  lockFd = null;
  try {
    if (fs.existsSync(LOCK_PATH)) fs.unlinkSync(LOCK_PATH);
  } catch (e) {
    console.error('releaseLock err:', e.message);
  }
}

/* -----------------------------
 * SIGNAL KEYS (compat)
 * ----------------------------- */

function getSignalBucketMs(signalCandleTs, bucketMs) {
  if (!signalCandleTs) return 0;
  return Math.floor(Number(signalCandleTs) / bucketMs);
}

function getSignalKey(trade, candleBucketMs) {
  const symbol = trade.symbol || 'BTC';
  const side = trade.type || trade.side || 'LONG';
  const reasonTag = trade.reasonTag || trade.reason_tag || '';
  const bucket = getSignalBucketMs(trade.signalCandleTs || trade.signal_candle_ts, candleBucketMs);
  return `signal:${LABEL}:${symbol}:${side}:${bucket}:${reasonTag}`;
}

function getOpenEventKey(trade, candleBucketMs) {
  return `open:${getSignalKey(trade, candleBucketMs)}`;
}

function getCloseEventKey(trade, closeReason) {
  return `close:${LABEL}:${String(trade.id)}:${closeReason || 'UNKNOWN'}`;
}

/* -----------------------------
 * NORMALIZERS (compat: same journal shape; additive fields are OK)
 * ----------------------------- */

function normalizeOpenTrade(trade, candleBucketMs) {
  return {
    id: trade.id,
    label: LABEL,
    symbol: trade.symbol || 'BTCUSDT',
    open_time_iso: trade.openedAt || new Date().toISOString(),
    signal_candle_ts: trade.signalCandleTs || null,
    side: trade.type || 'LONG',
    qty: trade.size ?? 0,
    entry_price: trade.entryPrice ?? null,
    sl: trade.stopLoss ?? null,
    tp: trade.takeProfit ?? null,
    reason_tag: trade.reasonTag || '',
    reason_details: trade.reasonDetails || {},
    exposure_usd: trade.exposureUSD ?? null,
    signal_key: getSignalKey(trade, candleBucketMs)
  };
}

function normalizeCloseTrade(trade, closeReason, candleBucketMs) {
  const openedAt = trade.openedAt || '';
  const closedAt = trade.closedAt || new Date().toISOString();
  const durationS = openedAt
    ? Math.round((new Date(closedAt).getTime() - new Date(openedAt).getTime()) / 1000)
    : null;

  return {
    id: trade.id,
    label: LABEL,
    symbol: trade.symbol || 'BTCUSDT',
    open_time_iso: openedAt,
    close_time_iso: closedAt,
    duration_s: durationS,
    side: trade.type || 'LONG',
    qty: trade.size ?? 0,
    entry_price: trade.entryPrice ?? null,
    exit_price: trade.exitPrice ?? null,
    sl: trade.stopLoss ?? null,
    tp: trade.takeProfit ?? null,
    profit: trade.profit ?? null,
    profit_pct: trade.profit_pct ?? null,
    fee_usd_est: trade.feeUsdEst ?? null,
    profit_after_fees_est: (trade.profit != null && trade.feeUsdEst != null) ? (trade.profit - trade.feeUsdEst) : null,
    reason_tag: trade.reasonTag || '',
    close_reason: closeReason || '',
    reason_details: trade.reasonDetails || {},
    exposure_usd: trade.exposureUSD ?? null,
    signal_key: getSignalKey(trade, candleBucketMs)
  };
}

/* -----------------------------
 * JOURNAL STATE MANAGEMENT
 * ----------------------------- */

function applyJournalEvent(event) {
  if (!event || !event.event_key) return;
  state.seenEventKeys.add(event.event_key);

  if (event.type === 'OPEN') {
    const t = event.trade;
    const trade = {
      id: t.id,
      symbol: t.symbol || 'BTCUSDT',
      entryPrice: Number(t.entry_price ?? 0),
      stopLoss: t.sl != null ? Number(t.sl) : null,
      takeProfit: t.tp != null ? Number(t.tp) : null,
      openedAt: t.open_time_iso,
      signalCandleTs: t.signal_candle_ts,
      size: Number(t.qty ?? 0),
      type: t.side || 'LONG',
      reasonTag: t.reason_tag || '',
      reasonDetails: t.reason_details || {},
      exposureUSD: t.exposure_usd ?? null
    };

    const signalKey = t.signal_key;
    state.openTradesById.set(String(trade.id), trade);
    state.openTradeIdBySignalKey.set(signalKey, String(trade.id));
    state.recentSignalSeenAt.set(signalKey, Date.now());
    markSnapshotDirty();
  }

  if (event.type === 'CLOSE') {
    const t = event.trade;
    const id = String(t.id ?? '');
    const signalKey = t.signal_key || '';

    state.openTradesById.delete(id);
    if (signalKey) state.openTradeIdBySignalKey.delete(signalKey);
    if (signalKey) state.recentSignalSeenAt.set(signalKey, Date.now());

    markSnapshotDirty();
  }
}

function rebuildStateFromJournal() {
  state.openTradesById.clear();
  state.openTradeIdBySignalKey.clear();
  state.seenEventKeys.clear();
  state.recentSignalSeenAt.clear();

  const events = loadJsonl(getJournalPath(new Date().toISOString()));
  for (const ev of events) applyJournalEvent(ev);

  // ensure snapshot reflects journal state after restart
  flushOpenPositionsSnapshot(true);
}

function persistJournalEvent(event) {
  if (state.seenEventKeys.has(event.event_key)) return true;
  const ok = appendJsonl(getJournalPath(event.ts || new Date().toISOString()), event);
  if (!ok) return false;
  applyJournalEvent(event);
  return true;
}

function getOpenTradesArray() {
  return Array.from(state.openTradesById.values());
}

function markSnapshotDirty() {
  snapshotDirty = true;
}

function computeSnapshotHash(trades) {
  // Very small hash to avoid rewriting identical snapshot.
  // We hash stable fields; exclude volatile runtime fields.
  const s = JSON.stringify(trades.map(t => ({
    id: t.id,
    symbol: t.symbol,
    entryPrice: t.entryPrice,
    stopLoss: t.stopLoss,
    takeProfit: t.takeProfit,
    openedAt: t.openedAt,
    size: t.size,
    reasonTag: t.reasonTag
  })));
  return crypto.createHash('sha1').update(s).digest('hex');
}

function flushOpenPositionsSnapshot(force = false) {
  try {
    const trades = getOpenTradesArray();
    const h = computeSnapshotHash(trades);

    if (!force && !snapshotDirty && h === lastSnapshotHash) return true;

    const ok = writeJsonFileAtomic(
      OPEN_POSITIONS_PATH,
      trades.map(t => ({
        id: t.id,
        symbol: t.symbol,
        entryPrice: t.entryPrice,
        stopLoss: t.stopLoss,
        takeProfit: t.takeProfit,
        openedAt: t.openedAt,
        signalCandleTs: t.signalCandleTs,
        size: t.size,
        type: t.type,
        reasonTag: t.reasonTag,
        reasonDetails: t.reasonDetails,
        exposureUSD: t.exposureUSD
      }))
    );

    if (ok) {
      snapshotDirty = false;
      lastSnapshotHash = h;
    }

    return ok;
  } catch (e) {
    console.error('flushOpenPositionsSnapshot err:', e.message);
    return false;
  }
}

function startSnapshotTimer(intervalMs = 5000) {
  snapshotTimer = setInterval(() => {
    try {
      flushOpenPositionsSnapshot(false);
    } catch (e) {
      console.error('snapshot timer err:', e.message);
    }
  }, Math.max(1000, intervalMs));
}

function cleanupRecentSignals(maxAgeMs = 10 * 60 * 1000) {
  const now = Date.now();
  for (const [k, ts] of state.recentSignalSeenAt.entries()) {
    if ((now - ts) > maxAgeMs) state.recentSignalSeenAt.delete(k);
  }
}

function hasOpenSignal(signalKey) {
  return state.openTradeIdBySignalKey.has(signalKey);
}

function canOpenSignal(signalKey, cooldownMs) {
  const now = Date.now();
  const lastSeen = state.recentSignalSeenAt.get(signalKey) || 0;
  return (now - lastSeen) >= cooldownMs;
}

/* -----------------------------
 * HTTP (robust retry + backoff)
 * ----------------------------- */

function jitter(ms) {
  const j = 0.15; // ±15%
  return Math.round(ms * (1 - j + Math.random() * 2 * j));
}

async function httpGetWithRetry(url, opts = {}, retries = 4, baseDelayMs = 250, timeoutMs = 2500) {
  let lastErr = null;

  for (let i = 0; i < retries; i++) {
    try {
      const resp = await axios.get(url, {
        timeout: timeoutMs,
        validateStatus: () => true,
        ...opts
      });

      const status = resp?.status || 0;
      if (status >= 200 && status < 300) return resp;

      // Rate limit / abuse protection: backoff
      if (status === 429 || status === 418) {
        const delay = jitter(baseDelayMs * Math.pow(2, i));
        console.warn('HTTP rate limited', { status, url: url.split('?')[0], delay });
        await sleep(delay);
        continue;
      }

      // Transient server errors
      if (status >= 500 && status < 600) {
        const delay = jitter(baseDelayMs * Math.pow(2, i));
        console.warn('HTTP server error', { status, url: url.split('?')[0], delay });
        await sleep(delay);
        continue;
      }

      // Other non-2xx: do not retry too aggressively
      lastErr = new Error(`HTTP ${status}`);
      console.error('HTTP non-2xx', { status, url: url.split('?')[0], data: resp?.data });
      break;
    } catch (e) {
      lastErr = e;
      const delay = jitter(baseDelayMs * Math.pow(2, i));
      console.warn('HTTP error', { attempt: i + 1, url: url.split('?')[0], err: e.message, delay });
      if (i < retries - 1) await sleep(delay);
    }
  }

  throw lastErr || new Error('httpGetWithRetry failed');
}

async function getTicker(symbol, timeoutMs) {
  const base = getApiBase();
  const url = `${base}/api/v3/ticker/price?symbol=${symbol}`;

  try {
    const headers = ACTIVE_API_KEY ? { 'X-MBX-APIKEY': ACTIVE_API_KEY } : undefined;
    const r = await httpGetWithRetry(url, headers ? { headers } : {}, 4, 250, timeoutMs);
    const px = Number(r?.data?.price);
    return Number.isFinite(px) ? px : null;
  } catch (e) {
    console.error('getTicker failed', EXCHANGE, e.message);
    return null;
  }
}

async function getTickerCached(symbol, timeoutMs, maxAgeMs) {
  const now = Date.now();
  if (state.lastTickerCache.price !== null && (now - state.lastTickerCache.ts) <= maxAgeMs) {
    return state.lastTickerCache.price;
  }
  const price = await getTicker(symbol, timeoutMs);
  if (price !== null) state.lastTickerCache = { ts: now, price };
  return price;
}

async function getRecentKlines(symbol, limit, interval, timeoutMs) {
  const base = getApiBase();
  const url = `${base}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;

  try {
    const headers = ACTIVE_API_KEY ? { 'X-MBX-APIKEY': ACTIVE_API_KEY } : undefined;
    const r = await httpGetWithRetry(url, headers ? { headers } : {}, 4, 250, timeoutMs);
    if (!Array.isArray(r?.data)) return null;
    return r.data;
  } catch (e) {
    console.error('getRecentKlines failed', EXCHANGE, e.message);
    return null;
  }
}

/* -----------------------------
 * OPTIONAL: Signed requests (kept for compatibility; not used by default)
 * ----------------------------- */

async function signedBinanceRequest(method, endpoint, params) {
  const base = getApiBase();
  const qs = new URLSearchParams(params).toString();
  const signature = crypto.createHmac('sha256', ACTIVE_API_SECRET).update(qs).digest('hex');
  const url = `${base}${endpoint}?${qs}&signature=${signature}`;
  const headers = { 'X-MBX-APIKEY': ACTIVE_API_KEY };
  return axios({ method, url, headers, timeout: 5000 });
}

async function PlaceLiveOpenOrderOco(trade, symbol) {
  // This is exchange-specific (Binance-style OCO). Disabled unless explicitly enabled.
  const ENABLE_LIVE = String(process.env.ENABLE_LIVE || cfg.ENABLE_LIVE || '0') === '1';
  if (!ENABLE_LIVE) return;

  if (!ACTIVE_API_KEY || !ACTIVE_API_SECRET) {
    console.error('ENABLE_LIVE=1 but missing API keys; skipping live order.');
    return;
  }

  if (!isValidNumber(trade.entryPrice) || !isValidNumber(trade.stopLoss) || !isValidNumber(trade.takeProfit)) {
    console.error('Invalid trade values for live OCO', { id: trade.id });
    return;
  }

  try {
    const qty = trade.size;
    const res = await signedBinanceRequest('POST', '/api/v3/order/oco', {
      symbol,
      side: 'SELL',
      quantity: String(qty),
      price: trade.takeProfit.toFixed(2),
      stopPrice: trade.stopLoss.toFixed(2),
      stopLimitPrice: trade.stopLoss.toFixed(2),
      stopLimitTimeInForce: 'GTC'
    });
    console.log('LIVE OCO ORDER PLACED:', JSON.stringify(res.data || res));
  } catch (e) {
    console.error('LIVE OCO ORDER ERROR:', e.message, e.response?.data || '');
  }
}

/* -----------------------------
 * TRADING LOGIC (paper)
 * ----------------------------- */

function buildCloseReason(trade, market) {
  if (trade.stopLoss && market <= trade.stopLoss) return 'SL';
  if (trade.takeProfit && market >= trade.takeProfit) return 'TP';
  return 'OTHER';
}

async function openTrade(trade, symbol, candleBucketMs, signalCooldownMs) {
  const signalKey = getSignalKey(trade, candleBucketMs);

  if (hasOpenSignal(signalKey)) {
    console.log('skip duplicated signal already open', signalKey);
    return false;
  }
  if (!canOpenSignal(signalKey, signalCooldownMs)) {
    console.log('skip repeated signal in cooldown', signalKey);
    return false;
  }

  const event = {
    ts: new Date().toISOString(),
    type: 'OPEN',
    event_key: getOpenEventKey(trade, candleBucketMs),
    trade: normalizeOpenTrade(trade, candleBucketMs)
  };

  const ok = persistJournalEvent(event);
  if (!ok) return false;

  console.log(
    'OPEN (paper):', trade.openedAt,
    'id=', trade.id,
    'regime=', trade.regime,
    'signalCandleTs=', trade.signalCandleTs,
    'signalKey=', signalKey,
    'entry=', trade.entryPrice,
    'SL=', trade.stopLoss,
    'TP=', trade.takeProfit,
    'REASON=', trade.reasonTag
  );

  // Optional live order (disabled by default)
  // await PlaceLiveOpenOrderOco(trade, symbol);

  return true;
}

async function closeTrade(trade, market, closeReason, symbol, candleBucketMs, feeRate) {
  if (!isValidNumber(market) || !isValidNumber(trade.entryPrice)) {
    console.error('closeTrade invalid values', { market, entryPrice: trade.entryPrice, id: trade.id });
    return false;
  }

  const qty = Number(trade.size || 0);
  trade.exitPrice = market;
  trade.profit = (market - trade.entryPrice) * qty;
  trade.profit_pct = trade.entryPrice > 0 ? (market - trade.entryPrice) / trade.entryPrice : null;
  trade.closedAt = new Date().toISOString();

  // Fee estimate (round-turn): (entry notional + exit notional) * feeRate
  const feeUsdEst = (trade.entryPrice * qty + market * qty) * (feeRate || 0);
  trade.feeUsdEst = feeUsdEst;

  const reason = closeReason || buildCloseReason(trade, market);
  const event = {
    ts: new Date().toISOString(),
    type: 'CLOSE',
    event_key: getCloseEventKey(trade, reason),
    trade: normalizeCloseTrade(trade, reason, candleBucketMs)
  };

  const ok = persistJournalEvent(event);
  if (!ok) return false;

  console.log(
    'CLOSE:',
    reason,
    'id=', trade.id,
    'signalKey=', getSignalKey(trade, candleBucketMs),
    'open=', trade.openedAt,
    'close=', trade.closedAt,
    'entry=', trade.entryPrice,
    'exit=', trade.exitPrice,
    'profit=', trade.profit,
    'profit%=', trade.profit_pct,
    'feeUsdEst=', feeUsdEst,
    'profit_after_fees_est=', trade.profit - feeUsdEst
  );

  return true;
}

/* -----------------------------
 * GRACEFUL SHUTDOWN
 * ----------------------------- */

async function gracefulShutdown(signal) {
  if (shutdownRequested) return;
  shutdownRequested = true;
  console.log('shutdown signal:', signal);

  try {
    if (snapshotTimer) clearInterval(snapshotTimer);
    flushOpenPositionsSnapshot(true);
  } catch (e) {
    console.error('shutdown flush err:', e.message);
  }

  releaseLock();
  process.exit(0);
}

/* -----------------------------
 * MAIN LOOP
 * ----------------------------- */

(async () => {
  console.log('Starting SA service (paper mode by default)');
  ensureDir(BASE_DIR);

  if (!acquireLock()) process.exit(1);

  // Reload config at startup (we still keep cfg loaded globally for early reads)
  const cfgLive = loadSkillConfig();

  // --- Primary settings (compatible with current config keys) ---
  const symbol = process.env.SYMBOL || cfgLive.SYMBOL || 'BTCUSDT';
  const monitorInterval = parseInt(process.env.MONITOR_INTERVAL_MS || cfgLive.MONITOR_INTERVAL_MS || 1000, 10);
  const maxPositions = parseInt(process.env.MAX_POSITIONS || cfgLive.MAX_POSITIONS || 2, 10);
  const minHoldS = parseInt(process.env.MIN_HOLD_SECONDS || cfgLive.MIN_HOLD_SECONDS || '60', 10);
  const timeStopMinutes = parseInt(process.env.TIME_STOP_MINUTES || cfgLive.TIME_STOP_MINUTES || 10, 10);
  const HTTP_TIMEOUT_MS = parseInt(process.env.HTTP_TIMEOUT_MS || cfgLive.HTTP_TIMEOUT_MS || 2500, 10);
  const candleBucketMs = parseInt(process.env.SIGNAL_BUCKET_MS || cfgLive.SIGNAL_BUCKET_MS || 60000, 10);
  const signalCooldownMs = parseInt(process.env.SIGNAL_COOLDOWN_MS || cfgLive.SIGNAL_COOLDOWN_MS || 180000, 10);

  const explosiveCandlePct = parseFloat(process.env.EXPLOSIVE_CANDLE_PCT || cfgLive.EXPLOSIVE_CANDLE_PCT || 0.003);
  const tradeUSD = parseFloat(process.env.TRADE_USD || cfgLive.TRADE_USD || 100.0);
  const minCandleBody = parseFloat(process.env.MIN_BODY_CANDLE || cfgLive.MIN_BODY_CANDLE || 0.5);

  const minSMASlope = parseFloat(process.env.MIN_SMA_SLOPE || cfgLive.MIN_SMA_SLOPE || 2.0);
  const SMA_WINDOW = parseInt(process.env.SMA_WINDOW || cfgLive.SMA_WINDOW || 60, 10);
  const limit = SMA_WINDOW + 2;

  const k_tp = parseFloat(process.env.K_TP || cfgLive.K_TP || '2.0');
  const k_sl = parseFloat(process.env.K_SL || cfgLive.K_SL || '1.2');

  const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || cfgLive.MIN_MOMENTUM_PCT || 0.0005);
  const MAX_MOMENTUM_PCT = parseFloat(process.env.MAX_MOMENTUM_PCT || cfgLive.MAX_MOMENTUM_PCT || 0.0012);

  const MOM_REDUCTION_PCT = parseFloat(process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || cfgLive.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || 0.0);
  const GREEN_KLINES = parseInt(process.env.GREEN_KLINES_FOR_REDUCTION || cfgLive.GREEN_KLINES_FOR_REDUCTION || 0, 10);

  const smaTol = parseFloat(process.env.SMA_TOLERANCE || cfgLive.SMA_TOLERANCE || 0.001);
  const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || cfgLive.SYMBOL_COOLDOWN_SECONDS || 0, 10);
  const feeRate = parseFloat(process.env.FEE_RATE || cfgLive.FEE_RATE || 0.0005);
  const ATR_WINDOW = parseInt(process.env.ATR_WINDOW || cfgLive.ATR_WINDOW || 14, 10);

  // snapshot interval: lower IO churn
  const SNAPSHOT_INTERVAL_MS = parseInt(process.env.SNAPSHOT_INTERVAL_MS || cfgLive.SNAPSHOT_INTERVAL_MS || 5000, 10);
  startSnapshotTimer(SNAPSHOT_INTERVAL_MS);

  // Align ticker cache to monitor interval (avoid pointless 300ms cache)
  const TICKER_CACHE_MS = parseInt(process.env.TICKER_CACHE_MS || cfgLive.TICKER_CACHE_MS || Math.max(250, Math.min(2000, monitorInterval)), 10);

  rebuildStateFromJournal();

  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

  while (!shutdownRequested) {
    const tsStartIter = fmtDateUtc1(new Date());

    try {
      cleanupRecentSignals();

      const klines = await getRecentKlines(symbol, limit, '1m', HTTP_TIMEOUT_MS);
      if (!Array.isArray(klines) || klines.length < 4) {
        console.warn('No/insufficient klines; sleeping and retrying...', { got: Array.isArray(klines) ? klines.length : null });
        await sleep(500);
        continue;
      }

      // Last CLOSED candle (Binance klines): index -2
      const lastClosed = klines[klines.length - 2];
      const candle = {
        ts: Number(lastClosed[0]),
        open: Number(lastClosed[1]),
        high: Number(lastClosed[2]),
        low: Number(lastClosed[3]),
        close: Number(lastClosed[4]),
        volume: Number(lastClosed[5])
      };

      if (![candle.ts, candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)) {
        console.warn('Invalid candle numbers; skipping iteration', candle);
        await sleep(250);
        continue;
      }

      const closes = getClosedCloses(klines);
      if (!Array.isArray(closes) || closes.length < 3) {
        console.warn('Invalid closes extracted; skipping iteration');
        await sleep(250);
        continue;
      }

      const last = closes[closes.length - 1];
      const prev = closes[closes.length - 2] || last;
      const momentum_pct = prev ? (last - prev) / prev : 0;

      // --- Filters (robust, conservative) ---
      const body = Math.abs(candle.close - candle.open);
      const range = candle.high - candle.low;
      const bodyRatio = range > 0 ? body / range : 0;
      const weakCandleBody = bodyRatio < minCandleBody;

      // Weak open: compare candle.open vs previous TWO closes (excluding current candle itself)
      let weakOpen = false;
      if (klines.length >= 6) {
        const prevClose1 = Number(klines[klines.length - 3][4]);
        const prevClose2 = Number(klines[klines.length - 4][4]);
        if (Number.isFinite(prevClose1) && Number.isFinite(prevClose2)) {
          weakOpen = candle.open < prevClose1 && candle.open < prevClose2;
        }
      }

      // Explosive candle detection: last 2 closed candles ranges
      let candleExplosive = false;
      try {
        const kA = klines[klines.length - 2];
        const kB = klines[klines.length - 3];
        const ranges = [kA, kB].map(k => {
          const hi = Number(k[2]);
          const lo = Number(k[3]);
          const cl = Number(k[4]);
          if (!Number.isFinite(hi) || !Number.isFinite(lo) || !Number.isFinite(cl) || cl <= 0) return 0;
          return (hi - lo) / cl;
        });
        candleExplosive = ranges.some(r => r > explosiveCandlePct);
      } catch (_) {
        candleExplosive = false;
      }

      // --- Indicators (SMA, ATR, regime) ---
      const { sma, smaPrev } = computeSmaPair(closes, SMA_WINDOW);
      const smaSlope = (sma != null && smaPrev != null) ? (sma - smaPrev) : 0;
      const priceNearSMA = (sma != null) ? (candle.close >= sma * (1 - smaTol)) : true;
      const priceAboveSMA = (sma != null) ? (candle.close > sma) : true;

      const { atrPct } = computeAtr(klines, ATR_WINDOW);
      const realizedVol = computeRealizedVol(closes, Math.min(60, Math.max(20, Math.floor(SMA_WINDOW / 2))));

      const regimeInfo = detectMarketRegime({
        atrPct,
        realizedVol,
        smaSlope,
        lastClose: candle.close,
        priceAboveSma: priceAboveSMA
      });

      const regime = regimeInfo.volRegime; // keep compatibility label (LOW_VOL|MID_VOL|HIGH_VOL)

      // Dynamic thresholds (bounded multipliers, see market-regime.sa.cjs)
      const dynamicMinMomentum = BASE_MIN_MOM * regimeInfo.kMinMomentum;
      const dynamicMaxMomentum = MAX_MOMENTUM_PCT; // keep as cap (do not inflate too much)
      const dynamicMinSlope = minSMASlope * regimeInfo.kMinSlope;

      const trendUp = smaSlope > dynamicMinSlope;

      // Green run detection (same as current logic)
      let green_run = 0;
      if (GREEN_KLINES > 0 && Array.isArray(klines)) {
        for (let j = klines.length - 2; j > 0 && green_run < GREEN_KLINES; j--) {
          const cur = Number(klines[j][4]);
          const op = Number(klines[j][1]);
          if (Number.isFinite(cur) && Number.isFinite(op) && cur > op) green_run++;
          else break;
        }
      }

      let effectiveMinMom = dynamicMinMomentum;
      if (green_run >= GREEN_KLINES && GREEN_KLINES > 0 && MOM_REDUCTION_PCT > 0) {
        effectiveMinMom = dynamicMinMomentum * (1 - MOM_REDUCTION_PCT);
      }

      const momentumOk = momentum_pct >= effectiveMinMom && momentum_pct <= dynamicMaxMomentum;

      // Final decision
      const shouldEnter = momentumOk &&
        trendUp &&
        priceNearSMA &&
        priceAboveSMA &&
        !weakCandleBody &&
        !weakOpen &&
        !candleExplosive;

      state.lastDecision = {
        momentum_pct: Number(momentum_pct.toFixed(6)),
        sma: sma != null ? Number(sma.toFixed(2)) : null,
        smaSlope: Number(smaSlope.toFixed(2)),
        atr_pct: Number((atrPct || 0).toFixed(6)),
        realized_vol: Number((realizedVol || 0).toFixed(6)),
        regime,
        microRegime: regimeInfo.microRegime,
        slopeNorm: regimeInfo.slopeNorm,
        priceNearSMA: !!priceNearSMA,
        priceAboveSMA: !!priceAboveSMA,
        trendUp: !!trendUp,
        weakCandleBody: !!weakCandleBody,
        weakOpen: !!weakOpen,
        candleExplosive: !!candleExplosive,
        shouldEnter: !!shouldEnter,
        effective_min_momentum: Number(effectiveMinMom.toFixed(6)),
        green_run
      };

      // Cooldown controls
      const nowTs = Date.now();
      const lastOpenTs = state.lastOpenBySymbol[symbol] || 0;
      const withinCooldown = (cooldown_s > 0) && ((nowTs - lastOpenTs) < cooldown_s * 1000);
      const candleMinute = Math.floor(candle.ts / 60000);
      const alreadyOpenedThisCandle = lastTradeCandle !== null && lastTradeCandle === candleMinute;

      // --- Entry ---
      if (shouldEnter && state.openTradesById.size < maxPositions && !withinCooldown && !alreadyOpenedThisCandle) {
        const entryPrice = candle.close;
        const effectiveATR = Number.isFinite(atrPct) ? atrPct : 0;

        // TP/SL based on ATR% (compatible)
        const stopLoss = entryPrice * (1 - k_sl * effectiveATR);
        const takeProfit = entryPrice * (1 + k_tp * effectiveATR);

        const stopDistanceUSD = entryPrice - stopLoss;

        // Current in-progress candle low as sanity check
        const current = klines[klines.length - 1];
        const currentLow = Number(current?.[3]);

        // Position size (USD exposure)
        const qty = Number((tradeUSD / entryPrice).toFixed(8));

        // Profitability with fee estimate (entry + exit at TP)
        const grossProfitAtTp = qty * (takeProfit - entryPrice);
        const feeEstAtTp = (entryPrice * qty + takeProfit * qty) * feeRate;
        const netProfitAtTp = grossProfitAtTp - feeEstAtTp;

        if (!isValidNumber(entryPrice) || !isValidNumber(stopLoss) || !isValidNumber(takeProfit) || !isValidNumber(qty)) {
          console.error('invalid trade values', { entryPrice, stopLoss, takeProfit, qty });
        } else if (stopDistanceUSD <= 0) {
          console.error('Trade skipped: Invalid stop distance', { entryPrice, stopLoss });
        } else if (Number.isFinite(currentLow) && currentLow <= stopLoss) {
          console.log('Trade skipped: SL inside candle', { entryPrice, stopLoss, currentLow, candleLow: candle.low });
        } else if (netProfitAtTp <= 0) {
          console.log('Trade skipped: Not profitable after fees (est)', {
            entryPrice,
            takeProfit,
            grossProfitAtTp,
            feeEstAtTp,
            netProfitAtTp
          });
        } else {
          const reasonDet = {
            momentum_pct: Number(momentum_pct.toFixed(6)),
            sma: sma != null ? Number(sma.toFixed(2)) : null,
            sma_slope: Number(smaSlope.toFixed(2)),
            atr_pct: Number((atrPct || 0).toFixed(6)),
            realized_vol: Number((realizedVol || 0).toFixed(6)),
            regime,
            micro_regime: regimeInfo.microRegime,
            slope_norm: regimeInfo.slopeNorm,
            base_min_momentum: BASE_MIN_MOM,
            effective_minimum: Number(effectiveMinMom.toFixed(6)),
            green_run_len: green_run,
            reduction_pct: MOM_REDUCTION_PCT
          };

          const reasonTag = (green_run >= GREEN_KLINES && MOM_REDUCTION_PCT > 0)
            ? 'momentum_with_green_run'
            : 'momentum_standard';

          const trade = {
            id: Date.now(),
            symbol,
            entryPrice,
            stopLoss,
            takeProfit,
            openedAt: new Date().toISOString(),
            signalCandleTs: candle.ts,
            size: qty,
            exposureUSD: Number((entryPrice * qty).toFixed(2)),
            type: 'LONG',
            regime,
            reasonTag,
            reasonDetails: reasonDet
          };

          const didOpen = await openTrade(trade, symbol, candleBucketMs, signalCooldownMs);
          if (didOpen) {
            lastTradeCandle = candleMinute;
            state.lastOpenBySymbol[symbol] = Date.now();
          }
        }
      }

      // --- Monitor open trades for ~1 minute ---
      const monitorStart = Date.now();
      while (!shutdownRequested && (Date.now() - monitorStart < 60 * 1000)) {
        const loopStart = Date.now();
        try {
          const market = await getTickerCached(symbol, HTTP_TIMEOUT_MS, TICKER_CACHE_MS);
          if (market !== null) {
            for (const tr of getOpenTradesArray()) {
              const age = (Date.now() - new Date(tr.openedAt).getTime()) / 1000;
              if (age < minHoldS) continue;

              if (age > timeStopMinutes * 60) {
                await closeTrade(tr, market, 'time_stop', symbol, candleBucketMs, feeRate);
              } else if (tr.stopLoss && market <= tr.stopLoss) {
                await closeTrade(tr, market, 'SL', symbol, candleBucketMs, feeRate);
              } else if (tr.takeProfit && market >= tr.takeProfit) {
                await closeTrade(tr, market, 'TP', symbol, candleBucketMs, feeRate);
              }
            }
          }
        } catch (e) {
          console.error('monitor err:', e.message);
        }

        const elapsed = Date.now() - loopStart;
        const sleepMs = Math.max(0, monitorInterval - elapsed);
        await sleep(sleepMs);
      }

      // --- Iteration summary (keep compatibility with current log parser) ---
      console.log(
        'ITER_SUMMARY:',
        'Started at: ' + tsStartIter,
        'with candle: ' + new Date(candle.ts + 3600000).toISOString().slice(11, 16),
        'volume last candle=' + candle.volume.toFixed(2),
        'momentum=' + (state.lastDecision.momentum_pct || 0),
        'sma=' + (state.lastDecision.sma || 'null'),
        'smaSlope=' + (state.lastDecision.smaSlope || 0),
        'priceNearSMA=' + (state.lastDecision.priceNearSMA ? 1 : 0),
        'priceAboveSMA=' + (state.lastDecision.priceAboveSMA ? 1 : 0),
        'trendUp=' + (state.lastDecision.trendUp ? 1 : 0),
        'shouldEnter=' + (state.lastDecision.shouldEnter ? 1 : 0),
        'effective_min_momentum=' + (state.lastDecision.effective_min_momentum || 0),
        'green_run=' + (state.lastDecision.green_run || 0),
        'regime=' + (state.lastDecision.regime || 'UNKNOWN'),
        'micro=' + (state.lastDecision.microRegime || 'NA'),
        'openTrades=' + state.openTradesById.size
      );
      console.log('---------------------------------------------\n');

      for (const ot of getOpenTradesArray()) {
        console.log('OPEN_TRADE:', JSON.stringify(ot));
      }

    } catch (e) {
      // Do not crash the service; log and continue.
      console.error('ITERATION_FATAL_ERR (caught):', e.message);
      await sleep(500);
    }
  }
})();
