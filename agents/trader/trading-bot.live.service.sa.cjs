#!/usr/bin/env node

/**
 * trading-bot.live.service.sa.cjs
 *
 * SA = Stabilized & Adaptive.
 *
 * Additions requested by Victor:
 * 1) Trade-rate limit: configurable max trades per hour (MAX_TRADES_PER_HOUR in config.json).
 * 2) Multi-crypto support: configurable SYMBOLS[] in config.json (backwards compatible with SYMBOL).
 * 3) Verbose mode ON by default: show symbol being analyzed, loop iteration, filter pass/fail,
 *    and when signals are generated/discarded.
 *
 * Hard constraints:
 * - Respect existing config.json (do not break; new params optional)
 * - Maintain inputs/outputs & journal JSONL format/rotation trade_journal_YYYYMMDD.jsonl
 * - Do not break current log patterns (keep ITER_SUMMARY / OPEN_TRADE style lines)
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
let iterCount = 0;

const state = {
  // journal/state
  openTradesById: new Map(),            // id -> trade
  openTradesBySymbol: new Map(),        // symbol -> Map(id -> trade)
  openTradeIdBySignalKey: new Map(),    // signalKey -> id
  seenEventKeys: new Set(),
  recentSignalSeenAt: new Map(),

  // runtime per-symbol
  symbolRuntime: new Map(),             // symbol -> { lastAnalyzedClosedTs, lastTradeCandleMinute, lastOpenTs, lastKlinesPollMs }
  decisionBySymbol: new Map(),
  tickerCacheBySymbol: new Map(),

  // trade-rate limiting
  recentOpenTimes: [],                  // timestamps (ms) for rolling 1h window
  openCountsByDay: new Map(),           // yyyymmdd (UTC) -> count of OPEN events

  // misc
  lastOpenBySymbol: {},

  // log de OPEN_TRADE (anti-spam)
  lastOpenTradesLogHash: '',
  lastOpenTradesLogCount: 0
};

function getSymbolRuntime(symbol) {
  if (!state.symbolRuntime.has(symbol)) {
    state.symbolRuntime.set(symbol, {
      lastAnalyzedClosedTs: null,
      lastTradeCandleMinute: null,
      lastOpenTs: 0,
      lastKlinesPollMs: 0,

      // ATR adaptive (per-symbol)
      atrHistory: [],
      lastAtrCandleTs: null
    });
  }
  return state.symbolRuntime.get(symbol);
}

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

// Keys are not required for public endpoints. We keep compatibility but avoid hard-exit.
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

function percentile(sortedOrUnsorted, q) {
  try {
    if (!Array.isArray(sortedOrUnsorted) || sortedOrUnsorted.length === 0) return 0;
    const p = Math.max(0, Math.min(1, Number(q)));
    const a = sortedOrUnsorted.slice().filter(Number.isFinite).sort((x, y) => x - y);
    if (a.length === 0) return 0;

    const idx = (a.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return a[lo];
    const w = idx - lo;
    return a[lo] * (1 - w) + a[hi] * w;
  } catch (_) {
    return 0;
  }
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

function nowIso() {
  return new Date().toISOString();
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
    return e && e.code === 'EPERM';
  }
}

function acquireLock() {
  ensureDir(BASE_DIR);

  if (fs.existsSync(LOCK_PATH)) {
    try {
      const raw = fs.readFileSync(LOCK_PATH, 'utf8');
      const info = JSON.parse(raw || '{}');
      const pid = Number(info.pid || 0);
      if (pid && isPidAlive(pid)) {
        console.error('lock exists, process seems alive:', { lock: LOCK_PATH, pid });
        return false;
      }
      console.warn('stale lock detected, removing:', { lock: LOCK_PATH, pid });
      try { fs.unlinkSync(LOCK_PATH); } catch (_) {}
    } catch (e) {
      console.warn('lock exists but unreadable, removing:', { lock: LOCK_PATH, err: e.message });
      try { fs.unlinkSync(LOCK_PATH); } catch (_) {}
    }
  }

  try {
    lockFd = fs.openSync(LOCK_PATH, 'wx');
    fs.writeFileSync(lockFd, JSON.stringify({ pid: process.pid, started_at: nowIso() }), 'utf8');
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
  const symbol = trade.symbol || 'BTCUSDT';
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
 * NORMALIZERS (compat: additive fields ok)
 * ----------------------------- */

function normalizeOpenTrade(trade, candleBucketMs) {
  return {
    id: trade.id,
    label: LABEL,
    symbol: trade.symbol || 'BTCUSDT',
    open_time_iso: trade.openedAt || nowIso(),
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
  const closedAt = trade.closedAt || nowIso();
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
 * TRADE-RATE LIMITING
 * ----------------------------- */

function pruneRecentOpens(nowMs, windowMs) {
  const w = Math.max(1000, windowMs || 3600000);
  state.recentOpenTimes = state.recentOpenTimes.filter(ts => (nowMs - ts) <= w);
}

function noteOpenTimestamp(ms) {
  if (!Number.isFinite(ms)) return;
  const nowMs = Date.now();
  state.recentOpenTimes.push(ms);
  pruneRecentOpens(nowMs, 3600000);
}

function countOpensLastHour() {
  const nowMs = Date.now();
  pruneRecentOpens(nowMs, 3600000);
  return state.recentOpenTimes.length;
}

function countOpensTodayUtc() {
  const ymd = getYmd(Date.now());
  return state.openCountsByDay.get(ymd) || 0;
}

/* -----------------------------
 * JOURNAL STATE MANAGEMENT
 * ----------------------------- */

function ensureOpenTradesSymbolMap(symbol) {
  if (!state.openTradesBySymbol.has(symbol)) state.openTradesBySymbol.set(symbol, new Map());
  return state.openTradesBySymbol.get(symbol);
}

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

    const symbol = trade.symbol;
    const signalKey = t.signal_key;

    state.openTradesById.set(String(trade.id), trade);
    ensureOpenTradesSymbolMap(symbol).set(String(trade.id), trade);

    state.openTradeIdBySignalKey.set(signalKey, String(trade.id));
    state.recentSignalSeenAt.set(signalKey, Date.now());

    // rebuild trade-rate limiting window from journal
    const ms = Date.parse(trade.openedAt);
    if (Number.isFinite(ms)) {
      noteOpenTimestamp(ms);
      const ymd = getYmd(ms);
      const prev = state.openCountsByDay.get(ymd) || 0;
      state.openCountsByDay.set(ymd, prev + 1);
    }

    markSnapshotDirty();
  }

  if (event.type === 'CLOSE') {
    const t = event.trade;
    const id = String(t.id ?? '');
    const symbol = t.symbol || 'BTCUSDT';
    const signalKey = t.signal_key || '';

    state.openTradesById.delete(id);
    const m = state.openTradesBySymbol.get(symbol);
    if (m) m.delete(id);

    if (signalKey) state.openTradeIdBySignalKey.delete(signalKey);
    if (signalKey) state.recentSignalSeenAt.set(signalKey, Date.now());

    markSnapshotDirty();
  }
}

function rebuildStateFromJournal() {
  state.openTradesById.clear();
  state.openTradesBySymbol.clear();
  state.openTradeIdBySignalKey.clear();
  state.seenEventKeys.clear();
  state.recentSignalSeenAt.clear();
  state.recentOpenTimes = [];
  state.openCountsByDay.clear();

  // reset OPEN_TRADE spam guard
  state.lastOpenTradesLogHash = '';
  state.lastOpenTradesLogCount = 0;

  const events = loadJsonl(getJournalPath(nowIso()));
  for (const ev of events) applyJournalEvent(ev);

  flushOpenPositionsSnapshot(true);
}

function persistJournalEvent(event) {
  if (state.seenEventKeys.has(event.event_key)) return true;
  const ok = appendJsonl(getJournalPath(event.ts || nowIso()), event);
  if (!ok) return false;
  applyJournalEvent(event);
  return true;
}

function getOpenTradesArray(symbol) {
  if (symbol) {
    const m = state.openTradesBySymbol.get(symbol);
    return m ? Array.from(m.values()) : [];
  }
  return Array.from(state.openTradesById.values());
}

function markSnapshotDirty() {
  snapshotDirty = true;
}

function computeSnapshotHash(trades) {
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
  const j = 0.15;
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

      if (status === 429 || status === 418) {
        const delay = jitter(baseDelayMs * Math.pow(2, i));
        console.warn('HTTP rate limited', { status, url: url.split('?')[0], delay });
        await sleep(delay);
        continue;
      }

      if (status >= 500 && status < 600) {
        const delay = jitter(baseDelayMs * Math.pow(2, i));
        console.warn('HTTP server error', { status, url: url.split('?')[0], delay });
        await sleep(delay);
        continue;
      }

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
    console.error('getTicker failed', EXCHANGE, symbol, e.message);
    return null;
  }
}

async function getTickerCached(symbol, timeoutMs, maxAgeMs) {
  const now = Date.now();
  const cache = state.tickerCacheBySymbol.get(symbol) || { ts: 0, price: null };
  if (cache.price !== null && (now - cache.ts) <= maxAgeMs) {
    return cache.price;
  }
  const price = await getTicker(symbol, timeoutMs);
  if (price !== null) state.tickerCacheBySymbol.set(symbol, { ts: now, price });
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
    console.error('getRecentKlines failed', EXCHANGE, symbol, e.message);
    return null;
  }
}

/* -----------------------------
 * RANK SYMBOLS (simple, low-frequency)
 * ----------------------------- */

async function rankSymbolsByRecentReturn(symbols, lookbackDays, httpTimeoutMs, verbose) {
  const days = Math.max(10, parseInt(lookbackDays || 90, 10));
  const scores = [];

  for (const sym of symbols) {
    const kl = await getRecentKlines(sym, days + 1, '1d', httpTimeoutMs);
    if (!Array.isArray(kl) || kl.length < 5) {
      if (verbose) console.log('RANK: skip symbol (no klines)', sym);
      continue;
    }

    // Use closes; exclude last (in-progress) daily candle by slicing -1
    const closed = kl.slice(0, kl.length - 1);
    const closes = closed.map(k => Number(k[4])).filter(Number.isFinite);
    if (closes.length < 5) continue;

    const first = closes[0];
    const last = closes[closes.length - 1];
    if (!(first > 0 && last > 0)) continue;

    const ret = (last / first) - 1;

    // Max drawdown on closes (rough but stable)
    let peak = closes[0];
    let mdd = 0;
    for (const c of closes) {
      if (c > peak) peak = c;
      if (peak > 0) {
        const dd = (peak - c) / peak;
        if (dd > mdd) mdd = dd;
      }
    }

    // Score: prefer higher return, penalize deep drawdown.
    const score = ret - 0.5 * mdd;
    scores.push({ symbol: sym, score, ret, mdd });
  }

  scores.sort((a, b) => b.score - a.score);
  if (verbose && scores.length) {
    console.log('RANK: symbols ordered (best first):');
    for (const s of scores.slice(0, Math.min(scores.length, 10))) {
      console.log('RANK:', s.symbol, 'score=', s.score.toFixed(4), 'ret=', s.ret.toFixed(4), 'mdd=', s.mdd.toFixed(4));
    }
  }

  const ordered = scores.map(s => s.symbol);
  // Keep any symbols that failed ranking at the end (original order)
  const missing = symbols.filter(s => !ordered.includes(s));
  return ordered.concat(missing);
}

/* -----------------------------
 * TRADING LOGIC (paper)
 * ----------------------------- */

function buildCloseReason(trade, market) {
  if (trade.stopLoss && market <= trade.stopLoss) return 'SL';
  if (trade.takeProfit && market >= trade.takeProfit) return 'TP';
  return 'OTHER';
}

async function openTrade(trade, candleBucketMs, signalCooldownMs, verbose) {
  const signalKey = getSignalKey(trade, candleBucketMs);

  if (hasOpenSignal(signalKey)) return false;
  if (!canOpenSignal(signalKey, signalCooldownMs)) return false;

  const event = {
    ts: nowIso(),
    type: 'OPEN',
    event_key: getOpenEventKey(trade, candleBucketMs),
    trade: normalizeOpenTrade(trade, candleBucketMs)
  };

  const ok = persistJournalEvent(event);
  if (!ok) return false;

  console.log(
    'OPEN (paper):', trade.openedAt,
    'id=', trade.id,
    'symbol=', trade.symbol,
    'regime=', trade.regime,
    'signalCandleTs=', trade.signalCandleTs,
    'signalKey=', signalKey,
    'entry=', trade.entryPrice,
    'SL=', trade.stopLoss,
    'TP=', trade.takeProfit,
    'REASON=', trade.reasonTag
  );

  return true;
}

async function closeTrade(trade, market, closeReason, candleBucketMs, feeRate) {
  if (!isValidNumber(market) || !isValidNumber(trade.entryPrice)) {
    console.error('closeTrade invalid values', { market, entryPrice: trade.entryPrice, id: trade.id });
    return false;
  }

  const qty = Number(trade.size || 0);
  trade.exitPrice = market;
  trade.profit = (market - trade.entryPrice) * qty;
  trade.profit_pct = trade.entryPrice > 0 ? (market - trade.entryPrice) / trade.entryPrice : null;
  trade.closedAt = nowIso();

  const feeUsdEst = (trade.entryPrice * qty + market * qty) * (feeRate || 0);
  trade.feeUsdEst = feeUsdEst;

  const reason = closeReason || buildCloseReason(trade, market);
  const event = {
    ts: nowIso(),
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
    'symbol=', trade.symbol,
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
 * MAIN LOOP (continuous, multi-symbol)
 * ----------------------------- */

(async () => {
  console.log('Starting SA service (continuous, multi-symbol capable)');
  ensureDir(BASE_DIR);

  if (!acquireLock()) process.exit(1);

  const cfgLive = loadSkillConfig();

  // Backward compatible: SYMBOL (string) OR SYMBOLS (array)
  const singleSymbol = (process.env.SYMBOL || cfgLive.SYMBOL || 'BTCUSDT').toUpperCase();
  let symbols = cfgLive.SYMBOLS;
  if (!Array.isArray(symbols) || symbols.length === 0) symbols = [singleSymbol];
  symbols = symbols.map(s => String(s || '').toUpperCase()).filter(Boolean);
  if (symbols.length === 0) symbols = [singleSymbol];

  // --- Primary settings ---
  const monitorInterval = parseInt(process.env.MONITOR_INTERVAL_MS || cfgLive.MONITOR_INTERVAL_MS || 5000, 10);
  const maxPositions = parseInt(process.env.MAX_POSITIONS || cfgLive.MAX_POSITIONS || 1, 10);
  const minHoldS = parseInt(process.env.MIN_HOLD_SECONDS || cfgLive.MIN_HOLD_SECONDS || '60', 10);
  const timeStopMinutes = parseInt(process.env.TIME_STOP_MINUTES || cfgLive.TIME_STOP_MINUTES || 10, 10);
  const HTTP_TIMEOUT_MS = parseInt(process.env.HTTP_TIMEOUT_MS || cfgLive.HTTP_TIMEOUT_MS || 2500, 10);
  const candleBucketMs = parseInt(process.env.SIGNAL_BUCKET_MS || cfgLive.SIGNAL_BUCKET_MS || cfgLive.CANDLE_MS || 60000, 10);
  const signalCooldownMs = parseInt(process.env.SIGNAL_COOLDOWN_MS || cfgLive.SIGNAL_COOLDOWN_MS || 180000, 10);

  const explosiveCandlePct = parseFloat(process.env.EXPLOSIVE_CANDLE_PCT || cfgLive.EXPLOSIVE_CANDLE_PCT || 0.003);
  const tradeUSD = parseFloat(process.env.TRADE_USD || cfgLive.TRADE_USD || 100.0);
  const minCandleBody = parseFloat(process.env.MIN_BODY_CANDLE || cfgLive.MIN_BODY_CANDLE || 0.5);
  const smaTol = parseFloat(process.env.SMA_TOLERANCE || cfgLive.SMA_TOLERANCE || 0.001);
  // Volume filter: interpret MIN_VOLUME as QUOTE volume (USDT) when available (kline[7]); fallback to base volume (kline[5]).
  const MIN_VOLUME = parseFloat(process.env.MIN_VOLUME || cfgLive.MIN_VOLUME || 0);
  // Minimum stop distance (legacy, ABS in price units; e.g., BTCUSDT: 80 means $80 move). Not multi-crypto friendly.
  const MIN_SL_USD = parseFloat(process.env.MIN_SL_USD || cfgLive.MIN_SL_USD || 0);
  // Recommended for multi-crypto: minimum stop distance as % of entry price (e.g. 0.001 = 0.1%). Overrides MIN_SL_USD when set.
  const MIN_SL_PCT = parseFloat(process.env.MIN_SL_PCT || cfgLive.MIN_SL_PCT || ''); // NaN if unset

  const minSMASlope = parseFloat(process.env.MIN_SMA_SLOPE || cfgLive.MIN_SMA_SLOPE || 2.0); // ABS (price units) legacy
  // Optional % slope threshold for multi-crypto (recommended). Example: 0.00003 ≈ 0.003% per candle.
  const minSMASlopePct = parseFloat(process.env.MIN_SMA_SLOPE_PCT || cfgLive.MIN_SMA_SLOPE_PCT || ''); // NaN if unset
  const SMA_WINDOW = parseInt(process.env.SMA_WINDOW || cfgLive.SMA_WINDOW || 60, 10);
  const limit = SMA_WINDOW + 2;

  const k_tp = parseFloat(process.env.K_TP || cfgLive.K_TP || '2.0');
  const k_sl = parseFloat(process.env.K_SL || cfgLive.K_SL || '1.2');

  const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || cfgLive.MIN_MOMENTUM_PCT || 0.0005);
  const MAX_MOMENTUM_PCT = parseFloat(process.env.MAX_MOMENTUM_PCT || cfgLive.MAX_MOMENTUM_PCT || 0.0012);

  const MOM_REDUCTION_PCT = parseFloat(process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || cfgLive.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || 0.0);
  const GREEN_KLINES = parseInt(process.env.GREEN_KLINES_FOR_REDUCTION || cfgLive.GREEN_KLINES_FOR_REDUCTION || 0, 10);

  const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || cfgLive.SYMBOL_COOLDOWN_SECONDS || 0, 10);
  const feeRate = parseFloat(process.env.FEE_RATE || cfgLive.FEE_RATE || 0.0005);
  const ATR_WINDOW = parseInt(process.env.ATR_WINDOW || cfgLive.ATR_WINDOW || 14, 10);
  const MIN_ATR_PCT = parseFloat(process.env.MIN_ATR_PCT || cfgLive.MIN_ATR_PCT || 0);

  // ATR adaptive (per symbol): require ATR% to be above a percentile of its own recent history.
  // This makes the filter scale naturally across BTC vs alts.
  const ATR_ADAPTIVE_ENABLED = (process.env.ATR_ADAPTIVE_ENABLED != null)
    ? String(process.env.ATR_ADAPTIVE_ENABLED) === '1'
    : !!cfgLive.ATR_ADAPTIVE_ENABLED;
  const ATR_ADAPTIVE_PCTL = parseFloat(process.env.ATR_ADAPTIVE_PCTL || cfgLive.ATR_ADAPTIVE_PCTL || 0.60);
  const ATR_ADAPTIVE_WINDOW = parseInt(process.env.ATR_ADAPTIVE_WINDOW || cfgLive.ATR_ADAPTIVE_WINDOW || 240, 10);
  const ATR_ADAPTIVE_MIN_SAMPLES = parseInt(process.env.ATR_ADAPTIVE_MIN_SAMPLES || cfgLive.ATR_ADAPTIVE_MIN_SAMPLES || 60, 10);

  // Optional: require market to be non-choppy by slope-normalization (abs(smaSlope) / (ATR_abs)).
  // Example: 0.30 matches the boundary used by detectMarketRegime() for CHOPPY.
  const MIN_SLOPE_NORM = parseFloat(process.env.MIN_SLOPE_NORM || cfgLive.MIN_SLOPE_NORM || 0);

  // Max trades per hour/day (per config)
  const MAX_TRADES_PER_HOUR = parseInt(process.env.MAX_TRADES_PER_HOUR || cfgLive.MAX_TRADES_PER_HOUR || 0, 10);
  const MAX_TRADES_PER_DAY = parseInt(process.env.MAX_TRADES_PER_DAY || cfgLive.MAX_TRADES_PER_DAY || 0, 10);

  // Verbose ON by default (new param optional). Env overrides.
  const VERBOSE = (process.env.VERBOSE != null)
    ? String(process.env.VERBOSE) !== '0'
    : (cfgLive.VERBOSE !== undefined ? !!cfgLive.VERBOSE : true);

  // snapshot interval: lower IO churn
  const SNAPSHOT_INTERVAL_MS = parseInt(process.env.SNAPSHOT_INTERVAL_MS || cfgLive.SNAPSHOT_INTERVAL_MS || 5000, 10);
  startSnapshotTimer(SNAPSHOT_INTERVAL_MS);

  // Align ticker cache to monitor interval
  const TICKER_CACHE_MS = parseInt(process.env.TICKER_CACHE_MS || cfgLive.TICKER_CACHE_MS || Math.max(250, Math.min(2000, monitorInterval)), 10);

  // Kline poll cadence (so we don't call klines every tick)
  const KLINES_POLL_MS = parseInt(process.env.KLINES_POLL_MS || cfgLive.KLINES_POLL_MS || 15000, 10);

  // Symbol ranking (prioritize recent winners; low-frequency)
  const RANKING_ENABLED = (cfgLive.SYMBOL_RANKING_ENABLED !== undefined)
    ? !!cfgLive.SYMBOL_RANKING_ENABLED
    : (symbols.length > 1);
  const RANK_LOOKBACK_DAYS = parseInt(process.env.SYMBOL_RANK_LOOKBACK_DAYS || cfgLive.SYMBOL_RANK_LOOKBACK_DAYS || 90, 10);
  const RANK_REFRESH_MINUTES = parseInt(process.env.SYMBOL_RANK_REFRESH_MINUTES || cfgLive.SYMBOL_RANK_REFRESH_MINUTES || 360, 10);

  if (VERBOSE) {
    console.log('VERBOSE=1 (default)');
  }
  console.log('Symbols configured:', symbols.join(','));
  console.log('MIN_VOLUME (quote preferred):', MIN_VOLUME);
  console.log('MIN_SL_USD:', MIN_SL_USD);
  console.log('MAX_TRADES_PER_HOUR:', MAX_TRADES_PER_HOUR);
  console.log('MAX_TRADES_PER_DAY:', MAX_TRADES_PER_DAY);

  rebuildStateFromJournal();

  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

  let symbolsOrdered = symbols.slice();
  let lastRankTs = 0;

  // Initial ranking (optional)
  if (RANKING_ENABLED && symbols.length > 1) {
    try {
      symbolsOrdered = await rankSymbolsByRecentReturn(symbols, RANK_LOOKBACK_DAYS, HTTP_TIMEOUT_MS, VERBOSE);
      lastRankTs = Date.now();
    } catch (e) {
      console.error('RANK init error:', e.message);
    }
  }

  while (!shutdownRequested) {
    iterCount++;
    const loopStart = Date.now();
    const tsStartIter = fmtDateUtc1(new Date());

    try {
      cleanupRecentSignals();

      // Refresh ranking occasionally
      if (RANKING_ENABLED && symbols.length > 1) {
        const ageMin = (Date.now() - lastRankTs) / 60000;
        if (!lastRankTs || ageMin >= RANK_REFRESH_MINUTES) {
          if (VERBOSE) console.log('RANK: refreshing symbols order...');
          try {
            symbolsOrdered = await rankSymbolsByRecentReturn(symbols, RANK_LOOKBACK_DAYS, HTTP_TIMEOUT_MS, VERBOSE);
            lastRankTs = Date.now();
          } catch (e) {
            console.error('RANK refresh error:', e.message);
          }
        }
      }

      // 1) Monitor open trades (TP/SL/time_stop) per symbol
      for (const sym of symbolsOrdered) {
        const openTrades = getOpenTradesArray(sym);
        if (!openTrades.length) continue;

        const market = await getTickerCached(sym, HTTP_TIMEOUT_MS, TICKER_CACHE_MS);
        if (market == null) continue;

        for (const tr of openTrades) {
          const ageS = (Date.now() - new Date(tr.openedAt).getTime()) / 1000;
          if (ageS < minHoldS) continue;

          if (ageS > timeStopMinutes * 60) {
            await closeTrade(tr, market, 'time_stop', candleBucketMs, feeRate);
          } else if (tr.stopLoss && market <= tr.stopLoss) {
            await closeTrade(tr, market, 'SL', candleBucketMs, feeRate);
          } else if (tr.takeProfit && market >= tr.takeProfit) {
            await closeTrade(tr, market, 'TP', candleBucketMs, feeRate);
          }
        }
      }

      // 2) Evaluate new candle entries per symbol (at most 1/min per symbol)
      for (const sym of symbolsOrdered) {
        const rt = getSymbolRuntime(sym);

        // Poll klines at cadence (avoid N symbols * every tick)
        if ((Date.now() - rt.lastKlinesPollMs) < KLINES_POLL_MS) continue;
        rt.lastKlinesPollMs = Date.now();

        const klines = await getRecentKlines(sym, limit, '1m', HTTP_TIMEOUT_MS);
        if (!Array.isArray(klines) || klines.length < 4) {
          if (VERBOSE) console.log('ANALYZE_SKIP: insufficient klines', { iter: iterCount, symbol: sym, got: Array.isArray(klines) ? klines.length : null });
          continue;
        }

        const lastClosed = klines[klines.length - 2];
        const _close = Number(lastClosed[4]);
        const _volBase = Number(lastClosed[5]);
        const _quoteVolEx = Number(lastClosed[7]);
        const _quoteVol = Number.isFinite(_quoteVolEx)
          ? _quoteVolEx
          : (Number.isFinite(_close) && Number.isFinite(_volBase) ? (_close * _volBase) : NaN);

        const candle = {
          ts: Number(lastClosed[0]),
          open: Number(lastClosed[1]),
          high: Number(lastClosed[2]),
          low: Number(lastClosed[3]),
          close: _close,
          volume: _volBase,
          // Prefer exchange-provided quote volume; fallback to close*baseVolume for consistent USDT-based MIN_VOLUME
          quoteVolume: _quoteVol
        };

        if (![candle.ts, candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)) {
          if (VERBOSE) console.log('ANALYZE_SKIP: invalid candle', { iter: iterCount, symbol: sym, candle });
          continue;
        }

        // Only analyze once per closed candle
        if (rt.lastAnalyzedClosedTs === candle.ts) continue;
        rt.lastAnalyzedClosedTs = candle.ts;

        const closes = getClosedCloses(klines);
        if (!Array.isArray(closes) || closes.length < 3) {
          if (VERBOSE) console.log('ANALYZE_SKIP: invalid closes', { iter: iterCount, symbol: sym });
          continue;
        }

        const last = closes[closes.length - 1];
        const prev = closes[closes.length - 2] || last;
        const momentum_pct = prev ? (last - prev) / prev : 0;

        // Filters
        const body = Math.abs(candle.close - candle.open);
        const range = candle.high - candle.low;
        const bodyRatio = range > 0 ? body / range : 0;
        const weakCandleBody = bodyRatio < minCandleBody;

        let weakOpen = false;
        if (klines.length >= 6) {
          const prevClose1 = Number(klines[klines.length - 3][4]);
          const prevClose2 = Number(klines[klines.length - 4][4]);
          if (Number.isFinite(prevClose1) && Number.isFinite(prevClose2)) {
            weakOpen = candle.open < prevClose1 && candle.open < prevClose2;
          }
        }

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

        const volMetric = Number.isFinite(candle.quoteVolume) ? candle.quoteVolume : candle.volume;
        const volumeOk = (MIN_VOLUME <= 0) ? true : (Number.isFinite(volMetric) && volMetric >= MIN_VOLUME);

        // Indicators
        const { sma, smaPrev } = computeSmaPair(closes, SMA_WINDOW);
        // Slope in ABS price units (legacy) + normalized slope in % of SMA (recommended for multi-crypto)
        const smaSlopeAbs = (sma != null && smaPrev != null) ? (sma - smaPrev) : 0;
        const smaSlopePct = (sma != null && smaPrev != null && smaPrev !== 0) ? (sma - smaPrev) / smaPrev : 0;
        const priceNearSMA = (sma != null) ? (candle.close >= sma * (1 - smaTol)) : true;
        const priceAboveSMA = (sma != null) ? (candle.close > sma) : true;

        const { atrPct } = computeAtr(klines, ATR_WINDOW);
        const atrPctNum = Number.isFinite(atrPct) ? atrPct : 0;

        // ATR adaptive (per symbol): update ATR history once per new closed candle.
        if (ATR_ADAPTIVE_ENABLED && Number.isFinite(candle.ts) && rt.lastAtrCandleTs !== candle.ts) {
          rt.lastAtrCandleTs = candle.ts;
          rt.atrHistory.push(atrPctNum);
          if (rt.atrHistory.length > ATR_ADAPTIVE_WINDOW) {
            rt.atrHistory.splice(0, rt.atrHistory.length - ATR_ADAPTIVE_WINDOW);
          }
        }

        let atrPctlThr = 0;
        if (ATR_ADAPTIVE_ENABLED && rt.atrHistory.length >= ATR_ADAPTIVE_MIN_SAMPLES) {
          atrPctlThr = percentile(rt.atrHistory, ATR_ADAPTIVE_PCTL);
        }

        const minAtrEffective = Math.max(MIN_ATR_PCT || 0, atrPctlThr || 0);
        const atrOk = (minAtrEffective > 0) ? (atrPctNum >= minAtrEffective) : true;

        const realizedVol = computeRealizedVol(closes, Math.min(60, Math.max(20, Math.floor(SMA_WINDOW / 2))));

        const regimeInfo = detectMarketRegime({
          atrPct: atrPctNum,
          realizedVol,
          smaSlope: smaSlopeAbs,
          lastClose: candle.close,
          priceAboveSma: priceAboveSMA
        });

        const slopeNormOk = (MIN_SLOPE_NORM > 0)
          ? (Number(regimeInfo.slopeNorm) >= MIN_SLOPE_NORM)
          : true;

        const regime = regimeInfo.volRegime;

        const dynamicMinMomentum = BASE_MIN_MOM * regimeInfo.kMinMomentum;
        const dynamicMinSlopeAbs = minSMASlope * regimeInfo.kMinSlope;
        const dynamicMinSlopePct = Number.isFinite(minSMASlopePct) ? (minSMASlopePct * regimeInfo.kMinSlope) : null;
        const dynamicMaxMomentum = MAX_MOMENTUM_PCT;

        // If MIN_SMA_SLOPE_PCT is set, use pct-based slope threshold (works across assets).
        // Otherwise fall back to ABS (price units) legacy behavior.
        const trendUp = (dynamicMinSlopePct != null)
          ? (smaSlopePct > dynamicMinSlopePct)
          : (smaSlopeAbs > dynamicMinSlopeAbs);

        // Green run
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

        const shouldEnter = atrOk &&
          slopeNormOk &&
          momentumOk &&
          trendUp &&
          priceNearSMA &&
          priceAboveSMA &&
          volumeOk &&
          !weakCandleBody &&
          !weakOpen &&
          !candleExplosive;

        const decision = {
          iter: iterCount,
          symbol: sym,
          momentum_pct: Number(momentum_pct.toFixed(6)),
          sma: sma != null ? Number(sma.toFixed(2)) : null,
          // Use adaptive precision so alts don't show 0.00 slope when slope is small
          smaSlope: Number(smaSlopeAbs.toFixed(candle.close < 10 ? 6 : 2)),
          smaSlopePct: Number(smaSlopePct.toFixed(6)),
          minSlopeAbs: Number(dynamicMinSlopeAbs.toFixed(candle.close < 10 ? 6 : 2)),
          minSlopePct: (dynamicMinSlopePct != null) ? Number(dynamicMinSlopePct.toFixed(6)) : null,
          atr_pct: Number((atrPctNum || 0).toFixed(6)),
          min_atr_floor: Number((MIN_ATR_PCT || 0).toFixed(6)),
          atr_pctl_thr: Number((atrPctlThr || 0).toFixed(6)),
          atr_pctl_q: ATR_ADAPTIVE_ENABLED ? Number((ATR_ADAPTIVE_PCTL || 0).toFixed(2)) : null,
          min_atr_pct: Number((minAtrEffective || 0).toFixed(6)),
          atrOk: !!atrOk,
          realized_vol: Number((realizedVol || 0).toFixed(6)),
          min_slope_norm: Number((MIN_SLOPE_NORM || 0).toFixed(3)),
          slopeNormOk: !!slopeNormOk,
          regime,
          microRegime: regimeInfo.microRegime,
          slopeNorm: regimeInfo.slopeNorm,
          priceNearSMA: !!priceNearSMA,
          priceAboveSMA: !!priceAboveSMA,
          volume_base: Number(candle.volume.toFixed(2)),
          volume_quote: Number.isFinite(candle.quoteVolume) ? Number(candle.quoteVolume.toFixed(2)) : null,
          volumeOk: !!volumeOk,
          trendUp: !!trendUp,
          weakCandleBody: !!weakCandleBody,
          weakOpen: !!weakOpen,
          candleExplosive: !!candleExplosive,
          shouldEnter: !!shouldEnter,
          effective_min_momentum: Number(effectiveMinMom.toFixed(6)),
          green_run
        };

        state.decisionBySymbol.set(sym, decision);

        // Verbose: show filter pass/fail
        if (VERBOSE) {
          console.log(
            'ITER_SUMMARY:',
            'iter=' + iterCount,
            'symbol=' + sym,
            'Started at: ' + tsStartIter,
            'with candle: ' + new Date(candle.ts + 3600000).toISOString().slice(11, 16),
            'volume last candle=' + candle.volume.toFixed(2),
            'quoteVol=' + (Number.isFinite(candle.quoteVolume) ? candle.quoteVolume.toFixed(2) : 'null'),
            'volOk=' + (volumeOk ? 1 : 0),
            'momentum=' + decision.momentum_pct,
            'sma=' + (decision.sma || 'null'),
            'smaSlope=' + decision.smaSlope,
            'smaSlopePct=' + decision.smaSlopePct,
            'minSlopeAbs=' + decision.minSlopeAbs,
            'minSlopePct=' + (decision.minSlopePct == null ? 'null' : decision.minSlopePct),
            'trendMode=' + (decision.minSlopePct == null ? 'abs' : 'pct'),
            'atr_pct=' + decision.atr_pct,
            'min_atr_pct=' + decision.min_atr_pct,
            'atrOk=' + (decision.atrOk ? 1 : 0),
            'slopeNorm=' + decision.slopeNorm,
            'min_slope_norm=' + decision.min_slope_norm,
            'slopeNormOk=' + (decision.slopeNormOk ? 1 : 0),
            'rv=' + decision.realized_vol,
            'regime=' + decision.regime,
            'micro=' + decision.microRegime,
            'priceNearSMA=' + (decision.priceNearSMA ? 1 : 0),
            'priceAboveSMA=' + (decision.priceAboveSMA ? 1 : 0),
            'trendUp=' + (decision.trendUp ? 1 : 0),
            'momentumOk=' + (momentumOk ? 1 : 0),
            'weakBody=' + (weakCandleBody ? 1 : 0),
            'weakOpen=' + (weakOpen ? 1 : 0),
            'explosive=' + (candleExplosive ? 1 : 0),
            'shouldEnter=' + (decision.shouldEnter ? 1 : 0),
            'effective_min_momentum=' + decision.effective_min_momentum,
            'green_run=' + decision.green_run,
            'openTrades=' + state.openTradesById.size,
            'openTradesSym=' + getOpenTradesArray(sym).length
          );
        } else {
          // Keep minimal compatibility line if verbose is disabled
          console.log(
            'ITER_SUMMARY:',
            'iter=' + iterCount,
            'symbol=' + sym,
            'Started at: ' + tsStartIter,
            'momentum=' + decision.momentum_pct,
            'sma=' + (decision.sma || 'null'),
            'smaSlope=' + decision.smaSlope,
            'smaSlopePct=' + decision.smaSlopePct,
            'minSlopePct=' + (decision.minSlopePct == null ? 'null' : decision.minSlopePct),
            'atr_pct=' + decision.atr_pct,
            'atrOk=' + (decision.atrOk ? 1 : 0),
            'slopeNorm=' + decision.slopeNorm,
            'slopeNormOk=' + (decision.slopeNormOk ? 1 : 0),
            'shouldEnter=' + (decision.shouldEnter ? 1 : 0),
            'openTrades=' + state.openTradesById.size
          );
        }

        console.log('---------------------------------------------\n');

        // Entry gating
        const nowTs = Date.now();
        const lastOpenTs = state.lastOpenBySymbol[sym] || 0;
        const withinCooldown = (cooldown_s > 0) && ((nowTs - lastOpenTs) < cooldown_s * 1000);
        const candleMinute = Math.floor(candle.ts / 60000);
        const alreadyOpenedThisCandle = rt.lastTradeCandleMinute !== null && rt.lastTradeCandleMinute === candleMinute;

        if (!shouldEnter) continue;

        if (state.openTradesById.size >= maxPositions) continue;

        if (withinCooldown) continue;

        if (alreadyOpenedThisCandle) continue;

        if (MAX_TRADES_PER_HOUR > 0) {
          const opensLastHour = countOpensLastHour();
          if (opensLastHour >= MAX_TRADES_PER_HOUR) continue;
        }

        if (MAX_TRADES_PER_DAY > 0) {
          const opensToday = countOpensTodayUtc();
          if (opensToday >= MAX_TRADES_PER_DAY) continue;
        }

        // Build trade
        const entryPrice = candle.close;
        const effectiveATR = atrPctNum;

        // TP/SL sizing:
        // - Base on ATR% (k_sl/k_tp)
        // - Enforce a minimum stop distance (prefer MIN_SL_PCT for multi-crypto; fallback MIN_SL_USD legacy)
        const rr = (k_sl > 0) ? (k_tp / k_sl) : 2.0;
        let riskDist = entryPrice * (k_sl * effectiveATR);

        if (Number.isFinite(MIN_SL_PCT) && MIN_SL_PCT > 0) {
          const minDist = entryPrice * MIN_SL_PCT;
          if (riskDist < minDist) riskDist = minDist;
        } else if (MIN_SL_USD > 0 && riskDist < MIN_SL_USD) {
          riskDist = MIN_SL_USD;
        }

        const stopLoss = entryPrice - riskDist;
        const takeProfit = entryPrice + riskDist * rr;

        const stopDistanceUSD = riskDist;
        const current = klines[klines.length - 1];
        const currentLow = Number(current?.[3]);

        const qty = Number((tradeUSD / entryPrice).toFixed(8));

        const grossProfitAtTp = qty * (takeProfit - entryPrice);
        const feeEstAtTp = (entryPrice * qty + takeProfit * qty) * feeRate;
        const netProfitAtTp = grossProfitAtTp - feeEstAtTp;

        if (!isValidNumber(entryPrice) || !isValidNumber(stopLoss) || !isValidNumber(takeProfit) || !isValidNumber(qty)) {
          console.error('invalid trade values', { symbol: sym, entryPrice, stopLoss, takeProfit, qty });
          continue;
        }
        if (stopDistanceUSD <= 0) continue;
        if (Number.isFinite(currentLow) && currentLow <= stopLoss) continue;
        if (netProfitAtTp <= 0) continue;

        const reasonDet = {
          momentum_pct: Number(momentum_pct.toFixed(6)),
          sma: sma != null ? Number(sma.toFixed(2)) : null,
          sma_slope: Number(smaSlopeAbs.toFixed(candle.close < 10 ? 6 : 2)),
          sma_slope_pct: Number(smaSlopePct.toFixed(6)),
          min_sma_slope_abs: Number(dynamicMinSlopeAbs.toFixed(candle.close < 10 ? 6 : 2)),
          min_sma_slope_pct: (dynamicMinSlopePct != null) ? Number(dynamicMinSlopePct.toFixed(6)) : null,
          atr_pct: Number((atrPctNum || 0).toFixed(6)),
          min_atr_floor: Number((MIN_ATR_PCT || 0).toFixed(6)),
          atr_pctl_thr: Number((atrPctlThr || 0).toFixed(6)),
          atr_pctl_q: ATR_ADAPTIVE_ENABLED ? Number((ATR_ADAPTIVE_PCTL || 0).toFixed(2)) : null,
          min_atr_pct: Number((minAtrEffective || 0).toFixed(6)),
          realized_vol: Number((realizedVol || 0).toFixed(6)),
          min_slope_norm: Number((MIN_SLOPE_NORM || 0).toFixed(3)),
          regime,
          micro_regime: regimeInfo.microRegime,
          slope_norm: regimeInfo.slopeNorm,
          base_min_momentum: BASE_MIN_MOM,
          effective_minimum: Number(effectiveMinMom.toFixed(6)),
          green_run_len: green_run,
          reduction_pct: MOM_REDUCTION_PCT,
          iter: iterCount
        };

        const reasonTag = (green_run >= GREEN_KLINES && MOM_REDUCTION_PCT > 0)
          ? 'momentum_with_green_run'
          : 'momentum_standard';

        const trade = {
          id: Date.now(),
          symbol: sym,
          entryPrice,
          stopLoss,
          takeProfit,
          openedAt: nowIso(),
          signalCandleTs: candle.ts,
          size: qty,
          exposureUSD: Number((entryPrice * qty).toFixed(2)),
          type: 'LONG',
          regime,
          reasonTag,
          reasonDetails: reasonDet
        };


        const didOpen = await openTrade(trade, candleBucketMs, signalCooldownMs, VERBOSE);
        if (didOpen) {
          rt.lastTradeCandleMinute = candleMinute;
          state.lastOpenBySymbol[sym] = Date.now();
        }
      }

      // Print open trades (compat) in verbose, but avoid spamming the same OPEN_TRADE every tick.
      // We only emit when the open-trades snapshot changes (open/close/update).
      if (VERBOSE) {
        const tradesNow = getOpenTradesArray();
        const hNow = computeSnapshotHash(tradesNow);
        const changed = (hNow !== state.lastOpenTradesLogHash) || (tradesNow.length !== state.lastOpenTradesLogCount);
        if (changed) {
          state.lastOpenTradesLogHash = hNow;
          state.lastOpenTradesLogCount = tradesNow.length;
          for (const ot of tradesNow) {
            console.log('OPEN_TRADE:', JSON.stringify(ot));
          }
        }
      }

    } catch (e) {
      console.error('ITERATION_FATAL_ERR (caught):', e.message);
    }

    const elapsed = Date.now() - loopStart;
    const sleepMs = Math.max(0, monitorInterval - elapsed);
    await sleep(sleepMs);
  }
})();
