#!/usr/bin/env node

/**
 * AegisTrade v4.2-14-3
 * Filter hierarchy
 * Add MEXC as exchange
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
require('dotenv').config();

/* PATHS & CONFIG */
const LABEL = process.env.LABEL || 'V4.2';
const BASE_DIR = path.join(__dirname, 'skills', `live-forward-${LABEL.toLowerCase()}`);
const CONFIG_PATH = path.join(BASE_DIR, 'config.json');
const LOCK_PATH = path.join(BASE_DIR, 'bot.lock');
const OPEN_POSITIONS_PATH = path.join(BASE_DIR, 'open_positions.json');

function loadSkillConfig() {return readJsonFile(CONFIG_PATH, {});}
function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    console.error('readJsonFile err:', filePath, e.message);
    return fallback;
  }
}
const cfg = loadSkillConfig();
const EXCHANGE = process.env.EXCHANGE || cfg.EXCHANGE || 'binance';

/* STATE */
let shutdownRequested = false;
let lockFd = null;
let snapshotTimer = null;
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

/* API EXCHANGE */
const API_BASES = {
  binance: 'https://api.binance.com',
  mexc: 'https://api.mexc.com'
};

function getApiBase() {return API_BASES[EXCHANGE] || API_BASES.binance;}

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

if (!ACTIVE_API_KEY || !ACTIVE_API_SECRET) {
  console.error(`API keys for ${EXCHANGE} not set in .env`);
  process.exit(1);
}

console.log("Exchange:", EXCHANGE);
console.log("API Base:", getApiBase());


/* HELPERS */
function ensureDir(dir) { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); }
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function isValidNumber(n) { return typeof n === 'number' && Number.isFinite(n); }



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

function getJournalPath(dateValue) {return path.join(BASE_DIR, `trade_journal_${getYmd(dateValue)}.jsonl`);}

function acquireLock() {
  ensureDir(BASE_DIR);
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
  return {
    id: trade.id,
    label: LABEL,
    symbol: trade.symbol || 'BTCUSDT',
    open_time_iso: openedAt,
    close_time_iso: closedAt,
    duration_s: openedAt ? Math.round((new Date(closedAt).getTime() - new Date(openedAt).getTime()) / 1000) : null,
    side: trade.type || 'LONG',
    qty: trade.size ?? 0,
    entry_price: trade.entryPrice ?? null,
    exit_price: trade.exitPrice ?? null,
    sl: trade.stopLoss ?? null,
    tp: trade.takeProfit ?? null,
    profit: trade.profit ?? null,
    profit_pct: trade.profit_pct ?? null,
    reason_tag: trade.reasonTag || '',
    close_reason: closeReason || '',
    reason_details: trade.reasonDetails || {},
    exposure_usd: trade.exposureUSD ?? null,
    signal_key: getSignalKey(trade, candleBucketMs)
  };
}

function fmtDateUtc1(d) {
  const dt = new Date(d.getTime() + 60 * 60 * 1000);
  const Y = dt.getUTCFullYear();
  const M = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const D = String(dt.getUTCDate()).padStart(2, '0');
  const h = String(dt.getUTCHours()).padStart(2, '0');
  const m = String(dt.getUTCMinutes()).padStart(2, '0');
  const s = String(dt.getUTCSeconds()).padStart(2, '0');
  return `${Y}-${M}-${D} ${h}:${m}:${s}`;
}

/* JOURNAL STATE MANAGEMENT */
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
  }

  if (event.type === 'CLOSE') {
    const t = event.trade;
    const id = String(t.id ?? '');
    const signalKey = t.signal_key || '';
    state.openTradesById.delete(id);
    if (signalKey) state.openTradeIdBySignalKey.delete(signalKey);
    if (signalKey) state.recentSignalSeenAt.set(signalKey, Date.now());
  }
}

function rebuildStateFromJournal() {
  state.openTradesById.clear();
  state.openTradeIdBySignalKey.clear();
  state.seenEventKeys.clear();
  state.recentSignalSeenAt.clear();

  const events = loadJsonl(getJournalPath(new Date().toISOString()));
  for (const event of events) applyJournalEvent(event);
  flushOpenPositionsSnapshot();
}

function persistJournalEvent(event) {
  if (state.seenEventKeys.has(event.event_key)) return true;
  const ok = appendJsonl(getJournalPath(event.ts || new Date().toISOString()), event);
  if (!ok) return false;
  applyJournalEvent(event);
  return true;
}

function getOpenTradesArray() {return Array.from(state.openTradesById.values());}

function flushOpenPositionsSnapshot() {
  return writeJsonFileAtomic(OPEN_POSITIONS_PATH, getOpenTradesArray().map(t => ({
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
  })));
}

function startSnapshotTimer() {
  snapshotTimer = setInterval(() => {
    try {
      flushOpenPositionsSnapshot();
    } catch (e) {
      console.error('flushOpenPositionsSnapshot err:', e.message);
    }
  }, 2000);
}

function cleanupRecentSignals(maxAgeMs = 10 * 60 * 1000) {
  const now = Date.now();
  for (const [k, ts] of state.recentSignalSeenAt.entries()) {
    if ((now - ts) > maxAgeMs) state.recentSignalSeenAt.delete(k);
  }
}

function hasOpenSignal(signalKey) {return state.openTradeIdBySignalKey.has(signalKey);}

function canOpenSignal(signalKey, cooldownMs) {
  const now = Date.now();
  const lastSeen = state.recentSignalSeenAt.get(signalKey) || 0;
  return (now - lastSeen) >= cooldownMs;
}

async function httpGetWithRetry(url, opts = {}, retries = 3, delayMs = 300, timeoutMs = 2500) {
  for (let i = 0; i < retries; i++) {
    try {
      return await axios.get(url, { ...opts, timeout: timeoutMs });
    } catch (e) {
      console.error('httpGetWithRetry attempt', i + 1, 'failed', e.message);
      if (i < retries - 1) await sleep(delayMs);
    }
  }
  throw new Error(`httpGetWithRetry failed after ${retries} attempts`);
}

async function getTicker(symbol, timeoutMs) {
  const base = getApiBase();
  const url = `${base}/api/v3/ticker/price?symbol=${symbol}`;
  try {
    const r = await httpGetWithRetry(url, ACTIVE_API_KEY ? { headers: { 'X-MBX-APIKEY': ACTIVE_API_KEY } } : {}, 3, 300, timeoutMs);
    if (r.data && (r.data.price || r.data.price === 0)) {return +r.data.price;}
  } catch (e) {
    console.error('getTicker failed', EXCHANGE, e.message);
  }
  return null;
}

async function getTickerCached(symbol, timeoutMs, maxAgeMs = 300) {
  const now = Date.now();
  if (state.lastTickerCache.price !== null && (now - state.lastTickerCache.ts) <= maxAgeMs) {
    return state.lastTickerCache.price;
  }
  const price = await getTicker(symbol, timeoutMs);
  if (price !== null) state.lastTickerCache = { ts: now, price };
  return price;
}

async function getRecentKlines(symbol, limit, interval, timeoutMs) {
  try {
    const base = getApiBase();
    const url = `${base}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const r = await httpGetWithRetry(url,
      ACTIVE_API_KEY ? { headers: { 'X-MBX-APIKEY': ACTIVE_API_KEY } } : {}, 3, 300, timeoutMs);
    return r && r.data ? r.data : null;
  } catch (e) {
    console.error('getRecentKlines failed', EXCHANGE, e.message);
    return null;
  }
}

async function signedBinanceRequest(method, endpoint, params) {
  const base = getApiBase();
  const qs = new URLSearchParams(params).toString();
  const signature = crypto.createHmac('sha256', ACTIVE_API_SECRET).update(qs).digest('hex');
  const url = `${base}${endpoint}?${qs}&signature=${signature}`;
  const headers = { 'X-MBX-APIKEY': ACTIVE_API_KEY };
  return axios({ method, url, headers, timeout: 5000 });
}

async function PlaceLiveOpenOrder(trade, symbol) {
  if (!isValidNumber(trade.entryPrice) || !isValidNumber(trade.stopLoss) || !isValidNumber(trade.takeProfit)) {
    console.error('Invalid trade values for live OCO', { trade });
    return;
  }
  try {
    const qty = trade.size;
    const res = await signedBinanceRequest('POST', '/api/v3/order/oco', {
      symbol,
      side: 'SELL',             
      quantity: String(qty),
      price: trade.takeProfit.toFixed(2),      // TP
      stopPrice: trade.stopLoss.toFixed(2),    // SL 
      stopLimitPrice: trade.stopLoss.toFixed(2),
      stopLimitTimeInForce: 'GTC'  
    });
    console.log('LIVE OCO ORDER PLACED:', JSON.stringify(res.data || res));
  } catch (e) {
    console.error('LIVE OCO ORDER ERROR:', e.message, e.response?.data || '');
  }
}

/* TRADE LOGINC */
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
    'signalCandleTs=', trade.signalCandleTs,
    'signalKey=', signalKey,
    'entry=', trade.entryPrice,
    'SL=', trade.stopLoss,
    'TP=', trade.takeProfit,
    'REASON=', trade.reasonTag
  );

  // await PlaceLiveOpenOrder(trade, symbol);
  return true;
}

async function closeTrade(trade, market, closeReason, symbol, candleBucketMs) {
if (!isValidNumber(market) || !isValidNumber(trade.entryPrice)) {
    console.error('closeTrade invalid values', { market, entryPrice: trade.entryPrice, id: trade.id });
    return false;
  }

  trade.exitPrice = market;
  trade.profit = (market - trade.entryPrice) * (trade.size || 0);
  trade.closedAt = new Date().toISOString();

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
    'profit=', trade.profit
  );
  return true;
}

function detectVolatilityRegime(atr_pct) {
if (atr_pct < 0.0008) {
    return { regime: "LOW_VOL", k_: 0.7}}

  if (atr_pct > 0.0015) {
    return { regime: "HIGH_VOL", k_: 1.2}}

  return {regime: "MID_VOL", k_: 1.0}
  }

/* GRATEFUL SHUTDOWN */
async function gracefulShutdown(signal) {
  if (shutdownRequested) return;
  shutdownRequested = true;
  console.log('shutdown signal:', signal);
  try {
    if (snapshotTimer) clearInterval(snapshotTimer);
    flushOpenPositionsSnapshot();
  } catch (e) {
    console.error('shutdown flush err:', e.message);
  }
  releaseLock();
  process.exit(0);
}

/* MAIN */
(async () => {
  console.log('Starting v4.2-14-03 (paper mode)');
  ensureDir(BASE_DIR);

  if (!acquireLock()) process.exit(1);

  const cfg = loadSkillConfig();
  const symbol = process.env.SYMBOL || cfg.SYMBOL || 'BTCUSDT';
  const monitorInterval = parseInt(process.env.MONITOR_INTERVAL_MS || cfg.MONITOR_INTERVAL_MS || 1000, 10);
  const maxPositions = parseInt(process.env.MAX_POSITIONS || cfg.MAX_POSITIONS || 2, 10);
  const minHoldS = parseInt(process.env.MIN_HOLD_SECONDS || cfg.MIN_HOLD_SECONDS || '60', 10);
  const timeStopMinutes = parseInt(process.env.TIME_STOP_MINUTES || cfg.TIME_STOP_MINUTES || 10, 10);
  const HTTP_TIMEOUT_MS = parseInt(process.env.HTTP_TIMEOUT_MS || cfg.HTTP_TIMEOUT_MS || 2500, 10);
  const candleBucketMs = parseInt(process.env.SIGNAL_BUCKET_MS || cfg.SIGNAL_BUCKET_MS || 60000, 10);
  const signalCooldownMs = parseInt(process.env.SIGNAL_COOLDOWN_MS || cfg.SIGNAL_COOLDOWN_MS || 180000, 10);
  const explosiveCandlePct = parseFloat(process.env.EXPLOSIVE_CANDLE_PCT || cfg.EXPLOSIVE_CANDLE_PCT || 0.003);
  const tradeUSD = parseFloat(process.env.TRADE_USD || cfg.TRADE_USD || 100.0);
  const minCandleBody = parseFloat(process.env.MIN_BODY_CANDLE || cfg.MIN_BODY_CANDLE || 0.5);
  const minSMASlope = parseFloat(process.env.MIN_SMA_SLOPE || cfg.MIN_SMA_SLOPE || 2.0);
  const SMA_WINDOW = parseInt(process.env.SMA_WINDOW || cfg.SMA_WINDOW || 60, 10);
  const limit = SMA_WINDOW + 2;
  const k_tp = parseFloat(process.env.K_TP || cfg.K_TP || '2.0');
  const k_sl = parseFloat(process.env.K_SL || cfg.K_SL || '1.2');
  const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || cfg.MIN_MOMENTUM_PCT || 0.0005);
  const MAX_MOMENTUM_PCT = parseFloat(process.env.MAX_MOMENTUM_PCT || cfg.MAX_MOMENTUM_PCT || 0.0012);
  const MOM_REDUCTION_PCT = parseFloat(process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || cfg.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || 0.0);
  const GREEN_KLINES = parseInt(process.env.GREEN_KLINES_FOR_REDUCTION || cfg.GREEN_KLINES_FOR_REDUCTION || 0, 10);
  const smaTol = parseFloat(process.env.SMA_TOLERANCE || cfg.SMA_TOLERANCE || 0.001);  
  const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || cfg.SYMBOL_COOLDOWN_SECONDS || 0, 10);
  const fees= parseFloat(process.env.FEE_RATE || cfg.FEE_RATE || 0.0015);
  const ATR_WINDOW = parseInt(cfg.ATR_WINDOW || 14, 10);
  
  rebuildStateFromJournal();
  startSnapshotTimer();

  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

  while (!shutdownRequested) {
    const tsStartIter = fmtDateUtc1(new Date());
    cleanupRecentSignals();
    
    const klines = await getRecentKlines(symbol, limit, '1m', HTTP_TIMEOUT_MS);

    let momentum_pct = 0;
    let sma = null;
    let smaPrev = null;
    let atr_pct = 0;
    let candle = null;
    let weakCandleBody = false;
    let weakOpen = false;
    let candleExplosive = false;
    let dynamicMinMomentum = BASE_MIN_MOM;
    let dynamicMaxMomentum = MAX_MOMENTUM_PCT;
    let dynamicMinSlope  = minSMASlope;

    // Calculations based on klines
    if (Array.isArray(klines) && klines.length >= 3) {
      const lastClosed = klines[klines.length - 2];
      candle = {
        ts: lastClosed[0],
        open: +lastClosed[1],
        high: +lastClosed[2],
        low: +lastClosed[3],
        close: +lastClosed[4],
        volume: +lastClosed[5],
        tsclose: +lastClosed[6]
      };

      const closes = klines.map(c => +c[4]);
      const last = closes[closes.length - 2];
      const prev = closes[closes.length - 3] || last;
      momentum_pct = prev ? (last - prev) / prev : 0;

      // Filter I: weak candle body
      const body = Math.abs(candle.close - candle.open);
      const range = candle.high - candle.low;
      const bodyRatio = range > 0 ? body / range : 0;
      weakCandleBody = bodyRatio < minCandleBody ;

      if (weakCandleBody) { console.log("Trade skipped: weak candle body:" , Number(bodyRatio.toFixed(2)) );}

      // Filter II & III: weak open when current below previous candles AND explosive candle detection
      if (Array.isArray(klines) && klines.length >= 4 && candle) {
        // Weak open: current open below previous two closes
        const prev1Close = +klines[klines.length - 2][4];
        const prev2Close = +klines[klines.length - 3][4];
        weakOpen = candle.open < prev1Close && candle.open < prev2Close;
        if (weakOpen) { console.log("Trade skipped: detect a weak open-candle", Number(candle.open.toFixed(2)) ) ;}
        // Explosive candle: check last 2 candles ranges
        const ranges = [2, 3].map(i => {
          const k = klines[klines.length - i];
          return (+k[2] - +k[3]) / +k[4]; // (high - low) / close
        });
        candleExplosive = ranges.some(r => r > explosiveCandlePct);
        if (candleExplosive) {console.log("Filter: explosive candle:", {ranges: ranges.map(r => Number(r.toFixed(6))) });}
      }

      
      if (!weakCandleBody && !weakOpen && !candleExplosive) {
        // SMA calculation
        if (closes.length >= SMA_WINDOW + 1) {
          const lastWindow = closes.slice(-SMA_WINDOW - 1, -1);
          sma = lastWindow.reduce((a, b) => a + b, 0) / lastWindow.length;
          const prevWindow = closes.slice(-SMA_WINDOW - 2, -2);
          smaPrev = prevWindow.length ? prevWindow.reduce((a, b) => a + b, 0) / prevWindow.length : null;
        } else {
          sma = closes.reduce((a, b) => a + b, 0) / closes.length;
          smaPrev = null;
        }
        // ATR calculation
        const trs = [];
        for (let i = klines.length - ATR_WINDOW - 1; i < klines.length - 1; i++) {
          const high = +klines[i][2];
          const low = +klines[i][3];
          const prevClose = +klines[i - 1][4];
          trs.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
        }
        const atr = trs.length ? trs.reduce((a, b) => a + b, 0) / trs.length : 0;
        atr_pct = atr / (last || 1);
        const vol = detectVolatilityRegime(atr_pct);
        const regime = vol.regime
        dynamicMinMomentum = dynamicMinMomentum* vol.k_;
        dynamicMaxMomentum = dynamicMaxMomentum * vol.k_;
        dynamicMinSlope = dynamicMinSlope * vol.k_;
        console.log(`Volatility regime: ${regime} | ATR%: ${atr_pct.toFixed(6)} | ` +  `k_: ${vol.k_}  `);
      }
    }
        
    // SMA slope y price near SMA
    const smaSlope = (sma !== null && smaPrev !== null) ? (sma - smaPrev) : 0;
    const priceNearSMA = (sma !== null && candle) ? (candle.close >= sma * (1 - smaTol)) : true;
    const trendUp = smaSlope > dynamicMinSlope;
    const priceAboveSMA = (sma !== null && candle) ? (candle.close > sma) : true;
    
    // Filter by momentum with dynamic reduction if we have a run of green candles
    let green_run = 0;
    if (GREEN_KLINES > 0 && Array.isArray(klines)) {
      for (let j = klines.length - 2; j > 0 && green_run < GREEN_KLINES; j--) {
        const cur = +klines[j][4];
        const op = +klines[j][1];
        if (cur > op) green_run++; else break;
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

    const nowTs = Date.now();
    const lastOpenTs = state.lastOpenBySymbol[symbol] || 0;
    const withinCooldown = (cooldown_s > 0) && ((nowTs - lastOpenTs) < cooldown_s * 1000);
    const candleMinute = candle ? Math.floor(candle.ts / 60000) : null;
    const alreadyOpenedThisCandle = candleMinute !== null && lastTradeCandle === candleMinute;

    // Final entry decision
    if (candle && shouldEnter && state.openTradesById.size < maxPositions && !withinCooldown && !alreadyOpenedThisCandle) {
      const entryPrice = candle.close;
      const effectiveATR = atr_pct;
      let stopLoss = entryPrice * (1 - k_sl * effectiveATR);
      const takeProfit = entryPrice * (1 + k_tp * effectiveATR);

      const stopDistanceUSD = entryPrice - stopLoss;

      // Filter: stop loss inside candle
      const current = klines[klines.length - 1];
      const currentLow  = +current[3];
      
      // Filter by profitability with fees
      const qty = Number((tradeUSD / entryPrice).toFixed(8));
      const grossProfit = qty * (takeProfit - entryPrice);
      const feeRate = tradeUSD * fees;
      const netProfit = grossProfit - feeRate;
      console.log('Aprox. net profit: ', {grossProfit, netProfit});

      if (stopDistanceUSD <= 0) {
        console.error('Trade skipped: Invalid stop distance', { entryPrice, stopLoss });
      } else if (currentLow <= stopLoss) {
        console.log('Trade skipped: SL inside candle', { entryPrice, stopLoss, currentLow, candleLow: candle.low });
      } else if (netProfit <= 0) { 
        console.log('Trade skipped: Not profitable after fees', { entryPrice, takeProfit, grossProfit, netProfit, feeRate });
      }else {

         const reasonDet = {
          momentum_pct: Number(momentum_pct.toFixed(6)),
          sma: sma != null ? Number(sma.toFixed(2)) : null,
          atr_pct: Number(atr_pct.toFixed(6)),
          base_min_momentum: BASE_MIN_MOM,
          effective_minimum: Number(effectiveMinMom.toFixed(6)),
          green_run_len: green_run,
          reduction_pct: MOM_REDUCTION_PCT
        };
        const reasonTag = (green_run >= GREEN_KLINES && MOM_REDUCTION_PCT > 0) ? 'momentum_with_green_run' : 'momentum_standard';

        if (!isValidNumber(entryPrice) || !isValidNumber(stopLoss) || !isValidNumber(takeProfit) || !isValidNumber(qty)) {
          console.error('invalid trade values', { entryPrice, stopLoss, takeProfit, qty });
        } else {
          const trade = {
            id: Date.now(),
            symbol: symbol,
            entryPrice,
            stopLoss,
            takeProfit,
            openedAt: new Date().toISOString(),
            signalCandleTs: candle.ts,
            size: qty,
            exposureUSD: Number((entryPrice * qty).toFixed(2)),
            type: 'LONG',
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
    }

    const monitorStart = Date.now();
    while (!shutdownRequested && (Date.now() - monitorStart < 60 * 1000)) {
      const loopStart = Date.now();
      try {
        const market = await getTickerCached(symbol, HTTP_TIMEOUT_MS, 300);
        if (market !== null) {
          for (const trade of getOpenTradesArray()) {
            const age = (Date.now() - new Date(trade.openedAt).getTime()) / 1000;
            if (age < minHoldS) continue;

            if (age > timeStopMinutes * 60) {
              await closeTrade(trade, market, 'time_stop', symbol, candleBucketMs);
            } else if (trade.stopLoss && market <= trade.stopLoss) {
              await closeTrade(trade, market, 'SL', symbol, candleBucketMs);
            } else if (trade.takeProfit && market >= trade.takeProfit) {
              await closeTrade(trade, market, 'TP', symbol, candleBucketMs);
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

    console.log(
      'ITER_SUMMARY:',
      'Started at: ' + tsStartIter,
      'with candle: ' + new Date(candle.ts + 3600000).toISOString().slice(11,16),
      "volume last candle=" + candle.volume.toFixed(2),
      'momentum=' + (state.lastDecision.momentum_pct || 0),
      'sma=' + (state.lastDecision.sma || 'null'),
      'smaSlope=' + (state.lastDecision.smaSlope || 0),
      'priceNearSMA=' + (state.lastDecision.priceNearSMA ? 1 : 0),
      'priceAboveSMA=' + (state.lastDecision.priceAboveSMA ? 1 : 0),      
      'trendUp=' + (state.lastDecision.trendUp ? 1 : 0),
      'shouldEnter=' + (state.lastDecision.shouldEnter ? 1 : 0),
      'effective_min_momentum=' + (state.lastDecision.effective_min_momentum || 0),
      'green_run=' + (state.lastDecision.green_run || 0),
      'openTrades=' + state.openTradesById.size
    );
    console.log('---------------------------------------------\n');

    for (const ot of getOpenTradesArray()) {
      console.log('OPEN_TRADE:', JSON.stringify(ot));
    }
  }
})();
