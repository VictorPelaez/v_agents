#!/usr/bin/env node

/**
 * Live forward service (paper trading)
 *
 * Main goals:
 * - Keep compatibility with current env vars and current trade shape.
 * - Use daily rotating JSON files only.
 * - Prevent duplicate logical opens/closes in JSON and CSV.
 * - Improve robustness and keep the loop fast.
 * - Keep comments in English.
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const child_process = require('child_process');

const astBase = process.env.BINANCE_API_BASE || 'https://api.binance.com';
const LABEL = process.env.LABEL || 'V4.2';
const BASE_DIR = path.join(__dirname, 'skills', `live-forward-${LABEL.toLowerCase()}`);
const CONFIG_PATH = path.join(BASE_DIR, 'config.json');
const STATE_PATH = path.join(BASE_DIR, 'runtime_state.json');
const CSV_PATH_DEFAULT = path.join(BASE_DIR, 'TRADE_LOG_ad.csv');
const FIFO_PATH_DEFAULT = path.join(BASE_DIR, 'csv_cmd.fifo');

let lastTradeCandle = null;
let shutdownRequested = false;
let runtimeStateDirty = false;
let flushStateTimer = null;
let pruneStateTimer = null;

const fileLocks = new Map();

let BINANCE_API_KEY = process.env.BINANCE_API_KEY || '';
try {
  if (!BINANCE_API_KEY) {
    const keyPath = path.join(__dirname, 'API_KEYS.md');
    if (fs.existsSync(keyPath)) {
      BINANCE_API_KEY = fs.readFileSync(keyPath, 'utf8').split(/\r?\n/)[0].trim();
    }
  }
} catch (e) {
  BINANCE_API_KEY = '';
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function isValidNumber(n) {
  return typeof n === 'number' && Number.isFinite(n);
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

/**
 * Atomic write with a unique tmp file per call.
 * This avoids collisions like:
 * ENOENT rename runtime_state.json.tmp -> runtime_state.json
 */
function writeJsonFileAtomic(filePath, value) {
  try {
    ensureDir(path.dirname(filePath));

    const uniqueSuffix = `${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}`;
    const tmpPath = `${filePath}.${uniqueSuffix}.tmp`;

    fs.writeFileSync(tmpPath, JSON.stringify(value, null, 2), 'utf8');
    fs.renameSync(tmpPath, filePath);
    return true;
  } catch (e) {
    console.error('writeJsonFileAtomic err:', filePath, e.message);
    return false;
  }
}

function loadSkillConfig() {
  return readJsonFile(CONFIG_PATH, {});
}

function getYmd(dateValue) {
  const dt = new Date(dateValue || Date.now());
  return String(dt.getUTCFullYear()) +
    String(dt.getUTCMonth() + 1).padStart(2, '0') +
    String(dt.getUTCDate()).padStart(2, '0');
}

function getOpenTradesFile(dateValue) {
  return path.join(BASE_DIR, `open_trades_${getYmd(dateValue)}.json`);
}

function getCloseTradesFile(dateValue) {
  return path.join(BASE_DIR, `close_trades_${getYmd(dateValue)}.json`);
}

function ensureDailyFile(filePath) {
  if (!fs.existsSync(filePath)) {
    writeJsonFileAtomic(filePath, []);
  }
}

function roundPriceForKey(value, decimals = 2) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) return '0.00';
  return num.toFixed(decimals);
}

/**
 * Technical key for exact event identity.
 */
function getTechnicalTradeKey(t, type) {
  if (t && t.id !== undefined && t.id !== null) return `${type}:${String(t.id)}`;
  const openTime = t.openedAt || t.open_time_iso || '';
  const closeTime = t.closedAt || t.close_time_iso || '';
  const entry = t.entryPrice ?? t.entry_price ?? '';
  const exit = t.exitPrice ?? t.exit_price ?? '';
  return `${type}:${openTime}:${closeTime}:${entry}:${exit}`;
}

/**
 * Logical key for open dedupe.
 * Two opens are considered the same trade if they share:
 * - label
 * - symbol
 * - side
 * - same candle minute
 * - rounded entry price
 * - same reason tag
 */
function getOpenBusinessKey(t) {
  const ts = new Date(t.openedAt || t.open_time_iso || Date.now()).getTime();
  const candleMinute = Math.floor(ts / 60000);
  const entry = roundPriceForKey(t.entryPrice ?? t.entry_price, 2);
  const side = t.type || t.side || 'LONG';
  const symbol = t.symbol || 'BTC';
  const reasonTag = t.reasonTag || t.reason_tag || '';
  return `open:${LABEL}:${symbol}:${side}:${candleMinute}:${entry}:${reasonTag}`;
}

/**
 * Logical key for close dedupe.
 * Same trade id + same close reason = same logical close.
 */
function getCloseBusinessKey(t, closeReason) {
  const symbol = t.symbol || 'BTC';
  const id = String(t.id ?? '');
  const reason = closeReason || t.close_reason || 'UNKNOWN';
  return `close:${LABEL}:${symbol}:${id}:${reason}`;
}

function normalizeOpenTradeRecord(t) {
  return {
    id: t.id,
    label: LABEL,
    symbol: t.symbol || 'BTC',
    open_time_iso: t.openedAt || t.open_time_iso || new Date().toISOString(),
    side: t.type || t.side || 'LONG',
    qty: t.size ?? t.qty ?? 0,
    entry_price: t.entryPrice ?? t.entry_price ?? null,
    sl: t.stopLoss ?? t.sl ?? null,
    tp: t.takeProfit ?? t.tp ?? null,
    reason_tag: t.reasonTag || t.reason_tag || '',
    reason_details: t.reasonDetails || t.reason_details || {},
    exposure_usd: t.exposureUSD ?? t.exposure_usd ?? null
  };
}

function normalizeCloseTradeRecord(t, closeReason) {
  const openedAt = t.openedAt || t.open_time_iso || '';
  const closedAt = t.closedAt || t.close_time_iso || new Date().toISOString();
  const durationSeconds = t.duration_s || (openedAt && closedAt
    ? Math.round((new Date(closedAt).getTime() - new Date(openedAt).getTime()) / 1000)
    : null);

  return {
    id: t.id,
    label: LABEL,
    symbol: t.symbol || 'BTC',
    open_time_iso: openedAt,
    close_time_iso: closedAt,
    duration_s: durationSeconds,
    side: t.type || t.side || 'LONG',
    qty: t.size ?? t.qty ?? 0,
    entry_price: t.entryPrice ?? t.entry_price ?? null,
    exit_price: t.exitPrice ?? t.exit_price ?? null,
    sl: t.stopLoss ?? t.sl ?? null,
    tp: t.takeProfit ?? t.tp ?? null,
    profit: t.profit ?? null,
    profit_pct: t.profit_pct ?? null,
    reason_tag: t.reasonTag || t.reason_tag || '',
    close_reason: closeReason || t.close_reason || '',
    reason_details: t.reasonDetails || t.reason_details || {},
    exposure_usd: t.exposureUSD ?? t.exposure_usd ?? null
  };
}

function loadRuntimeState() {
  return readJsonFile(STATE_PATH, {
    persistedOpenKeys: {},
    persistedCloseKeys: {},
    csvOpenKeys: {},
    csvCloseKeys: {}
  });
}

const runtimeState = loadRuntimeState();

function markRuntimeStateDirty() {
  runtimeStateDirty = true;
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * In-process lock per file path.
 * This prevents concurrent read-modify-write races inside the same process.
 */
async function withFileLock(filePath, fn) {
  while (fileLocks.get(filePath)) {
    await sleep(5);
  }
  fileLocks.set(filePath, true);
  try {
    return await fn();
  } finally {
    fileLocks.delete(filePath);
  }
}

async function saveRuntimeState() {
  return withFileLock(STATE_PATH, async () => {
    const ok = writeJsonFileAtomic(STATE_PATH, runtimeState);
    if (ok) runtimeStateDirty = false;
    return ok;
  });
}

async function flushRuntimeState() {
  if (!runtimeStateDirty) return true;
  return saveRuntimeState();
}

function startRuntimeStateAutoFlush() {
  flushStateTimer = setInterval(() => {
    flushRuntimeState().catch(e => {
      console.error('flushRuntimeState err', e.message);
    });
  }, 2000);
}

function pruneRuntimeState(maxAgeMs = 7 * 24 * 60 * 60 * 1000) {
  const now = Date.now();
  for (const bucket of ['persistedOpenKeys', 'persistedCloseKeys', 'csvOpenKeys', 'csvCloseKeys']) {
    const obj = runtimeState[bucket] || {};
    for (const [k, ts] of Object.entries(obj)) {
      if (!ts || (now - ts) > maxAgeMs) delete obj[k];
    }
  }
  markRuntimeStateDirty();
}

function startRuntimeStatePrune() {
  pruneStateTimer = setInterval(() => {
    try {
      pruneRuntimeState();
    } catch (e) {
      console.error('pruneRuntimeState err', e.message);
    }
  }, 60 * 60 * 1000);
}

function hasStateKey(bucket, key) {
  return !!(runtimeState[bucket] && runtimeState[bucket][key]);
}

function markOnceAfterSuccess(bucket, key) {
  runtimeState[bucket] = runtimeState[bucket] || {};
  runtimeState[bucket][key] = Date.now();
  markRuntimeStateDirty();
}

async function upsertTradeRecord(filePath, record, businessKey) {
  return withFileLock(filePath, async () => {
    ensureDir(path.dirname(filePath));
    ensureDailyFile(filePath);

    const arr = readJsonFile(filePath, []);
    const list = Array.isArray(arr) ? arr : [];

    const nextRecord = { ...record, _dedupe_key: businessKey };
    const idx = list.findIndex(item => item && item._dedupe_key === businessKey);

    if (idx >= 0) {
      list[idx] = { ...list[idx], ...nextRecord };
    } else {
      list.push(nextRecord);
    }

    return writeJsonFileAtomic(filePath, list);
  });
}

async function persistOpenTrade(t) {
  try {
    const record = normalizeOpenTradeRecord(t);
    const businessKey = getOpenBusinessKey(record);

    if (hasStateKey('persistedOpenKeys', businessKey)) return true;

    const ok = await upsertTradeRecord(
      getOpenTradesFile(record.open_time_iso),
      record,
      businessKey
    );

    if (ok) markOnceAfterSuccess('persistedOpenKeys', businessKey);
    return ok;
  } catch (e) {
    console.error('persistOpenTrade err:', e.message || e);
    return false;
  }
}

async function persistCloseTrade(t, closeReason) {
  try {
    const record = normalizeCloseTradeRecord(t, closeReason);
    const businessKey = getCloseBusinessKey(record, closeReason);

    if (hasStateKey('persistedCloseKeys', businessKey)) return true;

    const ok = await upsertTradeRecord(
      getCloseTradesFile(record.close_time_iso),
      record,
      businessKey
    );

    if (ok) markOnceAfterSuccess('persistedCloseKeys', businessKey);
    return ok;
  } catch (e) {
    console.error('persistCloseTrade err:', e.message || e);
    return false;
  }
}

/**
 * CSV writer with logical dedupe.
 * It first tries FIFO, then falls back to the Python helper.
 */
function writeTradeCsv(entry, closeReason = '') {
  try {
    const action = entry.action || 'close';
    const onceBucket = action === 'open' ? 'csvOpenKeys' : 'csvCloseKeys';
    const businessKey = action === 'open'
      ? getOpenBusinessKey(entry)
      : getCloseBusinessKey(entry, closeReason || entry.close_reason || '');

    if (hasStateKey(onceBucket, businessKey)) return true;

    const payload = {
      id: entry.id,
      label: LABEL,
      symbol: entry.symbol || 'BTC',
      open_time_iso: entry.openedAt || entry.open_time_iso || '',
      close_time_iso: entry.closedAt || entry.close_time_iso || entry.timestamp || '',
      duration_s: entry.duration_s || (
        (entry.closedAt || entry.close_time_iso || entry.timestamp) && (entry.openedAt || entry.open_time_iso)
          ? Math.round(
              (new Date(entry.closedAt || entry.close_time_iso || entry.timestamp).getTime() -
                new Date(entry.openedAt || entry.open_time_iso).getTime()) / 1000
            )
          : ''
      ),
      side: entry.type || entry.side || 'LONG',
      qty: entry.size ?? entry.qty ?? '',
      entry_price: entry.entryPrice ?? entry.entry_price ?? '',
      exit_price: entry.exitPrice ?? entry.exit_price ?? '',
      sl: entry.stopLoss ?? entry.sl ?? '',
      tp: entry.takeProfit ?? entry.tp ?? '',
      profit: entry.profit ?? '',
      profit_pct: entry.profit_pct ?? '',
      reason_details: entry.reasonDetails || entry.reason_details || ''
    };

    const fifoPath = process.env.CSV_FIFO_PATH || FIFO_PATH_DEFAULT;
    try {
      const stream = fs.createWriteStream(fifoPath, { flags: 'a' });
      const cmdObj = { cmd: action, data: payload };
      stream.write(JSON.stringify(cmdObj) + '\n');
      stream.end();

      markOnceAfterSuccess(onceBucket, businessKey);
      return true;
    } catch (e) {
      try {
        const pythonScriptPath = process.env.PYTHON_SCRIPT_PATH || path.join(__dirname, 'tools', 'csv_writer.py');
        const p = child_process.spawn('python3', [pythonScriptPath, action, JSON.stringify(payload)], {
          stdio: ['ignore', 'pipe', 'pipe']
        });

        p.stdout.on('data', d => console.log('csv_writer stdout:', d.toString().trim()));
        p.stderr.on('data', d => console.error('csv_writer stderr:', d.toString().trim()));
        p.on('exit', (code, signal) => {
          if (code !== 0) console.error('csv_writer exited non-zero', code, signal);
        });

        markOnceAfterSuccess(onceBucket, businessKey);
        return true;
      } catch (err) {
        console.error('writeTradeCsv fallback spawn err', err.message);
        return false;
      }
    }
  } catch (e) {
    console.error('writeTradeCsv err', e.message);
    return false;
  }
}

function buildCloseReason(t, market) {
  if (t.stopLoss && market <= t.stopLoss) return 'SL';
  if (t.takeProfit && market >= t.takeProfit) return 'TP';
  return 'OTHER';
}

/**
 * Additional in-memory protection against opening the same logical trade twice.
 */
function existsEquivalentOpenTrade(openTrades, candidate) {
  const candidateKey = getOpenBusinessKey(candidate);
  return openTrades.some(t => getOpenBusinessKey(t) === candidateKey);
}

/**
 * Single close path for all exit types.
 */
async function closeTrade(t, market, closeReason) {
  if (!isValidNumber(market) || !isValidNumber(t.entryPrice)) {
    console.error('closeTrade invalid price values', {
      market,
      entryPrice: t.entryPrice,
      id: t.id
    });
    return false;
  }

  t.exitPrice = market;
  t.profit = (market - t.entryPrice) * (t.size || 0);
  t.closedAt = new Date().toISOString();

  writeTradeCsv({ ...t, action: 'close', close_reason: closeReason }, closeReason);
  const ok = await persistCloseTrade(t, closeReason || buildCloseReason(t, market));

  if (ok) {
    console.log('CLOSE:', closeReason, t.id, t.exitPrice, t.profit);
  }

  return ok;
}

function loadClosedTradeIdsForToday() {
  const closePath = getCloseTradesFile(new Date().toISOString());
  ensureDailyFile(closePath);

  const data = readJsonFile(closePath, []);
  const ids = new Set();

  for (const row of Array.isArray(data) ? data : []) {
    if (row && row.id !== undefined && row.id !== null) {
      ids.add(String(row.id));
    }
  }

  return ids;
}

function backfillOpenTrades(openTrades, openTradesById) {
  const todayPath = getOpenTradesFile(new Date().toISOString());
  ensureDailyFile(todayPath);

  const closedIds = loadClosedTradeIdsForToday();
  const data = readJsonFile(todayPath, []);
  if (!Array.isArray(data)) return;

  const seen = new Set();

  for (const o of data) {
    try {
      const t = {
        id: o.id || Date.now(),
        entryPrice: Number(o.entry_price ?? o.entryPrice ?? 0),
        stopLoss: o.sl != null ? Number(o.sl) : null,
        takeProfit: o.tp != null ? Number(o.tp) : null,
        openedAt: o.open_time_iso || o.openedAt || new Date().toISOString(),
        size: Number(o.qty ?? o.size ?? 0),
        type: o.side || o.type || 'LONG',
        reasonTag: o.reason_tag || 'backfill',
        reasonDetails: o.reason_details || {},
        exposureUSD: o.exposure_usd ?? null,
        symbol: o.symbol || 'BTC'
      };

      if (closedIds.has(String(t.id))) continue;

      const key = getOpenBusinessKey(t);
      if (seen.has(key)) continue;
      seen.add(key);

      openTrades.push(t);
      openTradesById.set(t.id, t);
    } catch (e) {
      console.error('backfill parse err', e.message);
    }
  }
}

(async () => {
  console.log('starting live service (paper decisions)');
  ensureDir(BASE_DIR);

  const skillCfg = loadSkillConfig();
  startRuntimeStateAutoFlush();
  startRuntimeStatePrune();

  const symbol = process.env.SYMBOL || skillCfg.SYMBOL || 'BTCUSDT';
  const skillCapital = parseFloat(process.env.CAPITAL || skillCfg.CAPITAL || 1000);
  const riskPct = parseFloat(process.env.RISK_PCT || skillCfg.RISK_PCT || 0.01);
  const monitorInterval = parseInt(process.env.MONITOR_INTERVAL_MS || skillCfg.MONITOR_INTERVAL_MS || 1000, 10);
  const configuredCsvPath = process.env.CSV_PATH || skillCfg.CSV_PATH || '';
  const CSV_PATH = configuredCsvPath || CSV_PATH_DEFAULT;
  const maxPositions = parseInt(process.env.MAX_POSITIONS || skillCfg.MAX_POSITIONS || 2, 10);
  const minHoldS = parseInt(process.env.MIN_HOLD_SECONDS || skillCfg.MIN_HOLD_SECONDS || '60', 10);
  const timeStopMinutes = parseInt(process.env.TIME_STOP_MINUTES || skillCfg.TIME_STOP_MINUTES || 10, 10);
  const HTTP_TIMEOUT_MS = parseInt(process.env.HTTP_TIMEOUT_MS || skillCfg.HTTP_TIMEOUT_MS || 2500, 10);

  const openTrades = [];
  const openTradesById = new Map();
  const lastOpenBySymbol = {};
  let lastDecision = {};
  let lastTickerCache = { ts: 0, price: null };

  backfillOpenTrades(openTrades, openTradesById);

  async function httpGetWithRetry(url, opts = {}, retries = 3, delayMs = 300) {
    for (let i = 0; i < retries; i++) {
      try {
        return await axios.get(url, { ...opts, timeout: HTTP_TIMEOUT_MS });
      } catch (e) {
        console.error('httpGetWithRetry attempt', i + 1, 'failed', e.message);
        if (i < retries - 1) await sleep(delayMs);
      }
    }
    throw new Error(`httpGetWithRetry failed after ${retries} attempts`);
  }

  async function getTicker() {
    const url = `${astBase}/api/v3/ticker/price?symbol=${symbol}`;
    try {
      const r = await httpGetWithRetry(url, {
        headers: BINANCE_API_KEY ? { 'X-MBX-APIKEY': BINANCE_API_KEY } : {}
      });
      if (r.data && (r.data.price || r.data.price === 0)) return +r.data.price;
    } catch (e) {
      console.error('getTicker failed', e.message);
    }
    return null;
  }

  async function getTickerCached(maxAgeMs = 300) {
    const now = Date.now();
    if (lastTickerCache.price !== null && (now - lastTickerCache.ts) <= maxAgeMs) {
      return lastTickerCache.price;
    }
    const price = await getTicker();
    if (price !== null) {
      lastTickerCache = { ts: now, price };
    }
    return price;
  }

  async function getRecentKlines(limit, interval = '1m') {
    try {
      const url = `${astBase}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
      const r = await httpGetWithRetry(url, {
        headers: BINANCE_API_KEY ? { 'X-MBX-APIKEY': BINANCE_API_KEY } : {}
      });
      return r && r.data ? r.data : null;
    } catch (e) {
      console.error('getRecentKlines failed', e.message);
      return null;
    }
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

  async function gracefulShutdown(signal) {
    if (shutdownRequested) return;
    shutdownRequested = true;
    console.log('shutdown signal:', signal);

    try {
      if (flushStateTimer) clearInterval(flushStateTimer);
      if (pruneStateTimer) clearInterval(pruneStateTimer);
      await flushRuntimeState();
    } catch (e) {
      console.error('shutdown flush err', e.message);
    }

    process.exit(0);
  }

  process.on('SIGINT', () => {
    gracefulShutdown('SIGINT').catch(e => {
      console.error('SIGINT shutdown err', e.message);
      process.exit(1);
    });
  });

  process.on('SIGTERM', () => {
    gracefulShutdown('SIGTERM').catch(e => {
      console.error('SIGTERM shutdown err', e.message);
      process.exit(1);
    });
  });

  while (!shutdownRequested) {
    const SMA_WINDOW = parseInt(process.env.SMA_WINDOW || skillCfg.SMA_WINDOW || 60, 10);
    const limit = SMA_WINDOW + 2;
    const klines = await getRecentKlines(limit);

    let momentum_pct = 0;
    let sma = null;
    let smaPrev = null;
    let atr_pct = 0;
    let candle = null;
    let k_tp = parseFloat(process.env.K_TP || skillCfg.K_TP || '2.0');
    let k_sl = parseFloat(process.env.K_SL || skillCfg.K_SL || '1.2');
    let _k_tp_src = process.env.K_TP ? 'env' : (skillCfg.K_TP ? 'config' : 'default');
    let _k_sl_src = process.env.K_SL ? 'env' : (skillCfg.K_SL ? 'config' : 'default');

    if (Array.isArray(klines) && klines.length >= 3) {
      const lastClosed = klines[klines.length - 2];
      candle = {
        ts: lastClosed[0],
        open: +lastClosed[1],
        high: +lastClosed[2],
        low: +lastClosed[3],
        close: +lastClosed[4]
      };

      const closes = klines.map(c => +c[4]);
      const last = closes[closes.length - 2];
      const prev = closes[closes.length - 3] || last;
      momentum_pct = prev ? (last - prev) / prev : 0;

      if (closes.length >= SMA_WINDOW + 1) {
        const lastWindow = closes.slice(-SMA_WINDOW - 1, -1);
        sma = lastWindow.reduce((a, b) => a + b, 0) / lastWindow.length;

        const prevWindow = closes.slice(-SMA_WINDOW - 2, -2);
        smaPrev = prevWindow.length
          ? prevWindow.reduce((a, b) => a + b, 0) / prevWindow.length
          : null;
      } else {
        sma = closes.reduce((a, b) => a + b, 0) / closes.length;
      }

      const trs = [];
      for (let i = 1; i < klines.length; i++) {
        const high = +klines[i][2];
        const low = +klines[i][3];
        const prevClose = +klines[i - 1][4];
        trs.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
      }
      const atr = trs.length ? trs.reduce((a, b) => a + b, 0) / trs.length : 0;
      atr_pct = atr / (last || 1);
    }

    const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || skillCfg.MIN_MOMENTUM_PCT || 0.005);
    const MOM_REDUCTION_PCT = parseFloat(
      process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN ||
      skillCfg.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN ||
      0.0
    );
    const GREEN_KLINES = parseInt(
      process.env.GREEN_KLINES_FOR_REDUCTION ||
      skillCfg.GREEN_KLINES_FOR_REDUCTION ||
      0,
      10
    );
    const smaSlope = (sma !== null && smaPrev !== null) ? (sma - smaPrev) : 0;
    const smaTol = parseFloat(process.env.SMA_TOLERANCE || skillCfg.SMA_TOLERANCE || 0.001);
    const priceNearSMA = (sma !== null && candle) ? (candle.close >= sma * (1 - smaTol)) : true;
    const trendUp = smaSlope > 0;

    let green_run = 0;
    if (GREEN_KLINES > 0 && Array.isArray(klines)) {
      for (let j = klines.length - 2; j > 0 && green_run < GREEN_KLINES; j--) {
        const cur = +klines[j][4];
        const op = +klines[j][1];
        if (cur > op) green_run++;
        else break;
      }
    }

    let effectiveMinMom = BASE_MIN_MOM;
    if (green_run >= GREEN_KLINES && GREEN_KLINES > 0 && MOM_REDUCTION_PCT > 0) {
      effectiveMinMom = BASE_MIN_MOM * (1 - MOM_REDUCTION_PCT);
    }

    const momentumOk = momentum_pct >= effectiveMinMom;
    const shouldEnter = momentumOk && (trendUp || priceNearSMA);

    lastDecision = {
      momentum_pct: Number(momentum_pct.toFixed(6)),
      sma: sma != null ? Number(sma.toFixed(2)) : null,
      smaSlope: Number(smaSlope.toFixed(6)),
      priceNearSMA: !!priceNearSMA,
      trendUp: !!trendUp,
      shouldEnter: !!shouldEnter,
      effective_min_momentum: Number(effectiveMinMom.toFixed(6)),
      green_run
    };

    const entryCandidate = candle ? candle.close : null;
    const DUP_TOLERANCE_PCT = parseFloat(process.env.DUP_TOLERANCE_PCT || skillCfg.DUP_TOLERANCE_PCT || 1e-6);
    const CANDLE_MS = parseInt(process.env.CANDLE_MS || skillCfg.CANDLE_MS || 60000, 10);
    const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || skillCfg.SYMBOL_COOLDOWN_SECONDS || 0, 10);

    const alreadySimilar = entryCandidate !== null && openTrades.some(ot => {
      if (!isValidNumber(ot.entryPrice)) return false;
      const similarPrice = Math.abs(ot.entryPrice - entryCandidate) <= Math.abs(entryCandidate) * DUP_TOLERANCE_PCT;
      if (!similarPrice) return false;
      const openedTs = new Date(ot.openedAt).getTime();
      return (Date.now() - openedTs) <= CANDLE_MS;
    });

    const lastOpenTs = lastOpenBySymbol[symbol] || 0;
    const nowTs = Date.now();
    const withinCooldown = (cooldown_s > 0) && ((nowTs - lastOpenTs) < cooldown_s * 1000);
    const candleMinute = candle ? Math.floor(candle.ts / 60000) : null;
    const alreadyOpenedThisCandle = candleMinute !== null && lastTradeCandle === candleMinute;

    if (
      candle &&
      shouldEnter &&
      openTrades.length < maxPositions &&
      !alreadySimilar &&
      !withinCooldown &&
      !alreadyOpenedThisCandle
    ) {
      const entryPrice = candle.close;
      const effectiveATR = atr_pct;
      let stopLoss = entryPrice * (1 - k_sl * effectiveATR);
      const takeProfit = entryPrice * (1 + k_tp * effectiveATR);
      const MIN_SL_USD = parseFloat(process.env.MIN_SL_USD || skillCfg.MIN_SL_USD || 50);

      const stopDistance = entryPrice - stopLoss;
      if (stopDistance < MIN_SL_USD) stopLoss = entryPrice - MIN_SL_USD;

      const riskUSD = skillCapital * riskPct;
      const stopDistanceUSD = entryPrice - stopLoss;

      if (stopDistanceUSD <= 0) {
        console.error('invalid stop distance, skipping trade', { entryPrice, stopLoss });
      } else if (candle.low <= stopLoss) {
        console.log('trade skipped: SL inside candle (would have triggered before entry)', {
          entryPrice,
          stopLoss,
          candleLow: candle.low
        });
      } else {
        const qty = Number((riskUSD / stopDistanceUSD).toFixed(8));
        const reasonDet = {
          momentum_pct: Number(momentum_pct.toFixed(6)),
          sma: sma != null ? Number(sma.toFixed(2)) : null,
          atr_pct: Number(atr_pct.toFixed(6)),
          base_min_momentum: BASE_MIN_MOM,
          effective_minimum: Number(effectiveMinMom.toFixed(6)),
          green_run_len: green_run,
          reduction_pct: MOM_REDUCTION_PCT
        };
        const reasonTag = (green_run >= GREEN_KLINES && MOM_REDUCTION_PCT > 0)
          ? 'momentum_with_green_run'
          : 'momentum_standard';

        if (!isValidNumber(entryPrice) || !isValidNumber(stopLoss) || !isValidNumber(takeProfit) || !isValidNumber(qty)) {
          console.error('invalid trade values', { entryPrice, stopLoss, takeProfit, qty });
        } else {
          const trade = {
            id: Date.now(),
            symbol: 'BTC',
            entryPrice,
            stopLoss,
            takeProfit,
            openedAt: new Date().toISOString(),
            size: qty,
            exposureUSD: Number((entryPrice * qty).toFixed(2)),
            type: 'LONG',
            reasonTag,
            reasonDetails: reasonDet
          };

          if (existsEquivalentOpenTrade(openTrades, trade)) {
            console.log('skip duplicated logical trade', trade.entryPrice, trade.openedAt);
          } else {
            openTrades.push(trade);
            openTradesById.set(trade.id, trade);
            lastTradeCandle = candleMinute;
            lastOpenBySymbol[symbol] = Date.now();

            console.log(
              'OPEN (paper):',
              trade.openedAt,
              trade.entryPrice,
              'SL',
              trade.stopLoss,
              'TP',
              trade.takeProfit,
              'REASON',
              reasonTag,
              reasonDet
            );

            writeTradeCsv({ ...trade, action: 'open' });
            await persistOpenTrade(trade);
          }
        }
      }
    }

    const monitorStart = Date.now();
    while (!shutdownRequested && (Date.now() - monitorStart < 60 * 1000)) {
      try {
        const market = await getTickerCached(300);
        if (market !== null) {
          for (let i = openTrades.length - 1; i >= 0; i--) {
            const t = openTrades[i];
            const age = (Date.now() - new Date(t.openedAt).getTime()) / 1000;
            if (age < minHoldS) continue;

            let didClose = false;

            if (age > timeStopMinutes * 60) {
              didClose = await closeTrade(t, market, 'time_stop');
            } else if (t.stopLoss && market <= t.stopLoss) {
              didClose = await closeTrade(t, market, 'SL');
            } else if (t.takeProfit && market >= t.takeProfit) {
              didClose = await closeTrade(t, market, 'TP');
            }

            if (didClose) {
              openTradesById.delete(t.id);
              openTrades.splice(i, 1);
            }
          }
        }
      } catch (e) {
        console.error('monitor err', e.message);
      }

      await sleep(monitorInterval > 0 ? monitorInterval : 500);
    }

    const tsHuman = fmtDateUtc1(new Date());
    console.log(
      'ITER_SUMMARY:',
      tsHuman,
      'momentum=' + (lastDecision.momentum_pct || 0),
      'sma=' + (lastDecision.sma || 'null'),
      'smaSlope=' + (lastDecision.smaSlope || 0),
      'priceNearSMA=' + (lastDecision.priceNearSMA ? 1 : 0),
      'trendUp=' + (lastDecision.trendUp ? 1 : 0),
      'shouldEnter=' + (lastDecision.shouldEnter ? 1 : 0),
      'effective_min_momentum=' + (lastDecision.effective_min_momentum || 0),
      'green_run=' + (lastDecision.green_run || 0),
      'openTrades=' + openTrades.length,
      'csv=' + CSV_PATH
    );

    console.log('TRACE_DETAILS_JSON:', JSON.stringify({
      params: { k_tp, k_sl, k_tp_src: _k_tp_src, k_sl_src: _k_sl_src },
      decision: lastDecision,
      openTradesCount: openTrades.length
    }));

    if (openTrades.length > 0) {
      for (const ot of openTrades) {
        console.log('OPEN_TRADE:', JSON.stringify(ot));
      }
    }
  }
})();
