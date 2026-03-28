#!/usr/bin/env node

/**
 * trading-bot.live.service.sa.cjs
 *
 * SA = Stabilized & Adaptive.
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

// Local SA helpers (refactor: no logic changes)
const { ensureDir, readJsonFile } = require('./bot/sa/file_io.cjs');
const { sleep, isValidNumber, percentile } = require('./bot/sa/util.cjs');
const { getYmd, fmtDateUtc1, nowIso, makeGetJournalPath } = require('./bot/sa/time.cjs');
const { createFileLock } = require('./bot/sa/lock.cjs');
const { createSignalKeys } = require('./bot/sa/signal_keys.cjs');
const { createJsonIo } = require('./bot/sa/json_io.cjs');
const { createJournalStateManager } = require('./bot/sa/journal_state.cjs');
const { createRateLimiter } = require('./bot/sa/rate_limit.cjs');
const { createHttpRetry } = require('./bot/sa/http_retry.cjs');
const { createMarketData } = require('./bot/sa/market_data.cjs');
const { createNormalizers } = require('./bot/sa/normalizers.cjs');
const { buildCloseReason } = require('./bot/sa/trade_helpers.cjs');

// Exchange helpers (spot v3 signed endpoints)
const { createMexcSpotClient } = require('./exchange/mexc_spot_client.cjs');

// Risk helpers
const { evalSlPolicy } = require('./bot/sl_policy.cjs');

// LIVE execution engine
const { createLiveExecutorMexc } = require('./bot/live_executor_mexc.cjs');

// Impulse bypass helper
const { evaluateImpulseBypass } = require('./bot/sa/impulse_bypass.cjs');

// Market bias (logging only)
const { getMarketBias } = require('./bot/sa/market_bias.cjs');

const {
  getClosedCloses,
  computeSmaPair,
  computeAtr,
  computeRealizedVol,
  detectMarketRegime,
  computeDonchian,
  computeAdx,
  computeSupertrend
} = require('./market-regime.sa.cjs');

/* -----------------------------
 * PATHS & CONFIG
 * ----------------------------- */

const LABEL = process.env.LABEL || 'V4.2';
const BASE_DIR = path.join(__dirname, 'skills', `live-forward-${LABEL.toLowerCase()}`);
const CONFIG_PATH = path.join(BASE_DIR, 'config.json');
const LOCK_PATH = path.join(BASE_DIR, 'bot.lock');
const OPEN_POSITIONS_PATH = path.join(BASE_DIR, 'open_positions.json');


function loadSkillConfig() {
  return readJsonFile(CONFIG_PATH, {});
}

const cfg = loadSkillConfig();
const EXCHANGE = process.env.EXCHANGE || cfg.EXCHANGE || 'binance';

/* -----------------------------
 * STATE
 * ----------------------------- */

let shutdownRequested = false;
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
      lastAtrCandleTs: null,

      // IMPULSE3 bypass explosive
      impulseCount: 0
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

// Lazy-init MEXC client (only used in runMode=live paths)
let _mexcClient = null;
function getMexcClient() {
  if (_mexcClient) return _mexcClient;
  if (EXCHANGE !== 'mexc') throw new Error(`getMexcClient() called but EXCHANGE=${EXCHANGE}`);
  if (!ACTIVE_API_KEY || !ACTIVE_API_SECRET) throw new Error('MEXC API keys not set (MEXC_API_KEY/MEXC_API_SECRET)');
  _mexcClient = createMexcSpotClient({
    baseUrl: getApiBase(),
    apiKey: ACTIVE_API_KEY,
    apiSecret: ACTIVE_API_SECRET,
    timeoutMs: 2500,
  });
  return _mexcClient;
}

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

// Exchange clients live in ./exchange/* and are initialized lazily via getMexcClient().
// Generic helpers below are exchange-agnostic.

const { writeJsonFileAtomic, appendJsonl, loadJsonl } = createJsonIo({ ensureDir });

const getJournalPath = makeGetJournalPath(BASE_DIR);

/* -----------------------------
 * LOCK (robust)
 * ----------------------------- */

const { acquireLock, releaseLock } = createFileLock({
  ensureDir,
  baseDir: BASE_DIR,
  lockPath: LOCK_PATH,
  nowIso,
});

/* -----------------------------
 * SIGNAL KEYS (compat)
 * ----------------------------- */

const {
  getSignalBucketMs,
  getSignalKey,
  getOpenEventKey,
  getCloseEventKey,
} = createSignalKeys({ label: LABEL });

/* -----------------------------
 * NORMALIZERS (compat: additive fields ok)
 * ----------------------------- */

const { normalizeOpenTrade, normalizeCloseTrade } = createNormalizers({
  label: LABEL,
  nowIso,
  getSignalKey,
});

/* -----------------------------
 * TRADE-RATE LIMITING
 * ----------------------------- */

const {
  pruneRecentOpens,
  noteOpenTimestamp,
  countOpensLastHour,
  countOpensTodayUtc,
} = createRateLimiter({ state, getYmd });

/* -----------------------------
 * JOURNAL STATE MANAGEMENT
 * ----------------------------- */

const {
  applyJournalEvent,
  persistJournalEvent,
  rebuildStateFromJournal,
  computeSnapshotHash,
  flushOpenPositionsSnapshot,
  startSnapshotTimer,
  stopSnapshotTimer,
  getOpenTradesArray,
  cleanupRecentSignals,
  hasOpenSignal,
  canOpenSignal,
} = createJournalStateManager({
  state,
  label: LABEL,
  getJournalPath,
  nowIso,
  loadJsonl,
  appendJsonl,
  writeJsonFileAtomic,
  openPositionsPath: OPEN_POSITIONS_PATH,
  crypto,
  noteOpenTimestamp,
  getYmd,
});

/* -----------------------------
 * HTTP (robust retry + backoff)
 * ----------------------------- */

const { httpGetWithRetry } = createHttpRetry({ axios, sleep });

const {
  getTicker,
  getTickerCached,
  getRecentKlines,
  rankSymbolsByRecentReturn,
} = createMarketData({
  getApiBase,
  httpGetWithRetry,
  activeApiKey: ACTIVE_API_KEY,
  exchange: EXCHANGE,
  state,
});

/* -----------------------------
 * TRADING LOGIC (paper)
 * ----------------------------- */

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
    `OPEN (${trade.mode || 'paper'}):`, trade.openedAt,
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
  const realizedGross = Number(trade.realizedGross || 0);
  const realizedFeeUsdEst = Number(trade.realizedFeeUsdEst || 0);

  trade.exitPrice = market;
  trade.profit = realizedGross + (market - trade.entryPrice) * qty;
  trade.profit_pct = trade.entryPrice > 0 ? (market - trade.entryPrice) / trade.entryPrice : null;
  trade.closedAt = nowIso();

  // Fee estimation: allow per-side fee rates when available (maker/taker).
  // Default model (MEXC): maker 0%, taker FEE_RATE_TAKER (0.0005) applied **solo lado taker**.
  const feeRateEntry = (trade.feeRateEntry != null) ? Number(trade.feeRateEntry) : 0;
  const feeRateExit = (trade.feeRateExit != null) ? Number(trade.feeRateExit) : (feeRate || 0);
  const feeUsdEst = realizedFeeUsdEst + (trade.entryPrice * qty) * feeRateEntry + (market * qty) * feeRateExit;
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
    stopSnapshotTimer();
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
  const NO_SL_BEFORE_MINUTES = parseInt(process.env.NO_SL_BEFORE_MINUTES || cfgLive.NO_SL_BEFORE_MINUTES || 0, 10);

  const HTTP_TIMEOUT_MS = parseInt(process.env.HTTP_TIMEOUT_MS || cfgLive.HTTP_TIMEOUT_MS || 2500, 10);
  const candleBucketMs = parseInt(process.env.SIGNAL_BUCKET_MS || cfgLive.SIGNAL_BUCKET_MS || cfgLive.CANDLE_MS || 60000, 10);
  const signalCooldownMs = parseInt(process.env.SIGNAL_COOLDOWN_MS || cfgLive.SIGNAL_COOLDOWN_MS || 180000, 10);

  // --- Trading mode / risk policies (paper vs live) ---
  const TRADING_MODE = String(process.env.TRADING_MODE || cfgLive.TRADING_MODE || 'PAPER').toUpperCase();
  const ENABLE_LIVE = String(process.env.ENABLE_LIVE || cfgLive.ENABLE_LIVE || '0') === '1';
  const runMode = (TRADING_MODE === 'LIVE' && ENABLE_LIVE) ? 'live' : 'paper';

  // SL policies:
  // - CLASSIC: hard close when market <= SL classic
  // - EMERGENCY_ONLY: only emergency SL (wide) + time_stop
  // - SOFT_CLASSIC_WITH_EMERGENCY: emergency as airbag + classic SL with confirmation window
  const SL_POLICY = String(process.env.SL_POLICY || cfgLive.SL_POLICY || 'CLASSIC').toUpperCase();
  const EMERGENCY_SL_PCT = parseFloat(process.env.EMERGENCY_SL_PCT || cfgLive.EMERGENCY_SL_PCT || 0);
  const SL_CONFIRM_SECONDS = parseInt(process.env.SL_CONFIRM_SECONDS || cfgLive.SL_CONFIRM_SECONDS || 0, 10);
  const HALT_TRADING_ON_EMERGENCY = (process.env.HALT_TRADING_ON_EMERGENCY != null)
    ? String(process.env.HALT_TRADING_ON_EMERGENCY) === '1'
    : !!cfgLive.HALT_TRADING_ON_EMERGENCY;

  // LIVE guardrails
  if (TRADING_MODE === 'LIVE' && !ENABLE_LIVE) {
    console.error('TRADING_MODE=LIVE requires ENABLE_LIVE=1 (env)');
    process.exit(1);
  }
  if (runMode === 'live' && (!ACTIVE_API_KEY || !ACTIVE_API_SECRET)) {
    console.error(`API keys for ${EXCHANGE} not set in .env (runMode=live)`);
    process.exit(1);
  }

  const explosiveCandlePct = parseFloat(process.env.EXPLOSIVE_CANDLE_PCT || cfgLive.EXPLOSIVE_CANDLE_PCT || 0.003);
  const tradeUSD = parseFloat(process.env.TRADE_USD || cfgLive.TRADE_USD || 100.0);

  // Execution tuning (LIVE)
  const TP_ON_EXCHANGE = (process.env.TP_ON_EXCHANGE != null)
    ? String(process.env.TP_ON_EXCHANGE) === '1'
    : (cfgLive.TP_ON_EXCHANGE !== undefined ? !!cfgLive.TP_ON_EXCHANGE : true);
  const MAKER_ENTRY_ONLY = (process.env.MAKER_ENTRY_ONLY != null)
    ? String(process.env.MAKER_ENTRY_ONLY) === '1'
    : !!cfgLive.MAKER_ENTRY_ONLY;
  const MAKER_ENTRY_TIMEOUT_MS = parseInt(process.env.MAKER_ENTRY_TIMEOUT_MS || cfgLive.MAKER_ENTRY_TIMEOUT_MS || 15000, 10);
  const MAKER_ENTRY_MAX_ATTEMPTS = parseInt(process.env.MAKER_ENTRY_MAX_ATTEMPTS || cfgLive.MAKER_ENTRY_MAX_ATTEMPTS || 3, 10);
  const MAKER_ENTRY_RETRY_SLEEP_MS = parseInt(process.env.MAKER_ENTRY_RETRY_SLEEP_MS || cfgLive.MAKER_ENTRY_RETRY_SLEEP_MS || 750, 10);
  const MAKER_ENTRY_PRICE_OFFSET_PCT = parseFloat(process.env.MAKER_ENTRY_PRICE_OFFSET_PCT || cfgLive.MAKER_ENTRY_PRICE_OFFSET_PCT || 0);

  // Close-maker tuning (optional; defaults chosen for safety)
  const MAKER_CLOSE_PRICE_OFFSET_PCT = parseFloat(process.env.MAKER_CLOSE_PRICE_OFFSET_PCT || cfgLive.MAKER_CLOSE_PRICE_OFFSET_PCT || 0.0001);
  const MAKER_CLOSE_MAX_ATTEMPTS = parseInt(process.env.MAKER_CLOSE_MAX_ATTEMPTS || cfgLive.MAKER_CLOSE_MAX_ATTEMPTS || 3, 10);
  const MAKER_CLOSE_RETRY_SLEEP_MS = parseInt(process.env.MAKER_CLOSE_RETRY_SLEEP_MS || cfgLive.MAKER_CLOSE_RETRY_SLEEP_MS || 500, 10);
  const MAKER_CLOSE_TIMEOUT_MS = parseInt(process.env.MAKER_CLOSE_TIMEOUT_MS || cfgLive.MAKER_CLOSE_TIMEOUT_MS || 8000, 10);

  const ORDER_POLL_MS = parseInt(process.env.ORDER_POLL_MS || cfgLive.ORDER_POLL_MS || 500, 10);

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

  // TP: single target only (no partial scale-out). Legacy SCALE_TP_* params are ignored.

  const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || cfgLive.MIN_MOMENTUM_PCT || 0.0005);
  const MAX_MOMENTUM_PCT = parseFloat(process.env.MAX_MOMENTUM_PCT || cfgLive.MAX_MOMENTUM_PCT || 0.0012);

  const MOM_REDUCTION_PCT = parseFloat(process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || cfgLive.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || 0.0);
  const GREEN_KLINES = parseInt(process.env.GREEN_KLINES_FOR_REDUCTION || cfgLive.GREEN_KLINES_FOR_REDUCTION || 0, 10);

  const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || cfgLive.SYMBOL_COOLDOWN_SECONDS || 0, 10);
  const feeRate = parseFloat(process.env.FEE_RATE || cfgLive.FEE_RATE || 0.0005);
  const feeRateMaker = parseFloat(process.env.FEE_RATE_MAKER || cfgLive.FEE_RATE_MAKER || 0.0);
  const feeRateTaker = parseFloat(process.env.FEE_RATE_TAKER || cfgLive.FEE_RATE_TAKER || feeRate);
  const ATR_WINDOW = parseInt(process.env.ATR_WINDOW || cfgLive.ATR_WINDOW || 14, 10);
  const MIN_ATR_PCT = parseFloat(process.env.MIN_ATR_PCT || cfgLive.MIN_ATR_PCT || 0);
  // New: max ATR% cap (entry only). 0 disables.
  const MAX_ATR_PCT = parseFloat(process.env.MAX_ATR_PCT || cfgLive.MAX_ATR_PCT || 0);
  // Defensive (entry): hard skip entries when ATR% is below this threshold (0 disables).
  const MIN_ATR_PCT_HARD = parseFloat(process.env.MIN_ATR_PCT_HARD || cfgLive.MIN_ATR_PCT_HARD || 0);
  // Defensive (monitor): in HIGH_VOL+CHOPPY, avoid wick-triggered emergency exits; let soft SL policy handle.
  const DISABLE_EMERGENCY_WICK_IN_HV_CHOPPY = (process.env.DISABLE_EMERGENCY_WICK_IN_HV_CHOPPY != null)
    ? String(process.env.DISABLE_EMERGENCY_WICK_IN_HV_CHOPPY) === '1'
    : !!cfgLive.DISABLE_EMERGENCY_WICK_IN_HV_CHOPPY;

  // New: disable wick-based emergency SL always (all regimes). Leaves classic soft-confirm SL policy.
  const DISABLE_EMERGENCY_WICK_ALWAYS = (process.env.DISABLE_EMERGENCY_WICK_ALWAYS != null)
    ? String(process.env.DISABLE_EMERGENCY_WICK_ALWAYS) === '1'
    : !!cfgLive.DISABLE_EMERGENCY_WICK_ALWAYS;

  // Extra indicator configs (Phase B)
  const USE_ADX_FILTER = (process.env.USE_ADX_FILTER != null)
    ? String(process.env.USE_ADX_FILTER) === '1'
    : !!cfgLive.USE_ADX_FILTER;
  const ADX_WINDOW = parseInt(process.env.ADX_WINDOW || cfgLive.ADX_WINDOW || 14, 10);
  const MIN_ADX = parseFloat(process.env.MIN_ADX || cfgLive.MIN_ADX || 0);
  const ADX_REQUIRE_DI_BULL = (process.env.ADX_REQUIRE_DI_BULL != null)
    ? String(process.env.ADX_REQUIRE_DI_BULL) === '1'
    : (cfgLive.ADX_REQUIRE_DI_BULL !== undefined ? !!cfgLive.ADX_REQUIRE_DI_BULL : true);

  const USE_DONCHIAN_FILTER = (process.env.USE_DONCHIAN_FILTER != null)
    ? String(process.env.USE_DONCHIAN_FILTER) === '1'
    : !!cfgLive.USE_DONCHIAN_FILTER;
  const DONCHIAN_N = parseInt(cfgLive.DONCHIAN_N || 20, 10);

  const USE_SUPERTREND_FILTER = (process.env.USE_SUPERTREND_FILTER != null)
    ? String(process.env.USE_SUPERTREND_FILTER) === '1'
    : !!cfgLive.USE_SUPERTREND_FILTER;
  const SUPERTREND_ATR_WINDOW = parseInt(process.env.SUPERTREND_ATR_WINDOW || cfgLive.SUPERTREND_ATR_WINDOW || 10, 10);
  const SUPERTREND_MULT = parseFloat(process.env.SUPERTREND_MULT || cfgLive.SUPERTREND_MULT || 3.0);

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

  // Baseline-parity: hard skip LOW_VOL.
  // Backward compat: if SKIP_LOW_VOL_HARD is not provided, fall back to legacy SKIP_LOW_VOL.
  const SKIP_LOW_VOL_HARD = (process.env.SKIP_LOW_VOL_HARD != null)
    ? String(process.env.SKIP_LOW_VOL_HARD) === '1'
    : (cfgLive.SKIP_LOW_VOL_HARD !== undefined ? !!cfgLive.SKIP_LOW_VOL_HARD : !!cfgLive.SKIP_LOW_VOL);

  const SKIP_CHOPPY = (process.env.SKIP_CHOPPY != null)
    ? String(process.env.SKIP_CHOPPY) === '1'
    : !!cfgLive.SKIP_CHOPPY;
  const SKIP_CHOPPY_ONLY_IF_LOW_VOL = (process.env.SKIP_CHOPPY_ONLY_IF_LOW_VOL != null)
    ? String(process.env.SKIP_CHOPPY_ONLY_IF_LOW_VOL) === '1'
    : !!cfgLive.SKIP_CHOPPY_ONLY_IF_LOW_VOL;

  // Baseline-parity: skip CHOPPY only when HIGH_VOL+CHOPPY (defensive).
  const SKIP_CHOPPY_IN_HIGH_VOL = (process.env.SKIP_CHOPPY_IN_HIGH_VOL != null)
    ? String(process.env.SKIP_CHOPPY_IN_HIGH_VOL) === '1'
    : !!cfgLive.SKIP_CHOPPY_IN_HIGH_VOL;

  // New (config-clean): single knob for choppy skipping.
  // Values: off | low_vol | high_vol | extremes | always
  const CHOPPY_SKIP_MODE = String(process.env.CHOPPY_SKIP_MODE || cfgLive.CHOPPY_SKIP_MODE || '').toLowerCase() || null;

  // Baseline-parity: trend gate mode.
  // - strict: require trendUp as computed.
  // - no_trend: bypass trend gate (trendUp := true)
  const TREND_GATE_MODE = String(process.env.TREND_GATE_MODE || cfgLive.TREND_GATE_MODE || 'strict');

  // Optional per-symbol overrides (multi-coin friendly)
  //   MIN_SLOPE_NORM_BY_SYMBOL: { "BTCUSDT": 0.10, "XRPUSDT": 0.06 }
  //   MIN_SMA_SLOPE_PCT_BY_SYMBOL: { "BTCUSDT": 0.00012, "XRPUSDT": 0.00020 }
  const MIN_SLOPE_NORM_BY_SYMBOL = {};
  const MIN_SMA_SLOPE_PCT_BY_SYMBOL = {};

  try {
    const raw = cfgLive.MIN_SLOPE_NORM_BY_SYMBOL;
    if (raw && typeof raw === 'object') {
      for (const [k, v] of Object.entries(raw)) {
        const sym = String(k || '').toUpperCase();
        const num = parseFloat(v);
        if (sym && Number.isFinite(num)) MIN_SLOPE_NORM_BY_SYMBOL[sym] = num;
      }
    }
  } catch (_) {}

  try {
    const raw = cfgLive.MIN_SMA_SLOPE_PCT_BY_SYMBOL;
    if (raw && typeof raw === 'object') {
      for (const [k, v] of Object.entries(raw)) {
        const sym = String(k || '').toUpperCase();
        const num = parseFloat(v);
        if (sym && Number.isFinite(num)) MIN_SMA_SLOPE_PCT_BY_SYMBOL[sym] = num;
      }
    }
  } catch (_) {}

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
  // Safety: never rank if only 1 symbol.
  const RANKING_ENABLED = (symbols.length > 1) && (
    (cfgLive.SYMBOL_RANKING_ENABLED !== undefined) ? !!cfgLive.SYMBOL_RANKING_ENABLED : true
  );
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
  console.log('TRADING_MODE:', TRADING_MODE, '(runMode=', runMode + ')');
  console.log('SL_POLICY:', SL_POLICY, 'EMERGENCY_SL_PCT=', EMERGENCY_SL_PCT, 'SL_CONFIRM_SECONDS=', SL_CONFIRM_SECONDS);
  console.log('HALT_TRADING_ON_EMERGENCY:', HALT_TRADING_ON_EMERGENCY);

  rebuildStateFromJournal();

  // Initialize exchange client lazily (only if we are truly in LIVE mode)
  const mexc = (runMode === 'live') ? getMexcClient() : null;

  // --- LIVE execution engine (MEXC spot) ---
  const liveExec = (runMode === 'live')
    ? createLiveExecutorMexc({
        mexc,
        label: LABEL,
        symbols,
        candleBucketMs,
        httpTimeoutMs: HTTP_TIMEOUT_MS,
        makerEntryTimeoutMs: MAKER_ENTRY_TIMEOUT_MS,
        makerEntryMaxAttempts: MAKER_ENTRY_MAX_ATTEMPTS,
        makerEntryRetrySleepMs: MAKER_ENTRY_RETRY_SLEEP_MS,
        makerEntryPriceOffsetPct: MAKER_ENTRY_PRICE_OFFSET_PCT,
        makerCloseOffsetPct: MAKER_CLOSE_PRICE_OFFSET_PCT,
        makerCloseMaxAttempts: MAKER_CLOSE_MAX_ATTEMPTS,
        makerCloseRetrySleepMs: MAKER_CLOSE_RETRY_SLEEP_MS,
        makerCloseTimeoutMs: MAKER_CLOSE_TIMEOUT_MS,
        makerEntryOnly: MAKER_ENTRY_ONLY,
        orderPollMs: ORDER_POLL_MS,
        tpOnExchange: TP_ON_EXCHANGE,
        feeRateMaker,
        feeRateTaker,
        state,
        nowIso,
        getSignalKey,
        hasOpenSignal,
        canOpenSignal,
        getOpenEventKey,
        persistJournalEvent,
        normalizeOpenTrade,
        closeTrade,
        getTickerCached,
        verbose: VERBOSE,
        getOpenTradesArray,
      })
    : null;

  const openTradeLive = liveExec ? liveExec.openTradeLive : null;
  const closeTradeLiveMarket = liveExec ? liveExec.closeTradeLiveMarket : null;
  const closeTradeLiveMakerFirst = liveExec ? liveExec.closeTradeLiveMakerFirst : null;
  const reconcileLiveTpOrders = liveExec ? liveExec.reconcileLiveTpOrders : null;

  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

  let symbolsOrdered = symbols.slice();
  let lastRankTs = 0;

  // If emergency SL triggers and HALT_TRADING_ON_EMERGENCY=1, we stop opening new trades until UTC day rollover.
  let emergencyHaltYmd = null;

  // LIVE reconciliation cadence (best-effort)
  let lastLiveReconcileMs = 0;

  // Initial ranking (optional)
  if (RANKING_ENABLED && symbols.length > 1) {
    try {
      symbolsOrdered = await rankSymbolsByRecentReturn(symbols, RANK_LOOKBACK_DAYS, HTTP_TIMEOUT_MS, VERBOSE);
      lastRankTs = Date.now();
    } catch (e) {
      console.error('RANK init error:', e.message);
    }
  }

  // LIVE: best-effort reconciliation (attach TP orders / emit closes if TP filled while down)
  if (runMode === 'live') {
    try {
      await reconcileLiveTpOrders();
    } catch (e) {
      console.error('LIVE reconcile fatal (ignored):', e.message);
    }
  }

  while (!shutdownRequested) {
    iterCount++;
    const loopStart = Date.now();
    const tsStartIter = fmtDateUtc1(new Date());

    try {
      cleanupRecentSignals();

      // LIVE: periodic reconciliation (attach TP orders / emit closes if TP filled while down)
      if (runMode === 'live' && TP_ON_EXCHANGE) {
        const nowMs = Date.now();
        if (!lastLiveReconcileMs || (nowMs - lastLiveReconcileMs) >= 30_000) {
          try {
            await reconcileLiveTpOrders();
          } catch (e) {
            console.error('LIVE reconcile tick error (ignored):', e.message);
          }
          lastLiveReconcileMs = nowMs;
        }
      }

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


          // LIVE: if TP order exists on exchange, check fill status first (even before minHold).
          if (runMode === 'live' && TP_ON_EXCHANGE && tr.tpOrderId) {
            try {
              if (!mexc) throw new Error('mexc client not initialized');
              const tpOrd = await mexc.getOrder({ symbol: sym, orderId: tr.tpOrderId }, HTTP_TIMEOUT_MS);
              if (mexc.isOrderFilled(tpOrd)) {
                const exitP = mexc.orderAvgFillPrice(tpOrd) ?? market;
                tr.mode = 'live';
                tr.executionExit = 'maker';
                tr.feeRateExit = feeRateMaker;
                await closeTrade(tr, exitP, 'TP', candleBucketMs, feeRateMaker);
                continue;
              }
            } catch (e) {
              if (VERBOSE) console.error('LIVE TP status check failed:', e.message);
            }
          }

          // TP: single target only (no partial scale-out).

          if (ageS > timeStopMinutes * 60) {
            if (runMode === 'live') {
              if (closeTradeLiveMakerFirst) await closeTradeLiveMakerFirst(tr, 'time_stop', { makerTimeoutMs: MAKER_CLOSE_TIMEOUT_MS });
              else await closeTradeLiveMarket(tr, 'time_stop');
            } else {
              await closeTrade(tr, market, 'time_stop', candleBucketMs, feeRate);
            }
            continue;
          }

          if (ageS < minHoldS) continue;


          const slClassic = (tr.stopLossClassic != null) ? tr.stopLossClassic : null;
          let slEmergency = (tr.stopLossEmergency != null)
            ? tr.stopLossEmergency
            : (tr.stopLoss != null ? tr.stopLoss : null);

          // Defensive: optionally disable wick-driven emergency exits.
          if (DISABLE_EMERGENCY_WICK_ALWAYS) {
            slEmergency = null;
          }

          // Defensive: in HIGH_VOL+CHOPPY, avoid wick-driven emergency exits (let classic soft confirm handle)
          if (!DISABLE_EMERGENCY_WICK_ALWAYS && DISABLE_EMERGENCY_WICK_IN_HV_CHOPPY) {
            try {
              const rt = getSymbolRuntime(sym);
              const reg = rt && rt.lastRegimeInfo;
              if (reg && reg.volRegime === 'HIGH_VOL' && reg.microRegime === 'CHOPPY') {
                slEmergency = null;
              }
            } catch (_) {}
          }

          // SL policy evaluation (classic vs soft+emergency)
          const slRes = evalSlPolicy({
            slPolicy: SL_POLICY,
            market,
            slClassic,
            slEmergency,
            nowMs: Date.now(),
            breachStartMs: tr._slBreachStartMs,
            slConfirmSeconds: SL_CONFIRM_SECONDS,
          });

          tr._slBreachStartMs = slRes.breachStartMs;

          if (slRes.closeReason) {
            // Optional: ignore classic SL during the first N minutes to avoid early shakeouts.
            // Safety: we only block the classic 'SL' closeReason; emergency sl remains active.
            if (slRes.closeReason === 'SL' && Number.isFinite(NO_SL_BEFORE_MINUTES) && NO_SL_BEFORE_MINUTES > 0) {
              const ageMin = ageS / 60;
              if (ageMin < NO_SL_BEFORE_MINUTES) {
                // Do not close; keep monitoring (TP/time_stop may still close).
                // Note: breachStartMs remains tracked by evalSlPolicy state.
              } else {
                if (runMode === 'live') {
                  if (closeTradeLiveMakerFirst) await closeTradeLiveMakerFirst(tr, slRes.closeReason, { makerTimeoutMs: 5000 });
                  else await closeTradeLiveMarket(tr, slRes.closeReason);
                }
                else await closeTrade(tr, market, slRes.closeReason, candleBucketMs, feeRate);
                if (slRes.emergency && HALT_TRADING_ON_EMERGENCY) emergencyHaltYmd = getYmd(Date.now());
                continue;
              }
            } else {
              if (runMode === 'live') {
                if (closeTradeLiveMakerFirst) await closeTradeLiveMakerFirst(tr, slRes.closeReason, { makerTimeoutMs: 5000 });
                else await closeTradeLiveMarket(tr, slRes.closeReason);
              }
              else await closeTrade(tr, market, slRes.closeReason, candleBucketMs, feeRate);
              if (slRes.emergency && HALT_TRADING_ON_EMERGENCY) emergencyHaltYmd = getYmd(Date.now());
              continue;
            }
          }

          // 3) Take profit fallback (only if no TP on exchange)
          if (tr.takeProfit && market >= tr.takeProfit) {
            if (runMode === 'live' && TP_ON_EXCHANGE) {
              // If TP order wasn't placed/known, fall back to maker-first close with quick timeout.
              if (closeTradeLiveMakerFirst) await closeTradeLiveMakerFirst(tr, 'TP', { makerTimeoutMs: 5000 });
              else await closeTradeLiveMarket(tr, 'TP');
            } else {
              await closeTrade(tr, market, 'TP', candleBucketMs, feeRate);
            }
            continue;
          }
        }
      }

      // 2) Evaluate new candle entries per symbol (at most 1/min per symbol)
      for (const sym of symbolsOrdered) {
        const rt = getSymbolRuntime(sym);

        // Halt new entries for the rest of the UTC day after an emergency SL, if enabled.
        if (HALT_TRADING_ON_EMERGENCY && emergencyHaltYmd && emergencyHaltYmd === getYmd(Date.now())) {
          if (VERBOSE) console.log('HALT: emergency SL triggered today, skipping new entries', { iter: iterCount, symbol: sym, ymd: emergencyHaltYmd });
          continue;
        }

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
        const atrHardOk = (!Number.isFinite(MIN_ATR_PCT_HARD) || MIN_ATR_PCT_HARD <= 0)
          ? true
          : (atrPctNum >= MIN_ATR_PCT_HARD);
        const atrMaxOkBase = (!Number.isFinite(MAX_ATR_PCT) || MAX_ATR_PCT <= 0)
          ? true
          : (atrPctNum <= MAX_ATR_PCT);

        const realizedVol = computeRealizedVol(closes, Math.min(60, Math.max(20, Math.floor(SMA_WINDOW / 2))));

        // Extra indicators
        const adxInfo = computeAdx(klines, ADX_WINDOW);

        const donch = computeDonchian(klines, DONCHIAN_N);
        // Prior Donchian window (exclude the current closed candle) to avoid lookahead.
        let donchPrevHigh = null;
        let donchPrevLow = null;
        try {
          const closed = Array.isArray(klines) ? klines.slice(0, Math.max(0, klines.length - 1)) : [];
          if (closed.length >= DONCHIAN_N + 1) {
            const slice = closed.slice(-(DONCHIAN_N + 1), -1);
            let hi = -Infinity;
            let lo = Infinity;
            for (const k of slice) {
              const h = Number(k?.[2]);
              const l = Number(k?.[3]);
              if (!Number.isFinite(h) || !Number.isFinite(l)) { hi = -Infinity; lo = Infinity; break; }
              if (h > hi) hi = h;
              if (l < lo) lo = l;
            }
            if (hi !== -Infinity) donchPrevHigh = hi;
            if (lo !== Infinity) donchPrevLow = lo;
          }
        } catch (_) {}

        const st = computeSupertrend(klines, SUPERTREND_ATR_WINDOW, SUPERTREND_MULT);

        const regimeInfo = detectMarketRegime({
          atrPct: atrPctNum,
          realizedVol,
          smaSlope: smaSlopeAbs,
          lastClose: candle.close,
          priceAboveSma: priceAboveSMA
        });

        // Persist last regime snapshot for monitor-side defensive logic (best-effort)
        rt.lastRegimeInfo = {
          ts: candle.ts,
          volRegime: regimeInfo.volRegime,
          microRegime: regimeInfo.microRegime,
        };

        const minSlopeNormEff = Number.isFinite(MIN_SLOPE_NORM_BY_SYMBOL[sym])
          ? MIN_SLOPE_NORM_BY_SYMBOL[sym]
          : MIN_SLOPE_NORM;

        const slopeNormOk = (minSlopeNormEff > 0)
          ? (Number(regimeInfo.slopeNorm) >= minSlopeNormEff)
          : true;

        const regime = regimeInfo.volRegime;

        const dynamicMinMomentum = BASE_MIN_MOM * regimeInfo.kMinMomentum;
        const dynamicMinSlopeAbs = minSMASlope * regimeInfo.kMinSlope;

        // Per-symbol override for MIN_SMA_SLOPE_PCT (optional)
        const minSmaSlopePctEff = Number.isFinite(MIN_SMA_SLOPE_PCT_BY_SYMBOL[sym])
          ? MIN_SMA_SLOPE_PCT_BY_SYMBOL[sym]
          : minSMASlopePct;

        const dynamicMinSlopePct = Number.isFinite(minSmaSlopePctEff)
          ? (minSmaSlopePctEff * regimeInfo.kMinSlope)
          : null;

        const dynamicMaxMomentumBase = MAX_MOMENTUM_PCT;

        // If MIN_SMA_SLOPE_PCT is set, use pct-based slope threshold (works across assets).
        // Otherwise fall back to ABS (price units) legacy behavior.
        const trendUp = (dynamicMinSlopePct != null)
          ? (smaSlopePct > dynamicMinSlopePct)
          : (smaSlopeAbs > dynamicMinSlopeAbs);

        // Market bias (logging only)
        const marketBias = getMarketBias({
          regime,
          microRegime: regimeInfo.microRegime,
          slopeNorm: Number(regimeInfo.slopeNorm || 0),
          trendUp: !!trendUp,
          priceAboveSMA: !!priceAboveSMA,
          adx: Number(adxInfo?.adx || 0),
          diPlus: Number(adxInfo?.diPlus || 0),
          diMinus: Number(adxInfo?.diMinus || 0),
        });

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

        const momentumOkBase = momentum_pct >= effectiveMinMom && momentum_pct <= dynamicMaxMomentumBase;

        const adxOk = (!USE_ADX_FILTER || !(MIN_ADX > 0))
          ? true
          : (adxInfo.adx != null && Number.isFinite(adxInfo.adx) && adxInfo.adx >= MIN_ADX);

        const diBullOk = (!USE_ADX_FILTER || !ADX_REQUIRE_DI_BULL)
          ? true
          : (adxInfo.diPlus != null && adxInfo.diMinus != null && adxInfo.diPlus > adxInfo.diMinus);

        const donchBreakoutUp = (!USE_DONCHIAN_FILTER)
          ? true
          : (donchPrevHigh != null && Number.isFinite(donchPrevHigh) && candle.close > donchPrevHigh);

        const supertrendOk = (!USE_SUPERTREND_FILTER)
          ? true
          : (st.dir === 1);

        // IMPULSE mode (optional): relax caps ONLY in very strong context.
        // NOTE: computed AFTER donch/adx/trend are available, to avoid TDZ issues.
        const impulseEnabled = !!cfg.IMPULSE_ENABLED;
        const impulseRequireMicroTrending = (cfg.IMPULSE_REQUIRE_MICRO_TRENDING !== undefined)
          ? !!cfg.IMPULSE_REQUIRE_MICRO_TRENDING
          : true;
        const impulseMinAdx = Number.isFinite(Number(cfg.IMPULSE_MIN_ADX)) ? Number(cfg.IMPULSE_MIN_ADX) : 45;
        const impulseMaxAtrPct = Number.isFinite(Number(cfg.IMPULSE_MAX_ATR_PCT)) ? Number(cfg.IMPULSE_MAX_ATR_PCT) : 0;
        const impulseMaxMomentumPct = Number.isFinite(Number(cfg.IMPULSE_MAX_MOMENTUM_PCT)) ? Number(cfg.IMPULSE_MAX_MOMENTUM_PCT) : 0;

        // Compute an internal trend gate for IMPULSE (cannot reference trendUpEff here due to TDZ)
        const trendUpEffForImpulse = (String(TREND_GATE_MODE).toLowerCase() === 'no_trend') ? true : !!trendUp;

        const impulseCtx = impulseEnabled &&
          trendUpEffForImpulse &&
          priceAboveSMA &&
          donchBreakoutUp &&
          (adxInfo.adx != null && Number.isFinite(adxInfo.adx) && adxInfo.adx >= impulseMinAdx) &&
          (!impulseRequireMicroTrending || regimeInfo.microRegime === 'TRENDING');

        const maxMomEff = (impulseCtx && impulseMaxMomentumPct > 0) ? impulseMaxMomentumPct : dynamicMaxMomentumBase;
        const momentumOk = momentum_pct >= effectiveMinMom && momentum_pct <= maxMomEff;

        const maxAtrEff = (impulseCtx && impulseMaxAtrPct > 0) ? impulseMaxAtrPct : MAX_ATR_PCT;
        const atrMaxOk = (!Number.isFinite(maxAtrEff) || maxAtrEff <= 0)
          ? true
          : (atrPctNum <= maxAtrEff);

        // Baseline-parity defensive regime skips (normalized)
        const lowVolBlocked = !!SKIP_LOW_VOL_HARD && (regimeInfo.volRegime === 'LOW_VOL');

        const choppyBlocked = (() => {
          if (regimeInfo.microRegime !== 'CHOPPY') return false;

          // Preferred (clean config): CHOPPY_SKIP_MODE
          if (CHOPPY_SKIP_MODE) {
            if (CHOPPY_SKIP_MODE === 'off') return false;
            if (CHOPPY_SKIP_MODE === 'always') return true;
            if (CHOPPY_SKIP_MODE === 'low_vol') return regimeInfo.volRegime === 'LOW_VOL';
            if (CHOPPY_SKIP_MODE === 'high_vol') return regimeInfo.volRegime === 'HIGH_VOL';
            if (CHOPPY_SKIP_MODE === 'extremes') return (regimeInfo.volRegime === 'LOW_VOL' || regimeInfo.volRegime === 'HIGH_VOL');
            // Unknown mode => be conservative: do not block
            return false;
          }

          // Legacy knobs (backward compatible)
          if (!SKIP_CHOPPY) return false;

          // Optional: skip CHOPPY only when LOW_VOL
          if (SKIP_CHOPPY_ONLY_IF_LOW_VOL) {
            return regimeInfo.volRegime === 'LOW_VOL';
          }

          // Optional: separate guardrail for HIGH_VOL+CHOPPY
          if (regimeInfo.volRegime === 'HIGH_VOL') {
            return !!SKIP_CHOPPY_IN_HIGH_VOL;
          }

          // Default: block CHOPPY
          return true;
        })();

        const regimeOk = !lowVolBlocked && !choppyBlocked;

        // Trend gate mode (strict by default)
        const trendUpEff = (String(TREND_GATE_MODE).toLowerCase() === 'no_trend') ? true : !!trendUp;

        // Hour block filter (CLOSED_STRATEGY: skip trading at certain UTC hours)
        // Use the just-closed candle timestamp (not a TDZ 'current' reference)
        const entryHour = new Date(Number(candle.ts)).getUTCHours();
        const blockedHour = Array.isArray(cfg.BLOCKED_HOURS_UTC) && cfg.BLOCKED_HOURS_UTC.includes(entryHour);

        // IMPULSE3 bypass (module)
        const { explosiveBlock, impulseCount } = evaluateImpulseBypass({
          rt,
          cfg,
          baseMinMomentum: BASE_MIN_MOM,
          momentum_pct,
          volumeOk,
          priceAboveSMA,
          candleExplosive
        });

        const shouldEnter = regimeOk &&
          atrOk &&
          atrHardOk &&
          atrMaxOk &&
          slopeNormOk &&
          momentumOk &&
          trendUpEff &&
          priceNearSMA &&
          priceAboveSMA &&
          volumeOk &&
          adxOk &&
          diBullOk &&
          donchBreakoutUp &&
          supertrendOk &&
          !weakCandleBody &&
          !weakOpen &&
          !explosiveBlock &&
          !blockedHour;

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
          atrHardOk: !!atrHardOk,
          atrMaxOk: !!atrMaxOk,
          regimeOk: !!regimeOk,
          realized_vol: Number((realizedVol || 0).toFixed(6)),
          adx: (adxInfo.adx == null ? null : Number(adxInfo.adx.toFixed(2))),
          diPlus: (adxInfo.diPlus == null ? null : Number(adxInfo.diPlus.toFixed(2))),
          diMinus: (adxInfo.diMinus == null ? null : Number(adxInfo.diMinus.toFixed(2))),
          donchHigh: (donch.high == null ? null : Number(donch.high.toFixed(2))),
          donchLow: (donch.low == null ? null : Number(donch.low.toFixed(2))),
          supertrendDir: (st.dir == null ? null : st.dir),
          supertrend: (st.value == null ? null : Number(st.value.toFixed(2))),
          min_slope_norm: Number((minSlopeNormEff || 0).toFixed(3)),
          slopeNormOk: !!slopeNormOk,
          regime,
          microRegime: regimeInfo.microRegime,
          slopeNorm: Number((regimeInfo.slopeNorm || 0).toFixed(3)),
          priceNearSMA: !!priceNearSMA,
          priceAboveSMA: !!priceAboveSMA,
          volume_base: Number(candle.volume.toFixed(2)),
          volume_quote: Number.isFinite(candle.quoteVolume) ? Number(candle.quoteVolume.toFixed(2)) : null,
          volumeOk: !!volumeOk,
          trendUp: !!trendUp,
          trendUpEff: (String(TREND_GATE_MODE).toLowerCase() === 'no_trend') ? true : !!trendUp,
          weakCandleBody: !!weakCandleBody,
          weakOpen: !!weakOpen,
          candleExplosive: !!candleExplosive,
          adxOk: !!adxOk,
          diBullOk: !!diBullOk,
          donchBreakoutUp: !!donchBreakoutUp,
          supertrendOk: !!supertrendOk,
          shouldEnter: !!shouldEnter,
          effective_min_momentum: Number(effectiveMinMom.toFixed(6)),
          green_run,
          // IMPULSE3 bypass
          impulseCount: Number(impulseCount || 0),
          impulseBypassN: Number(cfg.IMPULSE3_BYPASS_EXPLOSIVE_N || 0)
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
            'atrHardOk=' + (decision.atrHardOk ? 1 : 0),
            'atrMaxOk=' + (decision.atrMaxOk ? 1 : 0),
            'regimeOk=' + (decision.regimeOk ? 1 : 0),
            'slopeNorm=' + decision.slopeNorm,
            'min_slope_norm=' + decision.min_slope_norm,
            'slopeNormOk=' + (decision.slopeNormOk ? 1 : 0),
            'rv=' + decision.realized_vol,
            'adx=' + (decision.adx == null ? 'null' : decision.adx),
            'di+=' + (decision.diPlus == null ? 'null' : decision.diPlus),
            'di-=' + (decision.diMinus == null ? 'null' : decision.diMinus),
            'donchH=' + (decision.donchHigh == null ? 'null' : decision.donchHigh),
            'donchL=' + (decision.donchLow == null ? 'null' : decision.donchLow),
            'stDir=' + (decision.supertrendDir == null ? 'null' : decision.supertrendDir),
            'regime=' + decision.regime,
            'micro=' + decision.microRegime,
            'priceNearSMA=' + (decision.priceNearSMA ? 1 : 0),
            'priceAboveSMA=' + (decision.priceAboveSMA ? 1 : 0),
            'trendUp=' + (decision.trendUp ? 1 : 0),
            'momentumOk=' + (momentumOk ? 1 : 0),
            'adxOk=' + (adxOk ? 1 : 0),
            'diBullOk=' + (diBullOk ? 1 : 0),
            'trendUpEff=' + (decision.trendUpEff ? 1 : 0),
            'donchOk=' + (donchBreakoutUp ? 1 : 0),
            'stOk=' + (supertrendOk ? 1 : 0),
            'weakBody=' + (weakCandleBody ? 1 : 0),
            'weakOpen=' + (weakOpen ? 1 : 0),
            'explosive=' + (candleExplosive ? 1 : 0),
            'shouldEnter=' + (decision.shouldEnter ? 1 : 0),
            'effective_min_momentum=' + decision.effective_min_momentum,
            'green_run=' + decision.green_run,
            'impulseCount=' + (impulseCount || 0),
            'impulseBypassN=' + (cfg.IMPULSE3_BYPASS_EXPLOSIVE_N || 0),
            'marketBias=' + marketBias,
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
            'atrHardOk=' + (decision.atrHardOk ? 1 : 0),
            'atrMaxOk=' + (decision.atrMaxOk ? 1 : 0),
            'regimeOk=' + (decision.regimeOk ? 1 : 0),
            'slopeNorm=' + decision.slopeNorm,
            'slopeNormOk=' + (decision.slopeNormOk ? 1 : 0),
            'shouldEnter=' + (decision.shouldEnter ? 1 : 0),
            'marketBias=' + marketBias,
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

        const takeProfit = entryPrice + riskDist * rr;

        const stopLossClassic = entryPrice - riskDist;
        const stopLossEmergency = (Number.isFinite(EMERGENCY_SL_PCT) && EMERGENCY_SL_PCT > 0)
          ? (entryPrice * (1 - EMERGENCY_SL_PCT))
          : stopLossClassic;

        // Which SL is "armed" as hard stop in state:
        // - CLASSIC: classic SL
        // - SOFT_CLASSIC_WITH_EMERGENCY: emergency SL is hard airbag; classic evaluated with confirmation
        // - EMERGENCY_ONLY: emergency SL only
        const stopLoss = (SL_POLICY === 'SOFT_CLASSIC_WITH_EMERGENCY' || SL_POLICY === 'EMERGENCY_ONLY')
          ? stopLossEmergency
          : stopLossClassic;

        const stopDistanceUSD = riskDist;
        const current = klines[klines.length - 1];
        const currentLow = Number(current?.[3]);

        // ---- TAMAÑO DINÁMICO IMPULSE ----
        const impulseMult = Number(cfg.IMPULSE_TRADE_MULT || 1.0);
        const effectiveTradeUsd = (impulseCtx && impulseMult > 1) ? tradeUSD * impulseMult : tradeUSD;
        // ----------------------------------

        const qty = Number((effectiveTradeUsd / entryPrice).toFixed(8));

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
          min_slope_norm: Number((minSlopeNormEff || 0).toFixed(3)),
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
          stopLossClassic,
          stopLossEmergency,
          takeProfit,
          mode: runMode,
          slPolicy: SL_POLICY,
          slConfirmSeconds: SL_CONFIRM_SECONDS,
          openedAt: nowIso(),
          signalCandleTs: candle.ts,
          size: qty,
          exposureUSD: Number((entryPrice * qty).toFixed(2)),
          type: 'LONG',
          regime,
          reasonTag,
          reasonDetails: reasonDet
        };


        const didOpen = (runMode === 'live')
          ? await openTradeLive(trade, signalCooldownMs)
          : await openTrade(trade, candleBucketMs, signalCooldownMs, VERBOSE);
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
