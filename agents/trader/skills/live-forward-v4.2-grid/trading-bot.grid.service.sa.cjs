#!/usr/bin/env node

// trading-bot.grid.service.sa.cjs — Grid trading bot (separated from momentum)
// Reuses common SA helpers; no interference with existing momentum bot.

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
const dotenv = require('dotenv');

// Reuse common SA helpers from local bot/sa
const { ensureDir, readJsonFile } = require('./bot/sa/file_io.cjs');
const { sleep, isValidNumber, percentile } = require('./bot/sa/util.cjs');
const { getYmd, fmtDateUtc1, nowIso, makeGetJournalPath } = require('./bot/sa/time.cjs');
const { createFileLock } = require('./bot/sa/lock.cjs');
const { createJsonIo } = require('./bot/sa/json_io.cjs');
const { createJournalStateManager } = require('./bot/sa/journal_state.cjs');
const { createRateLimiter } = require('./bot/sa/rate_limit.cjs');
const { createHttpRetry } = require('./bot/sa/http_retry.cjs');
const { createMarketData } = require('./bot/sa/market_data.cjs');
const { buildCloseReason } = require('./bot/sa/trade_helpers.cjs');
const { createMexcSpotClient } = require('./exchange/mexc_spot_client.cjs');

// Indicators (reuse momentum bot calculations for dashboard/journal)
const {
  getClosedCloses,
  computeSmaPair,
  computeAtr,
  computeRealizedVol,
  detectMarketRegime
} = require('../../market-regime.sa.cjs');

// Grid engine
const { GridEngine } = require('./bot/grid/engine.cjs');

// Config
const LABEL = 'grid';
// Code dir (where this service lives)
const CODE_DIR = path.join(__dirname);

// State dir (journals/snapshots/locks). Allows separating GRID PAPER vs GRID LIVE like the main LIVE bot.
const BASE_DIR = process.env.GRID_BASE_DIR ? path.resolve(process.env.GRID_BASE_DIR) : CODE_DIR;

// Env loading: allow reusing the LIVE bot .env (Victor request)
// Default: .env next to this service code (not state dir).
dotenv.config({ path: process.env.DOTENV_PATH || path.join(CODE_DIR, '.env') });

const CONFIG_PATH = process.env.GRID_CONFIG_PATH || path.join(CODE_DIR, 'config_grid.json');
const LOCK_PATH = path.join(BASE_DIR, 'grid.lock');
const OPEN_POSITIONS_PATH = path.join(BASE_DIR, 'open_positions.json');
const OPEN_ORDERS_PATH = path.join(BASE_DIR, 'open_orders.json');
const BALANCES_PATH = path.join(BASE_DIR, 'balances.json');
const FORCED_SELL_TIME_STOP_CANCELS_PATH = path.join(BASE_DIR, 'forced_sell_time_stop_cancels.jsonl');
const DISPOSAL_STATE_PATH = path.join(BASE_DIR, 'disposal_state.json');

function loadGridConfig() {
  return readJsonFile(CONFIG_PATH, {});
}

const cfg = loadGridConfig();
// Runtime overrides via env (LIVE launch uses env for safety)
cfg.TRADING_MODE = process.env.TRADING_MODE || cfg.TRADING_MODE || 'PAPER';

// Safety: only apply sizing/behavior overrides when explicitly in LIVE mode.
if (String(cfg.TRADING_MODE).toUpperCase() === 'LIVE') {
  if (process.env.CAPITAL_USD != null) cfg.CAPITAL_USD = Number(process.env.CAPITAL_USD);
  if (process.env.SEED_INVENTORY_PCT != null) cfg.SEED_INVENTORY_PCT = Number(process.env.SEED_INVENTORY_PCT);
  if (process.env.MAKER_ENTRY_PRICE_OFFSET_PCT != null) cfg.MAKER_ENTRY_PRICE_OFFSET_PCT = Number(process.env.MAKER_ENTRY_PRICE_OFFSET_PCT);
  if (process.env.SPACING_USD != null) cfg.SPACING_USD = Number(process.env.SPACING_USD);
  if (process.env.REQUIRE_KEYS != null) cfg.REQUIRE_KEYS = Number(process.env.REQUIRE_KEYS);
}
const EXCHANGE = process.env.EXCHANGE || cfg.EXCHANGE || 'binance';
const MARKET_DATA_EXCHANGE = process.env.MARKET_DATA_EXCHANGE || cfg.MARKET_DATA_EXCHANGE || EXCHANGE;

// Global state
let shutdownRequested = false;
let iterCount = 0;

const state = {
  openTradesById: new Map(),
  openTradesBySymbol: new Map(),
  openTradeIdBySignalKey: new Map(),
  seenEventKeys: new Set(),
  recentSignalSeenAt: new Map(),
  symbolRuntime: new Map(),
  decisionBySymbol: new Map(),
  tickerCacheBySymbol: new Map(),
  recentOpenTimes: [],
  openCountsByDay: new Map(),
  lastOpenBySymbol: {},
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
      grid: null,
      lastRebalanceTs: 0,
      lastReconcileTs: 0,
      lastMyTradesCheckTs: 0,

      // ATR adaptive (per-symbol), same concept as momentum bot (used for logging only)
      lastAtrCandleTs: null,
      atrHistory: [],

      // PAPER 3/100 rebalance gate telemetry for dashboard/snapshot
      lastRebalanceGate: null,
    });
  }
  return state.symbolRuntime.get(symbol);
}

function isPaper3100Mode() {
  try {
    return !isLiveMode() && String(BASE_DIR || '').includes('state_paper_3100');
  } catch (_) {
    return false;
  }
}

// Exchange clients
const API_BASES = { binance: 'https://api.binance.com', mexc: 'https://api.mexc.com' };
function getApiBase(exchange) { return API_BASES[exchange] || API_BASES.binance; }

const API_KEYS = {
  binance: { key: process.env.BINANCE_API_KEY || '', secret: process.env.BINANCE_API_SECRET || '' },
  mexc: { key: process.env.MEXC_API_KEY || '', secret: process.env.MEXC_API_SECRET || '' }
};
const ACTIVE_API_KEY = API_KEYS[EXCHANGE]?.key || '';
const ACTIVE_API_SECRET = API_KEYS[EXCHANGE]?.secret || '';

let _mexcClient = null;
function getMexcClient() {
  if (_mexcClient) return _mexcClient;
  if (EXCHANGE !== 'mexc') throw new Error(`getMexcClient() called but EXCHANGE=${EXCHANGE}`);
  if (!ACTIVE_API_KEY || !ACTIVE_API_SECRET) throw new Error('MEXC API keys not set (MEXC_API_KEY/MEXC_API_SECRET)');
  _mexcClient = createMexcSpotClient({
    baseUrl: getApiBase(EXCHANGE),
    apiKey: ACTIVE_API_KEY,
    apiSecret: ACTIVE_API_SECRET,
    timeoutMs: 2500
  });
  return _mexcClient;
}

const REQUIRE_KEYS = String(process.env.REQUIRE_KEYS || cfg.REQUIRE_KEYS || '0') === '1';
if (REQUIRE_KEYS && (!ACTIVE_API_KEY || !ACTIVE_API_SECRET)) {
  console.error(`API keys for ${EXCHANGE} not set in .env (REQUIRE_KEYS=1)`);
  process.exit(1);
}

console.log('Service: trading-bot.grid.service.sa.cjs');
console.log('Exchange (execution):', EXCHANGE);
console.log('Exchange (market data):', MARKET_DATA_EXCHANGE);
console.log('API Base (market data):', getApiBase(MARKET_DATA_EXCHANGE));

// Helpers
const { writeJsonFileAtomic, appendJsonl, loadJsonl } = createJsonIo({ ensureDir });
const getJournalPath = makeGetJournalPath(BASE_DIR);
const { acquireLock, releaseLock } = createFileLock({
  ensureDir,
  baseDir: BASE_DIR,
  lockPath: LOCK_PATH,
  nowIso
});

// Journal state manager (compatible)
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
  canOpenSignal
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
  noteOpenTimestamp: (ts) => state.recentOpenTimes.push(ts),
  getYmd
});

// HTTP
const { httpGetWithRetry } = createHttpRetry({ axios, sleep });

const { getTicker, getTickerCached, getRecentKlines, rankSymbolsByRecentReturn } = createMarketData({
  getApiBase: () => getApiBase(MARKET_DATA_EXCHANGE),
  httpGetWithRetry,
  activeApiKey: ACTIVE_API_KEY,
  exchange: MARKET_DATA_EXCHANGE,
  state
});

// Grid engine factory
function createGridEngine(symbol) {
  return new GridEngine({
    symbol,
    cfg,
    marketData: {
      getTicker: () => getTicker(symbol),
      getRecentKlines: (limit) => getRecentKlines(symbol, limit, '1m')
    },
    nowIso,
    sleep,
    isValidNumber,
    logger: console
  });
}

async function computeEntryContextForJournal(symbol) {
  // NOTE: calculation-only; does NOT affect grid entries.
  try {
    const rt = getSymbolRuntime(symbol);

    const SMA_WINDOW = Number(cfg.SMA_WINDOW ?? 20);
    const smaTol = Number(cfg.SMA_TOLERANCE ?? 0.001);

    const ATR_WINDOW = Number(cfg.ATR_WINDOW ?? 14);

    const BASE_MIN_MOM = Number(cfg.MIN_MOMENTUM_PCT ?? 0);
    const MAX_MOM = Number(cfg.MAX_MOMENTUM_PCT ?? 0);

    const minSmaSlopePct = Number(cfg.MIN_SMA_SLOPE_PCT ?? 0);

    const ATR_ADAPTIVE_ENABLED = !!cfg.ATR_ADAPTIVE_ENABLED;
    const ATR_ADAPTIVE_PCTL = Number(cfg.ATR_ADAPTIVE_PCTL ?? 0.55);
    const ATR_ADAPTIVE_WINDOW = Number(cfg.ATR_ADAPTIVE_WINDOW ?? 240);
    const ATR_ADAPTIVE_MIN_SAMPLES = Number(cfg.ATR_ADAPTIVE_MIN_SAMPLES ?? 60);

    const MIN_ATR_PCT = Number(cfg.MIN_ATR_PCT ?? 0);

    const GREEN_KLINES = Number(cfg.GREEN_KLINES_FOR_REDUCTION ?? 0);
    const MOM_REDUCTION_PCT = Number(cfg.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN ?? 0);

    const limit = Number(cfg.INDICATOR_KLINES_LIMIT ?? (Math.max(220, SMA_WINDOW + ATR_WINDOW + 50)));

    const klines = await getRecentKlines(symbol, limit, '1m');
    if (!Array.isArray(klines) || klines.length < Math.max(ATR_WINDOW + 3, SMA_WINDOW + 3)) return null;

    const lastClosed = klines[klines.length - 2];
    const candle = {
      ts: Number(lastClosed[0]),
      open: Number(lastClosed[1]),
      high: Number(lastClosed[2]),
      low: Number(lastClosed[3]),
      close: Number(lastClosed[4]),
      volume: Number(lastClosed[5])
    };

    const closes = getClosedCloses(klines);
    if (!Array.isArray(closes) || closes.length < 3) return null;

    const last = closes[closes.length - 1];
    const prev = closes[closes.length - 2] || last;
    const momentum_pct = prev ? (last - prev) / prev : 0;

    const { sma, smaPrev } = computeSmaPair(closes, SMA_WINDOW);
    const smaSlopeAbs = (sma != null && smaPrev != null) ? (sma - smaPrev) : 0;
    const smaSlopePct = (sma != null && smaPrev != null && smaPrev !== 0) ? (sma - smaPrev) / smaPrev : 0;

    const priceNearSMA = (sma != null) ? (candle.close >= sma * (1 - smaTol)) : true;
    const priceAboveSMA = (sma != null) ? (candle.close > sma) : true;

    const { atrPct } = computeAtr(klines, ATR_WINDOW);
    const atrPctNum = Number.isFinite(Number(atrPct)) ? Number(atrPct) : 0;

    // ATR adaptive history (per symbol)
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

    const realizedVol = computeRealizedVol(closes, Math.min(60, Math.max(20, Math.floor(SMA_WINDOW / 2))));

    const regimeInfo = detectMarketRegime({
      atrPct: atrPctNum,
      realizedVol,
      smaSlope: smaSlopeAbs,
      lastClose: candle.close,
      priceAboveSma: priceAboveSMA
    });

    // Green run (same metric as momentum bot)
    let green_run = 0;
    if (GREEN_KLINES > 0 && Array.isArray(klines)) {
      for (let j = klines.length - 2; j > 0 && green_run < GREEN_KLINES; j--) {
        const cur = Number(klines[j][4]);
        const op = Number(klines[j][1]);
        if (Number.isFinite(cur) && Number.isFinite(op) && cur > op) green_run++;
        else break;
      }
    }

    const dynamicMinMomentum = (BASE_MIN_MOM || 0) * (Number(regimeInfo?.kMinMomentum) || 1);

    let effectiveMinMom = dynamicMinMomentum;
    if (green_run >= GREEN_KLINES && GREEN_KLINES > 0 && MOM_REDUCTION_PCT > 0) {
      effectiveMinMom = dynamicMinMomentum * (1 - MOM_REDUCTION_PCT);
    }

    // NOTE: We store thresholds for dashboard display; grid engine does not use them.
    return {
      sma: (sma == null) ? null : Number(sma),
      price_near_sma: !!priceNearSMA,
      price_above_sma: !!priceAboveSMA,

      momentum_pct: Number(momentum_pct),
      base_min_momentum: Number(BASE_MIN_MOM || 0),
      effective_minimum: Number(effectiveMinMom || 0),
      max_momentum_pct: Number(MAX_MOM || 0),
      green_run,

      sma_slope: Number(smaSlopeAbs),
      sma_slope_pct: Number(smaSlopePct),
      min_sma_slope_pct: Number(minSmaSlopePct || 0),

      atr_pct: Number(atrPctNum),
      min_atr_pct: Number(minAtrEffective || 0),

      regime: regimeInfo?.volRegime ?? null,
      micro_regime: regimeInfo?.microRegime ?? null,
      slope_norm: Number(regimeInfo?.slopeNorm || 0),
      dist_from_sma_pct: (sma == null || !Number.isFinite(Number(candle.close)) || !Number.isFinite(Number(sma)) || Number(sma) === 0)
        ? null
        : Number((Number(candle.close) - Number(sma)) / Number(sma)),
    };
  } catch (e) {
    return null;
  }
}

async function flushOpenPositionsSnapshotWithPaperGate(force = false) {
  await flushOpenPositionsSnapshot(force);
  if (!isPaper3100Mode()) return;
  try {
    const rtEntries = Array.from(state.symbolRuntime.entries()).map(([symbol, rt]) => ({
      symbol,
      paper_rebalance_gate: rt?.lastRebalanceGate || null,
    }));
    const current = readJsonFile(OPEN_POSITIONS_PATH, []);
    const arr = Array.isArray(current) ? current : [];
    const meta = {
      _paper_rebalance_gate_meta: {
        ts: nowIso(),
        mode: 'PAPER_3100',
        symbols: rtEntries,
      }
    };
    await writeJsonFileAtomic(OPEN_POSITIONS_PATH, Object.assign(arr, meta));
  } catch (e) {
    console.warn(`[GRID PAPER 3100] snapshot gate meta write failed: ${e.message}`);
  }
}

function mergeReasonDetails(trade, extra) {
  if (!extra) return trade;
  const cur = trade.reason_details || trade.reasonDetails || {};
  trade.reason_details = { ...cur, ...extra };
  return trade;
}

function loadForcedTpCancelIndex() {
  try {
    if (!fs.existsSync(FORCED_SELL_TIME_STOP_CANCELS_PATH)) return new Set();
    const lines = fs.readFileSync(FORCED_SELL_TIME_STOP_CANCELS_PATH, 'utf8').split('\\n').filter(Boolean);
    const out = new Set();
    for (const line of lines) {
      try {
        const rec = JSON.parse(line);
        const buyId = String(rec?.buyOrderId || '');
        const active = rec?.active_skip_tp_reattach;
        if (buyId && active) out.add(buyId);
      } catch (_) {}
    }
    return out;
  } catch (_) {
    return new Set();
  }
}

async function appendForcedTpCancelAudit(rec) {
  try {
    await appendJsonl(FORCED_SELL_TIME_STOP_CANCELS_PATH, rec);
    return true;
  } catch (_) {
    return false;
  }
}

function isBuyIdForcedTpCancelled(buyId) {
  if (!buyId) return false;
  const idx = loadForcedTpCancelIndex();
  return idx.has(String(buyId));
}

function loadDisposalState() {
  try {
    if (!fs.existsSync(DISPOSAL_STATE_PATH)) {
      return { disposal_active: false, disposal_orders: [], items: [] };
    }
    const raw = JSON.parse(fs.readFileSync(DISPOSAL_STATE_PATH, 'utf8'));
    return {
      disposal_active: !!raw?.disposal_active,
      disposal_orders: Array.isArray(raw?.disposal_orders) ? raw.disposal_orders : [],
      items: Array.isArray(raw?.items) ? raw.items : [],
    };
  } catch (_) {
    return { disposal_active: false, disposal_orders: [], items: [] };
  }
}

async function saveDisposalState(st) {
  await writeJsonFileAtomic(DISPOSAL_STATE_PATH, {
    disposal_active: !!st?.disposal_active,
    disposal_orders: Array.isArray(st?.disposal_orders) ? st.disposal_orders : [],
    items: Array.isArray(st?.items) ? st.items : [],
    updatedAt: nowIso(),
  });
}

async function rebuildDisposalStateFromForcedCancels() {
  const thresholdCfg = Number(cfg.DISPOSAL_THRESHOLD ?? cfg.LEVELS_PER_SIDE ?? 1);
  const maxLevels = Math.max(1, Number(cfg.LEVELS_PER_SIDE || 1));
  const threshold = Math.min(maxLevels, Math.max(1, thresholdCfg));
  const state0 = loadDisposalState();
  const items = [];
  try {
    if (fs.existsSync(FORCED_SELL_TIME_STOP_CANCELS_PATH)) {
      const lines = fs.readFileSync(FORCED_SELL_TIME_STOP_CANCELS_PATH, 'utf8').split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          const rec = JSON.parse(line);
          const buyOrderId = String(rec?.buyOrderId || '');
          const entryPrice = Number(rec?.entryPrice ?? null);
          const qty = Number(rec?.qty ?? null);
          if (!buyOrderId || !Number.isFinite(entryPrice) || !Number.isFinite(qty)) continue;
          items.push({
            buyOrderId,
            symbol: String(rec?.symbol || 'BTCUSDT'),
            entryPrice,
            qty,
            cancelledAt: rec?.ts || null,
            targetPrice: Math.max(entryPrice, entryPrice + Number(cfg.SPACING_USD || 0)),
            daysWaiting: rec?.ts ? Math.max(0, (Date.now() - Date.parse(rec.ts)) / 86400000) : 0,
            limitOrderId: null,
            limitClientOrderId: null,
            status: 'waiting',
          });
        } catch (_) {}
      }
    }
  } catch (_) {}
  const next = {
    disposal_active: !!cfg.DISPOSAL_ENABLED && items.length >= threshold,
    disposal_orders: state0.disposal_orders || [],
    items,
  };
  await saveDisposalState(next);
  return next;
}

async function maybeRunDisposalModule(symbol) {
  const state1 = await rebuildDisposalStateFromForcedCancels();
  if (!cfg.DISPOSAL_ENABLED) return state1;
  return state1;
}

async function reconcileTpFillsFromMyTrades(symbol) {
  // Reconcile executed TP SELL fills from exchange into journal CLOSE events.
  // This prevents "missing closes" after restarts or when we miss an order poll.
  if (!isLiveMode()) return;

  const rt = getSymbolRuntime(symbol);
  const now = Date.now();
  const intervalMs = 10_000; // lightweight; myTrades limit small
  if (rt.lastMyTradesCheckTs && (now - rt.lastMyTradesCheckTs) < intervalMs) return;
  rt.lastMyTradesCheckTs = now;

  try {
    const mexc = getMexcClient();
    const tr = await mexc.myTrades({ symbol, limit: 50 });
    const arr = Array.isArray(tr) ? tr : (tr?.data || []);
    const sells = arr.filter((t) => !t.isBuyer);
    if (!sells.length) return;

    const opens = getOpenTradesArray(symbol);

    for (const s of sells) {
      const cid = String(s.clientOrderId || s.origClientOrderId || '');
      if (!cid.startsWith('TPSELL_')) continue;
      const buyId = buyIdFromTpSellClientId(cid);

      const sellPrice = Number(s.price);
      const sellQty = Number(s.qty);
      const sellTimeMs = Number(s.time || s.timestamp || 0);
      if (!Number.isFinite(sellPrice) || !Number.isFinite(sellQty) || !Number.isFinite(sellTimeMs)) continue;

      if (hasJournalCloseForTpFill(cid, buyId, sellTimeMs)) continue;

      // Try match: tp == sell price and qty == size
      let match = opens.find((t) => {
        const tp = Number(t.takeProfit ?? t.tp);
        const q = Number(t.size ?? t.qty);
        if (!Number.isFinite(tp) || !Number.isFinite(q)) return false;
        const qtyOk = Math.abs(q - sellQty) <= Math.max(1e-12, sellQty * 0.002);
        const priceOk = Math.abs(tp - sellPrice) <= 0.02;
        return qtyOk && priceOk;
      });
      if (!match && buyId) {
        match = findJournalOpenTradeById(buyId, symbol);
      }
      if (!match) continue;

      // Root-cause guard against bad CLOSE records from partial TP fills:
      // only reconcile once the TP order itself is confirmed FILLED on exchange.
      let ord = null;
      try {
        ord = await getOrderLive({
          symbol,
          orderId: s.orderId || s.order_id || null,
          clientOrderId: cid || null,
        });
      } catch (_) {
        ord = null;
      }
      if (!ord) continue;

      const status = String(ord.status || ord.state || '').toUpperCase();
      const ordQty = Number(ord.origQty || ord.quantity || ord.origQuantity || ord.qty);
      const ordExecutedQty = Number(ord.executedQty || ord.executedQuantity || ord.cumulativeQuantity);
      const orderFilled = status === 'FILLED' || status === 'DONE';
      if (!orderFilled) continue;

      // Use order-confirmed final values instead of an individual myTrades fill row.
      const entry = Number(match.entryPrice ?? match.entry_price ?? 0);
      const tradeQty = Number(match.size ?? match.qty ?? 0);
      const finalQty = Number.isFinite(tradeQty) && tradeQty > 0
        ? tradeQty
        : (Number.isFinite(ordQty) && ordQty > 0 ? ordQty : sellQty);
      if (!(Number.isFinite(finalQty) && finalQty > 0)) continue;

      // Extra safety: if exchange says FILLED but executed qty is materially below the trade qty, skip.
      if (Number.isFinite(ordExecutedQty) && ordExecutedQty > 0 && ordExecutedQty + Math.max(1e-12, finalQty * 0.002) < finalQty) {
        continue;
      }

      const exitPrice = Number(ord.price || ord.avgPrice || sellPrice);
      const closeTimeMs = Number(ord.updateTime || ord.transactTime || sellTimeMs || 0);
      const profit = (Number.isFinite(entry) && Number.isFinite(exitPrice)) ? (exitPrice - entry) * finalQty : null;

      const closeTrade = {
        ...match,
        close_time_iso: closeTimeMs > 0 ? new Date(closeTimeMs).toISOString() : new Date(sellTimeMs).toISOString(),
        exit_price: exitPrice,
        close_reason: 'TP',
        qty: finalQty,
        size: finalQty,
        profit,
        profit_pct: (entry > 0 && Number.isFinite(exitPrice)) ? ((exitPrice - entry) / entry) * 100 : null,
        fee_usd_est: 0,
        profit_after_fees_est: profit,
        tp_fill_client_id: cid,
      };

      await persistJournalEvent({ ts: closeTrade.close_time_iso || nowIso(), type: 'CLOSE', trade: closeTrade });
    }
  } catch (e) {
    // silent-ish; avoid log spam
  }
}

async function reconcileSymbolWithExchange(symbol) {
  // Purpose: prevent "ghost opens" when the user manually closes positions on the exchange.
  // Strategy:
  // - TP sell orders placed by the bot use clientOrderId = TPSELL_<SYMBOL>_L<level>_<batch13>
  // - If a trade is open in our journal state but no longer has a corresponding TPSELL on exchange,
  //   we DO NOT fabricate a CLOSE. Instead, we try to re-place/attach the missing TPSELL.
  //   (Maker-only rule: never market-sell; and never close a grid trade at a loss due to missing TP order.)
  if (!isLiveMode()) return;
  if (!cfg.RECONCILE_ENABLED) return;

  const rt = getSymbolRuntime(symbol);
  const intervalMin = Number(cfg.RECONCILE_INTERVAL_MINUTES ?? 5);
  const graceMin = Number(cfg.RECONCILE_GRACE_MINUTES ?? 15);
  const intervalMs = Math.max(60_000, intervalMin * 60_000);
  const graceMs = Math.max(0, graceMin * 60_000);

  const now = Date.now();
  if (rt.lastReconcileTs && (now - rt.lastReconcileTs) < intervalMs) return;
  rt.lastReconcileTs = now;

  try {
    const mexc = getMexcClient();
    const oo = await mexc.openOrders({ symbol });
    const arr = Array.isArray(oo) ? oo : (oo?.data || []);

    // Keep a set of BUY ids that still have an active TPSELL on exchange.
    const keepBuyIds = new Set();
    for (const o of arr) {
      const side = String(o?.side || '').toUpperCase();
      if (side !== 'SELL') continue;
      const cid = String(o?.clientOrderId || o?.origClientOrderId || '');
      if (!cid.startsWith('TPSELL_')) continue;
      const buyId = buyIdFromTpSellClientId(cid);
      if (buyId) keepBuyIds.add(buyId);
    }

    const opens = getOpenTradesArray(symbol);
    if (!opens.length) return;

    let fixed = 0;
    for (const t of opens) {
      const buyId = String(t?.id || '');
      if (!buyId) continue;
      if (keepBuyIds.has(buyId)) continue;
      if (isBuyIdForcedTpCancelled(buyId)) continue;

      // Only act on trades old enough (avoid racing while TP order is still being placed).
      const openedAt = Date.parse(t?.openedAt || t?.open_time_iso || '') || 0;
      if (openedAt && graceMs > 0 && (now - openedAt) < graceMs) continue;

      // Try to re-place/attach the missing TPSELL.
      const qty = Number(t?.size ?? t?.qty ?? 0);
      const tp = Number(t?.takeProfit ?? t?.tp ?? t?.take_profit ?? 0);
      if (!(qty > 0) || !(tp > 0)) continue;

      const tpCid = makeTpSellClientOrderIdFromBuyId(buyId);
      try {
        const placed = await placeLimitMakerLive({ symbol, side: 'SELL', price: tp, qty, clientOrderId: tpCid });
        // Journal it (so we can see it in the dashboard/journal even if state reloads).
        await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: null, price: placed.price, qty: placed.qty, orderId: tpCid, kind: 'GRID_SELL_TP', correspondingBuyOrderId: buyId, exchangeOrderId: placed.orderId, note: 'reconcile_replace_missing_tp' } });
        fixed++;
      } catch (e) {
        // Best effort. Do NOT fabricate a close.
      }
    }

    if (fixed > 0) {
      await flushOpenPositionsSnapshotWithPaperGate(true);
      console.log(`[GRID RECONCILE ${symbol}] re-placed missing TPSELL=${fixed}`);
    }
  } catch (e) {
    console.warn(`[GRID RECONCILE ${symbol}] failed: ${e.message}`);
  }
}

async function ensureTpSellsForJournalOpenTrades(symbol) {
  if (!isLiveMode()) return;
  try {
    const mexc = getMexcClient();

    // Get current open SELL orders once
    const oo = await mexc.openOrders({ symbol });
    const arr = Array.isArray(oo) ? oo : (oo?.data || []);
    const sellClientIds = new Set();
    for (const o of arr) {
      if (String(o?.side || '').toUpperCase() !== 'SELL') continue;
      const cid = String(o?.clientOrderId || o?.origClientOrderId || '');
      if (cid) sellClientIds.add(cid);
    }

    // Leer posiciones abiertas desde el snapshot (open_positions.json)
    const snapshotPath = OPEN_POSITIONS_PATH;
    let opens = [];
    try {
      if (fs.existsSync(snapshotPath)) {
        opens = JSON.parse(fs.readFileSync(snapshotPath, 'utf8'));
      }
    } catch (_) {}

    if (!Array.isArray(opens) || opens.length === 0) return;

    for (const t of opens) {
      const buyId = String(t?.id || '');
      if (!buyId) continue;

      const qty = Number(t?.size ?? t?.qty ?? 0);
      const tp = Number(t?.takeProfit ?? t?.tp ?? t?.take_profit ?? 0);
      if (!(qty > 0) || !(tp > 0)) continue;

      const cid = makeTpSellClientOrderIdFromBuyId(buyId);
      if (sellClientIds.has(cid)) continue; // already present on exchange
      if (isBuyIdForcedTpCancelled(buyId)) continue;

      // Place missing TPSELL
      const placed = await placeLimitMakerLive({ symbol, side: 'SELL', price: tp, qty, clientOrderId: cid });
      await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: null, price: placed.price, qty: placed.qty, orderId: cid, kind: 'GRID_SELL_TP', correspondingBuyOrderId: buyId, exchangeOrderId: placed.orderId, note: 'ensure_tp_for_journal_open' } });

      // Update set to avoid duplicate in the same loop
      sellClientIds.add(cid);
    }
  } catch (e) {
    // Best-effort; avoid spam
  }
}

function makeTpSellClientOrderIdFromBuyId(buyId) {
  // BUY ids are like: BUY-BTCUSDT-L10-1775677040703
  // TPSELL id must be <=32 chars and unique across levels.
  // Format: TPSELL_<SYMBOL>_L<level>_<batch13>
  const m = /^BUY-([A-Z0-9]+)-L(\d+)-(\d{13})$/.exec(String(buyId || ''));
  if (!m) throw new Error(`Invalid BUY id for TPSELL mapping: ${buyId}`);
  const symbol = m[1];
  const level = m[2];
  const batch = m[3];
  const cid = `TPSELL_${symbol}_L${level}_${batch}`;
  // Safety check for MEXC constraint: ^[0-9a-zA-Z_-]{1,32}$
  if (!/^[0-9A-Za-z_-]{1,32}$/.test(cid)) throw new Error(`Bad TPSELL clientOrderId: ${cid}`);
  return cid;
}

function buyIdFromTpSellClientId(cid) {
  const m = /^TPSELL_([A-Z0-9]+)_L(\d+)_([0-9]{13})$/.exec(String(cid || ''));
  if (!m) return null;
  return `BUY-${m[1]}-L${m[2]}-${m[3]}`;
}

function listRecentJournalFiles(maxFiles = 45) {
  try {
    return fs.readdirSync(BASE_DIR)
      .filter((f) => /^trade_journal_\d{8}\.jsonl$/.test(f))
      .sort()
      .slice(-maxFiles)
      .map((f) => path.join(BASE_DIR, f));
  } catch (_) {
    return [];
  }
}

function findJournalOpenTradeById(buyId, symbol) {
  if (!buyId) return null;
  let current = null;
  for (const file of listRecentJournalFiles()) {
    let lines = [];
    try {
      lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
    } catch (_) {
      continue;
    }
    for (const line of lines) {
      try {
        const rec = JSON.parse(line);
        const t = rec?.trade;
        if (!t || t.id !== buyId) continue;
        if (symbol && t.symbol !== symbol) continue;

        if (rec.type === 'OPEN') {
          current = {
            ...t,
            id: t.id,
            symbol: t.symbol,
            type: String(t.side || 'LONG').toUpperCase() === 'LONG' ? 'LONG' : 'SHORT',
            entryPrice: Number(t.entryPrice ?? t.entry_price),
            takeProfit: Number(t.takeProfit ?? t.tp),
            stopLoss: t.stopLoss ?? t.sl ?? null,
            size: Number(t.size ?? t.qty),
            openedAt: t.openedAt ?? t.open_time_iso ?? null,
            reasonDetails: t.reasonDetails ?? t.reason_details ?? null,
          };
        } else if (rec.type === 'CLOSE') {
          current = null;
        }
      } catch (_) {}
    }
  }
  return current;
}

function hasJournalCloseForTpFill(tpCid, buyId, sellTimeMs) {
  const closeMs = Number.isFinite(sellTimeMs) && sellTimeMs > 0 ? sellTimeMs : null;
  for (const file of listRecentJournalFiles()) {
    let lines = [];
    try {
      lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
    } catch (_) {
      continue;
    }
    for (const line of lines) {
      try {
        const rec = JSON.parse(line);
        if (rec?.type !== 'CLOSE' || !rec.trade) continue;
        const t = rec.trade;

        // Strongest idempotency key: same TP fill client id.
        if (tpCid && t.tp_fill_client_id === tpCid) return true;

        // A trade id must only ever have one CLOSE. If it already has one in journal, do not emit another.
        if (buyId && t.id === buyId) return true;

        // Legacy fallback: tolerate small timestamp deltas between runtime-close and myTrades-close.
        if (buyId && closeMs) {
          const existingCloseMs = Date.parse(String(t.close_time_iso || ''));
          if (Number.isFinite(existingCloseMs) && Math.abs(existingCloseMs - closeMs) <= 30_000) {
            return true;
          }
        }
      } catch (_) {}
    }
  }
  return false;
}

function levelFromTpSellClientId(cid, symbol) {
  // TPSELL_<SYMBOL>_L<level>_<batch13>
  const re = new RegExp(`^TPSELL_${String(symbol || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}_L(\\d+)_([0-9]{13})$`);
  const m = re.exec(String(cid || ''));
  if (!m) return null;
  const lvl = Number(m[1]);
  return Number.isFinite(lvl) ? lvl : null;
}

async function getLiveTpSellLocks(symbol) {
  // Safety gate:
  // - Do NOT place new BUY orders for levels that already have an active TPSELL on exchange.
  // - Also enforce a global cap: if TPSELL count >= LEVELS_PER_SIDE, do not place any new BUYs.
  // This is exchange-truth based and should not disturb already-open orders.
  if (!isLiveMode()) return { tpSellCount: 0, lockedLevels: new Set() };

  let arr = null;

  // Prefer recent snapshot (written by writeOpenOrdersSnapshot) to avoid extra API calls.
  try {
    if (fs.existsSync(OPEN_ORDERS_PATH)) {
      const st = fs.statSync(OPEN_ORDERS_PATH);
      const ageMs = Date.now() - Number(st.mtimeMs || 0);
      if (Number.isFinite(ageMs) && ageMs >= 0 && ageMs <= 15_000) {
        const tmp = JSON.parse(fs.readFileSync(OPEN_ORDERS_PATH, 'utf8'));
        if (Array.isArray(tmp)) arr = tmp;
      }
    }
  } catch (_) {}

  // Fallback: query exchange
  if (!arr) {
    try {
      const mexc = getMexcClient();
      const oo = await mexc.openOrders({ symbol });
      arr = Array.isArray(oo) ? oo : (oo?.data || []);
    } catch (_) {
      arr = [];
    }
  }

  const lockedLevels = new Set();
  let tpSellCount = 0;

  for (const o of arr) {
    const side = String(o?.side || '').toUpperCase();
    if (side !== 'SELL') continue;
    const cid = String(o?.clientOrderId || o?.origClientOrderId || '');
    if (!cid.startsWith('TPSELL_')) continue;
    tpSellCount++;
    const lvl = levelFromTpSellClientId(cid, symbol);
    if (lvl != null) lockedLevels.add(lvl);
  }

  return { tpSellCount, lockedLevels };
}

function isLiveMode() {
  return String(cfg.TRADING_MODE || 'PAPER').toUpperCase() === 'LIVE';
}

function isPaperExtraLayersEnabled() {
  if (isLiveMode()) return false;
  return !!cfg.PAPER_EXTRA_LAYERS_ENABLED;
}

function getPaperExtraLevelSpecs() {
  if (!isPaperExtraLayersEnabled()) return [];
  const spacing = Number(cfg.SPACING_USD || 0);
  const raw = Array.isArray(cfg.PAPER_EXTRA_LEVELS_USD) ? cfg.PAPER_EXTRA_LEVELS_USD : [];
  return raw
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x) && x > 0 && spacing > 0)
    .map((offsetUsd) => ({ offsetUsd, level: Math.round(offsetUsd / spacing) }))
    .filter((x) => Number.isFinite(x.level) && x.level > Number(cfg.LEVELS_PER_SIDE || 0))
    .sort((a, b) => a.level - b.level);
}

function getPaperExtraQtyPerLevel(basePrice) {
  const capPct = Number(cfg.PAPER_EXTRA_CAPITAL_PCT || 0);
  const specs = getPaperExtraLevelSpecs();
  if (!(capPct > 0) || !specs.length || !(Number.isFinite(basePrice) && basePrice > 0)) return null;
  const capUsd = Number(cfg.CAPITAL_USD || 0) * capPct;
  if (!(capUsd > 0)) return null;
  return (capUsd / specs.length) / basePrice;
}

function getTradeBuyLevel(grid, tr) {
  // Prefer explicit level stored at open time (stable across rebalance / buyLevels overwrites).
  const rd = tr?.reason_details || tr?.reasonDetails || {};
  const lvl0 = rd?.level;
  const lvlN = (lvl0 == null) ? null : Number(lvl0);
  if (Number.isFinite(lvlN)) return lvlN;

  // Fallback: infer from current buyLevels mapping (can fail after recenter/overwrite).
  const buyLevel = (typeof grid?.findLevelForBuyOrder === 'function') ? grid.findLevelForBuyOrder(tr?.id) : null;
  const buyLevelN = (buyLevel == null) ? null : Number(buyLevel);
  return Number.isFinite(buyLevelN) ? buyLevelN : null;
}

function hasOpenTradeForLevel(grid, level) {
  const target = Number(level);
  if (!Number.isFinite(target)) return false;

  for (const tr of grid.openTrades?.values?.() || []) {
    const buyLevel = getTradeBuyLevel(grid, tr);
    if (buyLevel === target) return true;
  }
  return false;
}

function ensurePaperExtraBuyLevels(grid) {
  if (!isPaperExtraLayersEnabled() || !grid) return;
  const specs = getPaperExtraLevelSpecs();
  if (!specs.length) return;
  const qty = getPaperExtraQtyPerLevel(Number(grid.basePrice || 0));
  if (!(Number.isFinite(qty) && qty > 0)) return;

  for (const spec of specs) {
    const level = spec.level;
    if (hasOpenTradeForLevel(grid, level)) continue;

    const desiredPrice = Number(grid.basePrice) - spec.offsetUsd;
    const existing = grid.buyLevels?.get?.(level);
    if (existing && existing.status === 'pending') {
      existing.price = desiredPrice;
      existing.qty = qty;
      existing.paperExtra = true;
      existing.offsetUsd = spec.offsetUsd;
      continue;
    }

    if (typeof grid.scheduleBuy === 'function') {
      grid.scheduleBuy(level, desiredPrice);
      const placed = grid.buyLevels?.get?.(level);
      if (placed) {
        placed.qty = qty;
        placed.paperExtra = true;
        placed.offsetUsd = spec.offsetUsd;
      }
    }
  }
}

function getMakerOffsetPct() {
  // Reuse the same constant concept as the LIVE bot.
  const v = process.env.MAKER_ENTRY_PRICE_OFFSET_PCT ?? cfg.MAKER_ENTRY_PRICE_OFFSET_PCT;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0.0002;
}

function pickNum(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k];
    const n = (v == null) ? NaN : Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function orderIdFromResp(resp) {
  // MEXC can return different shapes; keep it defensive.
  return resp?.orderId || resp?.data?.orderId || resp?.data || resp?.result?.orderId || null;
}

function isInsufficientBalanceError(e) {
  const parts = [
    e?.message,
    e?.response?.data?.msg,
    e?.response?.data?.message,
    e?.response?.data?.code,
    e?.response?.data,
  ].filter(Boolean).map((x) => String(typeof x === 'object' ? JSON.stringify(x) : x).toLowerCase());

  return parts.some((s) =>
    s.includes('insufficient') ||
    s.includes('balance') ||
    s.includes('fund') ||
    s.includes('not enough') ||
    s.includes('oversold') ||
    s.includes('available')
  );
}

async function placeLimitMakerLive({ symbol, side, price, qty, clientOrderId }) {
  const mexc = getMexcClient();
  const offset = getMakerOffsetPct();

  // Ensure maker by quoting away from spread if needed.
  const bt = await mexc.bookTicker(symbol);
  const bid = pickNum(bt, 'bidPrice', 'bid');
  const ask = pickNum(bt, 'askPrice', 'ask');

  let p = Number(price);
  if (side === 'BUY') {
    // BUY maker (live-parity conservative): if p is at/above bid, pull it slightly below bid.
    // This reduces the chance of becoming marketable due to fast spread changes.
    if (bid != null && p >= bid) p = bid * (1 - offset);
    // If p would cross the ask, also pull it below ask.
    if (ask != null && p >= ask) p = Math.min(p, ask * (1 - offset));
  } else {
    // SELL maker: only unsafe if it would cross the bid (marketable).
    // If p <= bid, push it just above bid to ensure post-only.
    if (bid != null && p <= bid) p = bid * (1 + offset);
    // NOTE: do NOT push sells above ask (that makes fills much less likely and can leave BTC stuck).
  }

  const norm = await mexc.normalizeLimit(symbol, qty, p);
  const params = {
    symbol,
    side,
    type: 'LIMIT_MAKER',
    quantity: norm.qty,
    price: norm.price,
    newClientOrderId: clientOrderId,
  };

  const resp = await mexc.placeOrder(params);
  const orderId = orderIdFromResp(resp);
  return { orderId, clientOrderId, price: norm.price, qty: norm.qty, raw: resp };
}

async function getOrderLive({ symbol, orderId, clientOrderId }) {
  const mexc = getMexcClient();
  return mexc.getOrder({ symbol, orderId, origClientOrderId: clientOrderId });
}


async function maybeForceCancelExpiredTpSells(symbol, grid) {
  if (!isLiveMode()) return;
  if (!cfg.GRID_SELL_TIME_STOP_ENABLED) return;
  const maxHours = Number(cfg.GRID_SELL_TIME_STOP_HOURS ?? 0);
  if (!(Number.isFinite(maxHours) && maxHours > 0)) return;

  let balances = null;
  try {
    const mexc = getMexcClient();
    const acc = await mexc.account(4000);
    const bals = acc?.balances || [];
    const usdt = bals.find((b) => String(b.asset).toUpperCase() === 'USDT');
    balances = { freeUsdt: usdt ? Number(usdt.free) : 0 };
  } catch (_) {
    return;
  }

  if (!(balances && Number.isFinite(balances.freeUsdt) && balances.freeUsdt > 0)) return;

  const mexc = getMexcClient();
  const oo = await mexc.openOrders({ symbol });
  const arr = Array.isArray(oo) ? oo : (oo?.data || []);
  const nowMs = Date.now();

  for (const o of arr) {
    const side = String(o?.side || '').toUpperCase();
    if (side !== 'SELL') continue;
    const cid = String(o?.clientOrderId || o?.origClientOrderId || '');
    if (!cid.startsWith('TPSELL_')) continue;
    const buyId = buyIdFromTpSellClientId(cid);
    if (!buyId) continue;
    if (isBuyIdForcedTpCancelled(buyId)) continue;

    const tr = grid?.openTrades?.get?.(buyId) || findJournalOpenTradeById(buyId, symbol);
    if (!tr) continue;
    const openedAt = Date.parse(String(tr?.openedAt || tr?.open_time_iso || ''));
    if (!Number.isFinite(openedAt) || openedAt <= 0) continue;

    const ageHours = (nowMs - openedAt) / 3600000;
    if (!(ageHours >= maxHours)) continue;

    try {
      await mexc.cancelOrder({ symbol, orderId: o.orderId, origClientOrderId: cid }, 4000);

      // Prevent automatic TP recreation after this intentional cancel.
      if (grid?.sellLevels?.size) {
        for (const [k, sell] of grid.sellLevels.entries()) {
          if (!sell) continue;
          if (String(sell.correspondingBuyOrderId || '') !== String(buyId)) continue;
          grid.sellLevels.delete(k);
          break;
        }
      }

      await appendForcedTpCancelAudit({
        ts: nowIso(),
        reason: 'GRID_SELL_TIME_STOP',
        active_skip_tp_reattach: true,
        symbol,
        buyOrderId: buyId,
        clientOrderId: cid,
        exchangeOrderId: o.orderId || null,
        openedAt: tr?.openedAt || tr?.open_time_iso || null,
        entryPrice: Number(tr?.entryPrice ?? tr?.entry_price ?? null),
        qty: Number(tr?.size ?? tr?.qty ?? null),
        tpPrice: Number(tr?.takeProfit ?? tr?.tp ?? o?.price ?? null),
        ageHours,
        usdtFree: balances.freeUsdt,
      });

      console.log(`[GRID LIVE ${symbol}] forced TP cancel by time-stop: buyId=${buyId} age=${ageHours.toFixed(2)}h usdtFree=${balances.freeUsdt.toFixed(2)}`);
    } catch (e) {
      console.warn(`[GRID LIVE ${symbol}] forced TP cancel failed for ${cid}: ${e.message}`);
    }
  }
}

async function cancelAllGridOpenOrdersLive(symbol, opts = {}) {
  const mexc = getMexcClient();
  const oo = await mexc.openOrders({ symbol });
  const arr = Array.isArray(oo) ? oo : (oo?.data || []);

  const keepOrderIds = opts.keepOrderIds instanceof Set ? opts.keepOrderIds : new Set(opts.keepOrderIds || []);
  const keepClientOrderIds = opts.keepClientOrderIds instanceof Set ? opts.keepClientOrderIds : new Set(opts.keepClientOrderIds || []);

  for (const o of arr) {
    const cid = o.clientOrderId || o.origClientOrderId;

    // Only cancel our grid-tagged orders (client ids we create)
    const isGridTagged = (
      cid && (
        String(cid).startsWith('BUY-') ||
        String(cid).startsWith('SELL-') ||
        String(cid).startsWith('GBUY-') ||
        String(cid).startsWith('GSELL-')
      )
    );
    if (!isGridTagged) continue;

    // SAFETY: never cancel active TP sells for already-open trades.
    // Caller provides a keep-list (exchange orderIds / clientOrderIds) derived from grid.sellLevels.
    if (keepOrderIds.has(String(o.orderId))) continue;
    if (cid && keepClientOrderIds.has(String(cid))) continue;

    try {
      await mexc.cancelOrder({ symbol, orderId: o.orderId, origClientOrderId: cid });
    } catch (e) {
      // ignore; may already be filled/canceled
    }
  }
}

// Symbol processing
async function processSymbol(symbol) {
  const rt = getSymbolRuntime(symbol);
  if (!rt.grid) {
    rt.grid = createGridEngine(symbol);
    await rt.grid.initialize();

    // LIVE startup behavior (important operational invariant):
    // on every restart we want to keep existing TPSELL orders intact,
    // cancel stale BUY ladder orders, and re-center the BUY ladder immediately.
    // This restores the pre-existing expected behavior Victor described.
    if (isLiveMode()) {
      try {
        await cancelAllGridOpenOrdersLive(symbol, { keepOrderIds: new Set(), keepClientOrderIds: new Set() });
      } catch (e) {
        console.warn(`[GRID LIVE ${symbol}] startup cancel BUY ladder failed: ${e.message}`);
      }

      try {
        await rt.grid.rebalance();
        rt.lastRebalanceTs = Date.now();
        rt.allowStartupBuyBootstrap = true;
        console.log(`[GRID LIVE ${symbol}] startup rebalance complete`);
      } catch (e) {
        console.warn(`[GRID LIVE ${symbol}] startup rebalance failed: ${e.message}`);
      }
    }

    // Journal seed OPEN trades first (so later SELL fills can CLOSE them)
    if (typeof rt.grid.drainSeedOpens === 'function') {
      for (const t of rt.grid.drainSeedOpens()) {
        const ctx = await computeEntryContextForJournal(symbol);
        mergeReasonDetails(t, ctx);
        await persistJournalEvent({ ts: nowIso(), type: 'OPEN', trade: t });
      }
    } else {
      console.warn('[GRID] drainSeedOpens() missing — seed inventory disabled or old engine loaded');
    }

    // Journal initial grid orders (PAPER only)
    if (!isLiveMode()) {
      ensurePaperExtraBuyLevels(rt.grid);
      for (const o of rt.grid.drainNewOrders()) {
        await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: o });
      }
    }

    // Force snapshot now so /api/positions sees these opens immediately
    await flushOpenPositionsSnapshot();

    // Startup safety: on first boot cycle, check exchange TP SELL orders immediately
    // and force-cancel any already-expired ones before normal loop logic continues.
    if (isLiveMode()) {
      try {
        await maybeForceCancelExpiredTpSells(symbol, rt.grid);
      } catch (e) {
        console.warn(`[GRID LIVE ${symbol}] startup time-stop check failed: ${e.message}`);
      }
    }
  }

  const grid = rt.grid;

  // Update market (price, ATR)
  await grid.updateMarket();

  // Verbose sensitivity/status line
  if (cfg.VERBOSE) {
    rt._statusEvery = rt._statusEvery || 0;
    rt._statusEvery++;
    // Log every ~6 iterations to avoid spam (with MONITOR_INTERVAL_MS=5s => ~30s)
    if (rt._statusEvery % 6 === 0) {
      const s = grid.getStatus();
      if (s && s.price) {
        const inRangeStr = (s.inRange === null) ? 'n/a' : (s.inRange ? 'IN_RANGE' : 'OUT_RANGE');
        const dLo = (s.distLower === null) ? 'n/a' : s.distLower.toFixed(2);
        const dUp = (s.distUpper === null) ? 'n/a' : s.distUpper.toFixed(2);
        const pos = (s.pos01 === null) ? 'n/a' : (s.pos01 * 100).toFixed(1) + '%';
        // Show nearest *pending* orders (more useful than theoretical L1 if already filled)
        let nextBuy = null;
        for (const o of grid.buyLevels?.values?.() || []) {
          if (o && o.status === 'pending' && Number.isFinite(Number(o.price))) {
            if (nextBuy === null || o.price > nextBuy) nextBuy = o.price;
          }
        }
        let nextSell = null;
        for (const o of grid.sellLevels?.values?.() || []) {
          if (o && o.status === 'pending' && Number.isFinite(Number(o.price))) {
            if (nextSell === null || o.price < nextSell) nextSell = o.price;
          }
        }

        const buyRef = (nextBuy !== null) ? nextBuy : (s.basePrice - s.spacing);
        const sellRef = (nextSell !== null) ? nextSell : (s.basePrice + s.spacing);

        const dBuy1 = s.price - buyRef; // <=0 means buy would fill at/under this price
        const dSell1 = sellRef - s.price; // <=0 means sell would fill at/over this price
        const counts = typeof grid.getOpenCounts === 'function' ? grid.getOpenCounts() : { total: 0 };

        const isoUtc = new Date().toISOString();
        const isoP2 = new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString();

        const utcDate = isoUtc.slice(0, 10);
        const utcTime = isoUtc.slice(11, 19);
        const p2Date = isoP2.slice(0, 10);
        const p2Time = isoP2.slice(11, 19);

        const tsUtc = `${utcDate} ${utcTime}Z`;
        const tsP2 = (p2Date === utcDate) ? `${p2Time}+02:00` : `${p2Date} ${p2Time}+02:00`;

        console.log(
          `${tsUtc} | ${tsP2} ` +
          `[GRID_STATUS ${symbol}] ` +
          `p=${s.price.toFixed(2)} base=${s.basePrice.toFixed(2)} spacing=${s.spacing.toFixed(2)} ` +
          `buyNext=${buyRef.toFixed(2)} (p-buy=${dBuy1.toFixed(2)}) ` +
          `sellNext=${sellRef.toFixed(2)} (sell-p=${dSell1.toFixed(2)}) ` +
          `range=[${s.lower.toFixed(2)},${s.upper.toFixed(2)}] ${inRangeStr} ` +
          `dLower=${dLo} dUpper=${dUp} pos=${pos} lvl≈${s.approxLevel} ` +
          `open=${counts.total}`
        );

        // Verbose-compatible indicator line (same key style as momentum bot).
        const ctx = await computeEntryContextForJournal(symbol);
        if (ctx) {
          console.log(
            'ITER_SUMMARY:',
            'bot=grid',
            'iter=' + iterCount,
            'symbol=' + symbol,
            'Started at: ' + tsUtc,
            'momentum=' + Number(ctx.momentum_pct).toFixed(6),
            'sma=' + (ctx.sma == null ? 'null' : Number(ctx.sma).toFixed(2)),
            'smaSlope=' + Number(ctx.sma_slope).toFixed(2),
            'smaSlopePct=' + Number(ctx.sma_slope_pct).toFixed(6),
            'minSlopePct=' + (ctx.min_sma_slope_pct == null ? 'null' : Number(ctx.min_sma_slope_pct).toFixed(6)),
            'atr_pct=' + Number(ctx.atr_pct).toFixed(6),
            'min_atr_pct=' + Number(ctx.min_atr_pct).toFixed(6),
            'regime=' + (ctx.regime ?? 'null'),
            'micro=' + (ctx.micro_regime ?? 'null'),
            'priceNearSMA=' + (ctx.price_near_sma ? 1 : 0),
            'priceAboveSMA=' + (ctx.price_above_sma ? 1 : 0),
            'effective_min_momentum=' + Number(ctx.effective_minimum).toFixed(6),
            'green_run=' + Number(ctx.green_run || 0),
            'openTrades=' + counts.total
          );
        }
      }
    }
  }

  // Rebalance if needed
  if (grid.shouldRebalance()) {
    let blockedByPaperGate = false;
    if (isPaper3100Mode() && cfg.PAPER_REBALANCE_SLOPE_DIST_GATE_ENABLED) {
      const ctx = await computeEntryContextForJournal(symbol);
      const slopeThr = Number(cfg.PAPER_REBALANCE_SLOPE_ABS_THRESHOLD_PCT || 0);
      const distThr = Number(cfg.PAPER_REBALANCE_DIST_ABS_THRESHOLD_PCT || 0);
      const slopePct = Number(ctx?.sma_slope_pct);
      const distPct = Number(ctx?.dist_from_sma_pct);
      const trendStrong = Number.isFinite(slopePct) ? Math.abs(slopePct) > slopeThr : false;
      const farFromSma = Number.isFinite(distPct) ? Math.abs(distPct) > distThr : false;
      blockedByPaperGate = !!(trendStrong && farFromSma);
      rt.lastRebalanceGate = {
        enabled: true,
        blocked: blockedByPaperGate,
        reason: blockedByPaperGate ? 'A1_SLOPE_DIST' : 'ALLOW',
        slope_abs_threshold_pct: slopeThr,
        dist_abs_threshold_pct: distThr,
        sma_slope_pct: Number.isFinite(slopePct) ? slopePct : null,
        dist_from_sma_pct: Number.isFinite(distPct) ? distPct : null,
        ts: nowIso(),
      };
      if (blockedByPaperGate) {
        console.log(`[GRID PAPER ${symbol}] Skip rebalance by A1 gate: slopePct=${Number.isFinite(slopePct) ? slopePct.toFixed(6) : 'null'} distPct=${Number.isFinite(distPct) ? distPct.toFixed(6) : 'null'} thrSlope=${slopeThr.toFixed(6)} thrDist=${distThr.toFixed(6)}`);
        await flushOpenPositionsSnapshotWithPaperGate(true);
      }
    }
    if (blockedByPaperGate) {
      // keep current ladder/inventory untouched
    } else if (isLiveMode()) {
      // Runtime policy:
      // - Always allow BUY ladder re-centering on rebalance.
      // - Never cancel TPSELL orders; only clear the stale BUY ladder.
      // - After rebalance, BUY placement remains guarded by per-level locks / caps.
      try {
        await cancelAllGridOpenOrdersLive(symbol, { keepOrderIds: new Set(), keepClientOrderIds: new Set() });
      } catch (e) {
        console.warn(`[GRID LIVE ${symbol}] cancel before rebalance failed: ${e.message}`);
      }
      await grid.rebalance();
      rt.lastRebalanceTs = Date.now();
    } else {
      await grid.rebalance();
      ensurePaperExtraBuyLevels(grid);
      rt.lastRebalanceTs = Date.now();
      await flushOpenPositionsSnapshotWithPaperGate(true);
    }
  }

  // LIVE: ensure pending BUY grid orders are placed on exchange
  if (isLiveMode()) {
    const isStartupBootstrap = !!rt.allowStartupBuyBootstrap;
    const locks = await getLiveTpSellLocks(symbol);
    const maxLvls = Number(cfg.LEVELS_PER_SIDE || 0);
    for (const [lvl, buy] of grid.buyLevels?.entries?.() || []) {
      if (!buy || buy.status !== 'pending') continue;
      if (buy.exchangeOrderId) continue;

      if (!isStartupBootstrap) {
        // Hard cap: if we already have >=LEVELS_PER_SIDE TPSELL open, do not add exposure.
        if (maxLvls > 0 && locks.tpSellCount >= maxLvls) continue;
        // Per-level lock: do not place a BUY for a level that still has an active TPSELL.
        if (locks.lockedLevels.has(Number(lvl))) continue;
      }

      const clientOrderId = buy.orderId || `GBUY-${symbol}-L${lvl}-${Date.now()}`;
      try {
        const placed = await placeLimitMakerLive({ symbol, side: 'BUY', price: buy.price, qty: buy.qty, clientOrderId });
        buy.orderId = clientOrderId;
        buy.clientOrderId = clientOrderId;
        buy.exchangeOrderId = placed.orderId;
        buy.price = placed.price;
        buy.qty = placed.qty;

        await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'BUY', level: lvl, price: buy.price, qty: buy.qty, orderId: buy.orderId, kind: 'GRID_BUY_LEVEL', exchangeOrderId: buy.exchangeOrderId } });
      } catch (e) {
        if (isInsufficientBalanceError(e)) {
          console.warn(`[GRID LIVE ${symbol}] skip BUY L${lvl}: insufficient free balance`);
        } else {
          console.warn(`[GRID LIVE ${symbol}] place BUY L${lvl} failed: ${e.message}`);
        }
      }
    }
    if (isStartupBootstrap) rt.allowStartupBuyBootstrap = false;
  }

  // Detect fills
  let fills = [];
  if (isLiveMode()) {
    const mexc = getMexcClient();

    // BUY fills
    for (const [lvl, buy] of grid.buyLevels?.entries?.() || []) {
      if (!buy || buy.status !== 'pending' || !buy.exchangeOrderId) continue;
      try {
        const ord = await getOrderLive({ symbol, orderId: buy.exchangeOrderId, clientOrderId: buy.clientOrderId || buy.orderId });
        if (mexc.isOrderFilled(ord)) {
          buy.status = 'filled';
          buy.fillTime = nowIso();
          const fillPrice = mexc.orderAvgFillPrice(ord) ?? buy.price;
          fills.push({ type: 'BUY', level: lvl, orderId: buy.orderId, exchangeOrderId: buy.exchangeOrderId, price: fillPrice, qty: buy.qty, fillTime: buy.fillTime, timestamp: Date.now() });
        }
      } catch (e) {
        // ignore transient
      }
    }

    // SELL fills
    for (const [key, sell] of grid.sellLevels?.entries?.() || []) {
      if (!sell || sell.status !== 'pending' || !sell.exchangeOrderId) continue;
      try {
        const ord = await getOrderLive({ symbol, orderId: sell.exchangeOrderId, clientOrderId: sell.clientOrderId || sell.orderId });
        if (mexc.isOrderFilled(ord)) {
          sell.status = 'filled';
          sell.fillTime = nowIso();
          const fillPrice = mexc.orderAvgFillPrice(ord) ?? sell.price;
          fills.push({ type: 'SELL', level: sell.level, orderId: sell.orderId, exchangeOrderId: sell.exchangeOrderId, buyOrderId: sell.correspondingBuyOrderId, price: fillPrice, qty: sell.qty, fillTime: sell.fillTime, timestamp: Date.now() });
        }
      } catch (e) {
        // ignore transient
      }
    }
  } else {
    fills = await grid.detectFills();
  }

  // Journal any newly scheduled orders since last loop.
  // In LIVE we avoid journaling internal simulated grid orders (BUY/SELL levels) because execution is real.
  if (!isLiveMode()) {
    ensurePaperExtraBuyLevels(grid);
    for (const o of grid.drainNewOrders()) {
      await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: o });
    }
  } else {
    // just drain to avoid accumulating
    grid.drainNewOrders();
  }

  // Process fills
  for (const fill of fills) {
    if (fill.type === 'BUY') {
      // Create OPEN trade
      const trade = grid.createOpenTrade(fill);
      const ctx = await computeEntryContextForJournal(symbol);
      mergeReasonDetails(trade, ctx);
      await persistJournalEvent({
        ts: nowIso(),
        type: 'OPEN',
        trade
      });
      // Schedule TP (internal)
      await grid.placeTpOrder(trade);

      // LIVE: place the TP SELL on exchange
      if (isLiveMode()) {
        // find the TP sell order by correspondingBuyOrderId
        for (const [k, sell] of grid.sellLevels?.entries?.() || []) {
          if (!sell || sell.status !== 'pending') continue;
          if (sell.correspondingBuyOrderId !== trade.id) continue;
          if (sell.exchangeOrderId) continue;

          // MEXC constraint: newClientOrderId must match ^[0-9a-zA-Z_-]{1,32}$.
          // IMPORTANT: clientOrderId must be UNIQUE per grid level.
          // Old scheme used only the batch suffix and caused collisions across L1..L10.
          // Format (<=32 chars): TPSELL_<SYMBOL>_L<level>_<batch13>
          const clientOrderId = makeTpSellClientOrderIdFromBuyId(trade.id);
          try {
            const placed = await placeLimitMakerLive({ symbol, side: 'SELL', price: sell.price, qty: sell.qty, clientOrderId });
            sell.orderId = clientOrderId;
            sell.clientOrderId = clientOrderId;
            sell.exchangeOrderId = placed.orderId;
            sell.price = placed.price;
            sell.qty = placed.qty;


            await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: sell.level, price: sell.price, qty: sell.qty, orderId: sell.orderId, kind: 'GRID_SELL_TP', correspondingBuyOrderId: trade.id, exchangeOrderId: sell.exchangeOrderId } });
          } catch (e) {
            const extra = e?.response?.data ? ` | resp=${JSON.stringify(e.response.data)}` : '';
            console.warn(`[GRID LIVE ${symbol}] place TP SELL maker failed: ${e.message}${extra}`);

            // Robustness: sometimes the exchange accepts the order but we fail to capture orderId (or we hit a duplicate clientId on retry).
            // Try to fetch by clientOrderId and attach it to prevent "BTC without TPSELL".
            try {
              const ord = await getOrderLive({ symbol, orderId: null, clientOrderId });
              const exId = ord?.orderId || ord?.order_id || null;
              if (exId) {
                sell.exchangeOrderId = exId;
                await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: sell.level, price: sell.price, qty: sell.qty, orderId: sell.orderId, kind: 'GRID_SELL_TP', correspondingBuyOrderId: trade.id, exchangeOrderId: sell.exchangeOrderId, note: 'attach_after_place_error' } });
              }
            } catch (_) {}

            // Maker-only policy: do NOT fallback to taker orders.
          }
        }
      }

      // Update open_positions.json promptly so dashboard shows the new open immediately
      await flushOpenPositionsSnapshot(true);
    } else if (fill.type === 'SELL') {
      // Find corresponding open trade
      const openTrade = grid.openTrades.get(fill.buyOrderId);
      if (openTrade) {
        const closeTrade = grid.createCloseTrade(openTrade, fill);

        // Defensive: if maker-only, force exit fee to 0 and recompute net fields.
        if (cfg.MAKER_ONLY) {
          closeTrade.fee_rate_exit = 0;
          const entry = Number(closeTrade.entry_price) || 0;
          const exitp = Number(closeTrade.exit_price) || 0;
          const qty = Number(closeTrade.qty) || 0;
          const feeEntry = entry * qty * (Number(closeTrade.fee_rate_entry) || 0);
          const feeExit = exitp * qty * 0;
          closeTrade.fee_usd_est = feeEntry + feeExit;
          closeTrade.profit_after_fees_est = (Number(closeTrade.profit) || 0) - closeTrade.fee_usd_est;
        }

        await persistJournalEvent({
          ts: closeTrade.close_time_iso || nowIso(),
          type: 'CLOSE',
          trade: closeTrade
        });
        // Replenish buy after sell (internal)
        await grid.placeNewBuyAfterSell(closeTrade);

        // LIVE: place any newly scheduled BUY level that is pending and not yet on exchange
        if (isLiveMode()) {
          const locks = await getLiveTpSellLocks(symbol);
          const maxLvls = Number(cfg.LEVELS_PER_SIDE || 0);
          for (const [lvl, buy] of grid.buyLevels?.entries?.() || []) {
            if (!buy || buy.status !== 'pending') continue;
            if (buy.exchangeOrderId) continue;

            if (maxLvls > 0 && locks.tpSellCount >= maxLvls) continue;
            if (locks.lockedLevels.has(Number(lvl))) continue;

            const clientOrderId = buy.orderId || `GBUY-${symbol}-L${lvl}-${Date.now()}`;
            try {
              const placed = await placeLimitMakerLive({ symbol, side: 'BUY', price: buy.price, qty: buy.qty, clientOrderId });
              buy.orderId = clientOrderId;
              buy.clientOrderId = clientOrderId;
              buy.exchangeOrderId = placed.orderId;
              buy.price = placed.price;
              buy.qty = placed.qty;

              await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'BUY', level: lvl, price: buy.price, qty: buy.qty, orderId: buy.orderId, kind: 'GRID_BUY_LEVEL', exchangeOrderId: buy.exchangeOrderId } });
            } catch (e) {
              if (isInsufficientBalanceError(e)) {
                console.warn(`[GRID LIVE ${symbol}] skip replenish BUY L${lvl}: insufficient free balance`);
              } else {
                console.warn(`[GRID LIVE ${symbol}] place replenish BUY failed: ${e.message}`);
              }
            }
          }
        }

        // Update open_positions.json promptly so dashboard reflects the close
        await flushOpenPositionsSnapshotWithPaperGate(true);
      } else {
        console.warn(`[GRID ${symbol}] SELL fill without matching open trade: ${fill.orderId}`);
      }
    }
  }

  // LIVE: reconcile TP SELL fills from exchange → journal CLOSE
  await reconcileTpFillsFromMyTrades(symbol);

  // LIVE: ensure TPSELL exists for ALL journal-open trades (even after restart).
  // NOTE: grid engines currently do not reconstruct sellLevels from journal on bootstrap, so relying only on grid.sellLevels
  // can leave BTC without an exit order.
  await ensureTpSellsForJournalOpenTrades(symbol);

  // LIVE: retry placing any missing TP SELL orders tracked in grid engine (fills within this runtime)
  if (isLiveMode()) {
    for (const [k, sell] of grid.sellLevels?.entries?.() || []) {
      if (!sell || sell.status !== 'pending') continue;
      if (!sell.correspondingBuyOrderId) continue;
      if (sell.exchangeOrderId) continue;

      const cid = makeTpSellClientOrderIdFromBuyId(sell.correspondingBuyOrderId);
      try {
        const placed = await placeLimitMakerLive({ symbol, side: 'SELL', price: sell.price, qty: sell.qty, clientOrderId: cid });
        sell.orderId = cid;
        sell.clientOrderId = cid;
        sell.exchangeOrderId = placed.orderId;
        sell.price = placed.price;
        sell.qty = placed.qty;
        await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: sell.level, price: sell.price, qty: sell.qty, orderId: sell.orderId, kind: 'GRID_SELL_TP', correspondingBuyOrderId: sell.correspondingBuyOrderId, exchangeOrderId: sell.exchangeOrderId, note: 'retry_place_tp' } });
      } catch (e) {
        // Maker-only: no fallback. But try to attach if the order already exists (duplicate clientId, late response, etc.).
        try {
          const ord = await getOrderLive({ symbol, orderId: null, clientOrderId: cid });
          const exId = ord?.orderId || ord?.order_id || null;
          if (exId) {
            sell.exchangeOrderId = exId;
            sell.clientOrderId = cid;
            await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: { bot: 'grid', symbol, side: 'SELL', level: sell.level, price: sell.price, qty: sell.qty, orderId: cid, kind: 'GRID_SELL_TP', correspondingBuyOrderId: sell.correspondingBuyOrderId, exchangeOrderId: sell.exchangeOrderId, note: 'attach_after_retry_error' } });
          }
        } catch (_) {}
      }
    }
  }

  await maybeForceCancelExpiredTpSells(symbol, grid);
  await maybeRunDisposalModule(symbol);

  // LIVE: reconcile journal opens vs exchange (handles manual closes/cancels)
  await reconcileSymbolWithExchange(symbol);

  // LIVE: write pending open orders snapshot for dashboard
  if (isLiveMode()) {
    await writeOpenOrdersSnapshot(symbol, grid);
  }

  // Rate limiting
  pruneRecentOpens();
}

// Bootstrap
async function bootstrap() {
  console.log('Bootstrapping state from journal...');
  await rebuildStateFromJournal();
  // Reconstruct grid engines from open trades? For now, we skip; paper bot starts fresh each session.
  console.log('State rebuilt. Open trades:', getOpenTradesArray().length);
}

async function writeOpenOrdersSnapshot(symbol, _grid) {
  if (!isLiveMode()) return;
  try {
    // Source of truth: exchange openOrders (so filled/canceled orders disappear immediately).
    const mexc = getMexcClient();
    const oo = await mexc.openOrders({ symbol });
    const arr = Array.isArray(oo) ? oo : (oo?.data || []);

    const out = arr.map((o) => ({
      symbol,
      side: o.side,
      price: Number(o.price),
      qty: Number(o.origQty || o.quantity || o.origQuantity || o.executedQty || o.executedQuantity),
      status: o.status,
      clientOrderId: o.clientOrderId || o.origClientOrderId || null,
      exchangeOrderId: o.orderId || null,
    }));

    await writeJsonFileAtomic(OPEN_ORDERS_PATH, out);

    // Also snapshot balances so dashboard can use exchange-truth to suppress ghost opens.
    try {
      const acc = await mexc.account(2500);
      const balances = acc?.balances || [];
      const pick = (asset) => balances.find((b) => String(b.asset).toUpperCase() === asset);
      const btc = pick('BTC');
      const usdt = pick('USDT');
      await writeJsonFileAtomic(BALANCES_PATH, {
        ts: nowIso(),
        BTC: btc ? { free: Number(btc.free), locked: Number(btc.locked) } : { free: 0, locked: 0 },
        USDT: usdt ? { free: Number(usdt.free), locked: Number(usdt.locked) } : { free: 0, locked: 0 },
      });
    } catch (_) {}
  } catch (e) {
    console.warn(`[GRID LIVE ${symbol}] writeOpenOrdersSnapshot failed: ${e.message}`);
  }
}

// Rate limiting helpers
function pruneRecentOpens() {
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  while (state.recentOpenTimes.length && state.recentOpenTimes[0] < oneHourAgo) {
    state.recentOpenTimes.shift();
  }
}
function noteOpenTimestamp(ts) {
  state.recentOpenTimes.push(ts);
  const ymd = getYmd(new Date(ts));
  const count = state.openCountsByDay.get(ymd) || 0;
  state.openCountsByDay.set(ymd, count + 1);
}
function countOpensLastHour() {
  pruneRecentOpens();
  return state.recentOpenTimes.length;
}
function countOpensTodayUtc() {
  const ymd = getYmd(new Date());
  return state.openCountsByDay.get(ymd) || 0;
}

// Main loop
async function mainLoop() {
  while (!shutdownRequested) {
    try {
      iterCount++;
      for (const symbol of cfg.SYMBOLS) {
        await processSymbol(symbol);
      }
    } catch (err) {
      console.error('Main loop error:', err);
    }
    await sleep(cfg.MONITOR_INTERVAL_MS || 5000);
  }
}

// Signal handlers
process.on('SIGINT', () => { console.log('Shutdown requested'); shutdownRequested = true; });
process.on('SIGTERM', () => { console.log('Shutdown requested'); shutdownRequested = true; });

// Start
(async () => {
  await acquireLock();
  await bootstrap();
  startSnapshotTimer((cfg.SNAPSHOT_INTERVAL_MINUTES || 5) * 60 * 1000);
  await mainLoop();
})();
