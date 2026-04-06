#!/usr/bin/env node

// trading-bot.grid.service.sa.cjs — Grid trading bot (separated from momentum)
// Reuses common SA helpers; no interference with existing momentum bot.

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const axios = require('axios');
require('dotenv').config();

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

// Grid engine
const { GridEngine } = require('./bot/grid/engine.cjs');

// Config
const LABEL = 'grid';
// Self-contained skill folder
const BASE_DIR = path.join(__dirname);
const CONFIG_PATH = path.join(BASE_DIR, 'config_grid.json');
const LOCK_PATH = path.join(BASE_DIR, 'grid.lock');
const OPEN_POSITIONS_PATH = path.join(BASE_DIR, 'open_positions.json');

function loadGridConfig() {
  return readJsonFile(CONFIG_PATH, {});
}

const cfg = loadGridConfig();
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
      lastRebalanceTs: 0
    });
  }
  return state.symbolRuntime.get(symbol);
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

// Symbol processing
async function processSymbol(symbol) {
  const rt = getSymbolRuntime(symbol);
  if (!rt.grid) {
    rt.grid = createGridEngine(symbol);
    await rt.grid.initialize();

    // Journal seed OPEN trades first (so later SELL fills can CLOSE them)
    if (typeof rt.grid.drainSeedOpens === 'function') {
      for (const t of rt.grid.drainSeedOpens()) {
        await persistJournalEvent({ ts: nowIso(), type: 'OPEN', trade: t });
      }
    } else {
      console.warn('[GRID] drainSeedOpens() missing — seed inventory disabled or old engine loaded');
    }

    // Journal initial grid orders (PAPER)
    for (const o of rt.grid.drainNewOrders()) {
      await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: o });
    }

    // Force snapshot now so /api/positions sees these opens immediately
    await flushOpenPositionsSnapshot();
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
      }
    }
  }

  // Rebalance if needed
  if (grid.shouldRebalance()) {
    await grid.rebalance();
    rt.lastRebalanceTs = Date.now();
  }

  // Detect fills
  const fills = await grid.detectFills();

  // Journal any newly scheduled orders since last loop
  for (const o of grid.drainNewOrders()) {
    await persistJournalEvent({ ts: nowIso(), type: 'ORDER', order: o });
  }

  // Process fills
  for (const fill of fills) {
    if (fill.type === 'BUY') {
      // Create OPEN trade
      const trade = grid.createOpenTrade(fill);
      await persistJournalEvent({
        ts: nowIso(),
        type: 'OPEN',
        trade
      });
      // Schedule TP
      await grid.placeTpOrder(trade);
      // Update open_positions.json promptly so dashboard shows the new open immediately
      await flushOpenPositionsSnapshot(true);
    } else if (fill.type === 'SELL') {
      // Find corresponding open trade
      const openTrade = grid.openTrades.get(fill.buyOrderId);
      if (openTrade) {
        const closeTrade = grid.createCloseTrade(openTrade, fill);
        await persistJournalEvent({
          ts: nowIso(),
          type: 'CLOSE',
          trade: closeTrade
        });
        // Replenish buy after sell
        await grid.placeNewBuyAfterSell(closeTrade);
        // Update open_positions.json promptly so dashboard reflects the close
        await flushOpenPositionsSnapshot(true);
      } else {
        console.warn(`[GRID ${symbol}] SELL fill without matching open trade: ${fill.orderId}`);
      }
    }
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