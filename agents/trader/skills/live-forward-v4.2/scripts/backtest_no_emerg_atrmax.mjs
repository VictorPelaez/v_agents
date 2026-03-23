#!/usr/bin/env node
/*
  Backtest (replay) for live-forward-v4.2 strategy using public MEXC spot klines.

  Notes:
  - Uses 1m klines.
  - Approximates intra-candle execution using OHLC:
      LONG: if low<=SL -> SL first; else if high>=TP -> TP.
    (Conservative ordering; when both hit in same candle, assumes SL.)
  - Time resolution: 1 minute.

  Output: summary metrics + CSV-like closes list.
*/

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const mr = require('/root/.openclaw/workspace/agents/trader/market-regime.sa.cjs');
const { evalSlPolicy } = require('/root/.openclaw/workspace/agents/trader/bot/sl_policy.cjs');

// Trend gate comparison:
// - strict: use trendUp as computed (default)
// - no_trend: bypass trendUp requirement (trendUp := true)
const TREND_GATE_MODE = String(process.env.TREND_GATE_MODE || 'strict');

// Defensive filters (backtesting only)
const DEF_SKIP_LOW_VOL_HARD = String(process.env.DEF_SKIP_LOW_VOL_HARD || '0') === '1';
const DEF_SKIP_CHOPPY_LOWVOL = String(process.env.DEF_SKIP_CHOPPY_LOWVOL || '0') === '1';
const DEF_SKIP_CHOPPY_HIGHVOL = String(process.env.DEF_SKIP_CHOPPY_HIGHVOL || '0') === '1';
// If enabled, skip entries whenever microRegime is CHOPPY (any vol regime)
const DEF_SKIP_CHOPPY_ALWAYS = String(process.env.DEF_SKIP_CHOPPY_ALWAYS || '0') === '1';
const DEF_MIN_ATR_PCT_HARD = Number(process.env.DEF_MIN_ATR_PCT_HARD || '0');
// If set (>0), use a wider emergency SL in HIGH_VOL+CHOPPY (reduces wick-triggered emergencies)
const DEF_EMERGENCY_SL_PCT_HV_CHOPPY = Number(process.env.DEF_EMERGENCY_SL_PCT_HV_CHOPPY || '0');
// If enabled, do NOT trigger wick-based emergency SL when HIGH_VOL+CHOPPY (let soft SL policy handle it)
const DEF_DISABLE_EMERGENCY_WICK_HV_CHOPPY = String(process.env.DEF_DISABLE_EMERGENCY_WICK_HV_CHOPPY || '0') === '1';
// If enabled, do NOT trigger wick-based emergency SL in HIGH_VOL (any micro regime)
const DEF_DISABLE_EMERGENCY_WICK_HIGHVOL = String(process.env.DEF_DISABLE_EMERGENCY_WICK_HIGHVOL || '0') === '1';
// If enabled, disable wick-based emergency SL always (all regimes)
const DEF_DISABLE_EMERGENCY_WICK_ALWAYS = String(process.env.DEF_DISABLE_EMERGENCY_WICK_ALWAYS || '0') === '1';
// Skip entries when ATR% is ABOVE this ceiling (entry only)
const DEF_MAX_ATR_PCT = Number(process.env.DEF_MAX_ATR_PCT || '0');

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function pct(x, digits = 3) { return (x * 100).toFixed(digits) + '%'; }

function percentile(arr, q) {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  const a = arr.filter(Number.isFinite).slice().sort((x, y) => x - y);
  if (!a.length) return 0;
  const p = Math.min(1, Math.max(0, q));
  const idx = (a.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return a[lo];
  const w = idx - lo;
  return a[lo] * (1 - w) + a[hi] * w;
}

async function fetchKlinesMexc(symbol, startTimeMs, endTimeMs) {
  const out = [];
  let cur = startTimeMs;
  let guard = 0;

  while (cur < endTimeMs && guard++ < 1000) {
    const url = new URL('https://api.mexc.com/api/v3/klines');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('interval', '1m');
    url.searchParams.set('limit', '1000');
    url.searchParams.set('startTime', String(cur));
    url.searchParams.set('endTime', String(endTimeMs));

    const res = await fetch(url.toString(), { headers: { 'accept': 'application/json' } });
    if (!res.ok) throw new Error(`MEXC klines ${symbol} HTTP ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) break;

    for (const k of data) out.push(k);

    const lastOpen = Number(data[data.length - 1]?.[0]);
    if (!Number.isFinite(lastOpen)) break;
    const next = lastOpen + 60_000;
    if (next <= cur) break;
    cur = next;

    // be gentle with rate limits
    await sleep(120);
  }

  // Ensure strictly increasing by open time
  out.sort((a, b) => Number(a[0]) - Number(b[0]));
  // Dedup by open time
  const dedup = [];
  let lastTs = null;
  for (const k of out) {
    const ts = Number(k[0]);
    if (!Number.isFinite(ts)) continue;
    if (ts === lastTs) continue;
    dedup.push(k);
    lastTs = ts;
  }
  return dedup;
}

async function fetchKlinesBinance(symbol, startTimeMs, endTimeMs) {
  const out = [];
  let cur = startTimeMs;
  let guard = 0;

  while (cur < endTimeMs && guard++ < 2000) {
    const url = new URL('https://api.binance.com/api/v3/klines');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('interval', '1m');
    url.searchParams.set('limit', '1000');
    url.searchParams.set('startTime', String(cur));
    url.searchParams.set('endTime', String(endTimeMs));

    const res = await fetch(url.toString(), { headers: { 'accept': 'application/json' } });
    if (!res.ok) throw new Error(`BINANCE klines ${symbol} HTTP ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) break;

    for (const k of data) out.push(k);

    const lastOpen = Number(data[data.length - 1]?.[0]);
    if (!Number.isFinite(lastOpen)) break;
    const next = lastOpen + 60_000;
    if (next <= cur) break;
    cur = next;

    await sleep(120);
  }

  out.sort((a, b) => Number(a[0]) - Number(b[0]));
  const dedup = [];
  let lastTs = null;
  for (const k of out) {
    const ts = Number(k[0]);
    if (!Number.isFinite(ts)) continue;
    if (ts === lastTs) continue;
    dedup.push(k);
    lastTs = ts;
  }
  return dedup;
}

function loadConfig(configPath) {
  const cfgPath = path.resolve(configPath);
  const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
  return cfg;
}

function feeEstimate(entry, exit, qty, feeRateEntry, feeRateExit) {
  const frIn = Number.isFinite(feeRateEntry) ? feeRateEntry : 0;
  const frOut = Number.isFinite(feeRateExit) ? feeRateExit : 0;
  return (entry * qty) * frIn + (exit * qty) * frOut;
}

function nowIsoFromMs(ms) {
  return new Date(ms).toISOString();
}

function backtestSymbolSeries({ symbol, klines, cfg, feeRateMaker, feeRateTaker, debugFunnel }) {
  // Partial TP settings (fixed for this experiment)
  const TP1_PCT = Number(cfg.TP1_PCT ?? 0.005);     // +0.5%
  const TP1_FRAC = Number(cfg.TP1_FRAC ?? 0.5);     // 50%
  const TIME_STOP_ASSUME_MAKER = (cfg.TIME_STOP_ASSUME_MAKER !== undefined) ? !!cfg.TIME_STOP_ASSUME_MAKER : true;

  const SMA_WINDOW = cfg.SMA_WINDOW ?? 60;

  // Extra indicator filters (match live service; when USE_* is false, they do nothing)
  const USE_ADX_FILTER = !!cfg.USE_ADX_FILTER;
  const ADX_WINDOW = Number(cfg.ADX_WINDOW ?? 14);
  const MIN_ADX = Number(cfg.MIN_ADX ?? 0);
  const ADX_REQUIRE_DI_BULL = (cfg.ADX_REQUIRE_DI_BULL !== undefined) ? !!cfg.ADX_REQUIRE_DI_BULL : true;

  const USE_DONCHIAN_FILTER = !!cfg.USE_DONCHIAN_FILTER;
  const DONCHIAN_N = Number(cfg.DONCHIAN_N ?? 20);

  const USE_SUPERTREND_FILTER = !!cfg.USE_SUPERTREND_FILTER;
  const SUPERTREND_ATR_WINDOW = Number(cfg.SUPERTREND_ATR_WINDOW ?? 10);
  const SUPERTREND_MULT = Number(cfg.SUPERTREND_MULT ?? 3.0);
  const ATR_WINDOW = cfg.ATR_WINDOW ?? 14;
  const MIN_ATR_PCT = cfg.MIN_ATR_PCT ?? 0;

  const ATR_ADAPTIVE_ENABLED = !!cfg.ATR_ADAPTIVE_ENABLED;
  const ATR_ADAPTIVE_PCTL = Number(cfg.ATR_ADAPTIVE_PCTL ?? 0.6);
  const ATR_ADAPTIVE_WINDOW = Number(cfg.ATR_ADAPTIVE_WINDOW ?? 240);
  const ATR_ADAPTIVE_MIN_SAMPLES = Number(cfg.ATR_ADAPTIVE_MIN_SAMPLES ?? 60);

  const MIN_SMA_SLOPE = Number(cfg.MIN_SMA_SLOPE ?? 0);
  const MIN_SMA_SLOPE_PCT = (cfg.MIN_SMA_SLOPE_PCT !== undefined && cfg.MIN_SMA_SLOPE_PCT !== null)
    ? Number(cfg.MIN_SMA_SLOPE_PCT)
    : null;

  const MIN_MOMENTUM_PCT = Number(cfg.MIN_MOMENTUM_PCT ?? 0.0005);
  const MAX_MOMENTUM_PCT = Number(cfg.MAX_MOMENTUM_PCT ?? 1);

  const MIN_SLOPE_NORM = Number(cfg.MIN_SLOPE_NORM ?? 0);
  const MIN_SLOPE_NORM_BY_SYMBOL = cfg.MIN_SLOPE_NORM_BY_SYMBOL || {};
  const minSlopeNormEff = Number.isFinite(Number(MIN_SLOPE_NORM_BY_SYMBOL[symbol]))
    ? Number(MIN_SLOPE_NORM_BY_SYMBOL[symbol])
    : MIN_SLOPE_NORM;

  const minCandleBody = Number(cfg.MIN_BODY_CANDLE ?? 0.5);
  // Important: if EXPLOSIVE_CANDLE_PCT is missing, default to a sensible non-zero value.
  // A 0 default would block all entries (bodyPct must be >= explosiveCandlePct).
  const explosiveCandlePct = Number(cfg.EXPLOSIVE_CANDLE_PCT ?? 0.003);
  const MIN_VOLUME = Number(cfg.MIN_VOLUME ?? 0);
  const smaTol = Number(cfg.SMA_TOLERANCE ?? 0.001);

  const k_tp = Number(cfg.K_TP ?? 2.0);
  const k_sl = Number(cfg.K_SL ?? 1.2);
  const MIN_SL_PCT = Number(cfg.MIN_SL_PCT ?? 0);
  const MIN_SL_USD = Number(cfg.MIN_SL_USD ?? 0);

  const SL_POLICY = String(cfg.SL_POLICY ?? 'CLASSIC').toUpperCase();
  const EMERGENCY_SL_PCT = Number(cfg.EMERGENCY_SL_PCT ?? 0);
  const SL_CONFIRM_SECONDS = Number(cfg.SL_CONFIRM_SECONDS ?? 0);

  const tradeUSD = Number(cfg.TRADE_USD ?? 100);
  const timeStopMinutes = Number(cfg.TIME_STOP_MINUTES ?? 35);
  const minHoldS = Number(cfg.MIN_HOLD_SECONDS ?? 120);

  // Experimental: ignore SL closes before this age (minutes). 0 disables.
  // Parity with live: NO_SL_BEFORE_MINUTES (live config)
  const DEF_NO_SL_BEFORE_MIN = Number(process.env.DEF_NO_SL_BEFORE_MIN || cfg.NO_SL_BEFORE_MINUTES || 0);

  // Experimental: late-window smaller TP target (e.g. TP/3 or TP/4) to exit before time_stop.
  const DEF_LATE_TP_DIV_ENABLED = String(process.env.DEF_LATE_TP_DIV_ENABLED || '0') === '1';
  const DEF_LATE_TP_DIV = Number(process.env.DEF_LATE_TP_DIV || 0);
  const DEF_LATE_TP_DIV_START_MIN = Number(process.env.DEF_LATE_TP_DIV_START_MIN || 25);
  const DEF_LATE_TP_DIV_END_MIN = Number(process.env.DEF_LATE_TP_DIV_END_MIN || 35);
  const DEF_LATE_TP_ASSUME_MAKER = String(process.env.DEF_LATE_TP_ASSUME_MAKER || '1') === '1';

  const FAIL_FAST_ENABLED = !!cfg.FAIL_FAST_ENABLED;
  const FAIL_FAST_MINUTES = Number(cfg.FAIL_FAST_MINUTES ?? 0);
  const FAIL_FAST_MIN_MFE_PCT = Number(cfg.FAIL_FAST_MIN_MFE_PCT ?? 0);
  const FAIL_FAST_MIN_DD_PCT = Number(cfg.FAIL_FAST_MIN_DD_PCT ?? 0);
  const EXIT_ON_SUPERTREND_FLIP = !!cfg.EXIT_ON_SUPERTREND_FLIP;
  const ST_FLIP_REQUIRE_ADVANTAGE = (cfg.ST_FLIP_REQUIRE_ADVANTAGE !== undefined) ? !!cfg.ST_FLIP_REQUIRE_ADVANTAGE : true;
  const ST_FLIP_MIN_MFE_PCT = Number(cfg.ST_FLIP_MIN_MFE_PCT ?? 0.001);

  // If true, model early discretionary exits as maker (fee=0) instead of taker.
  const EARLY_EXIT_ASSUME_MAKER = !!cfg.EARLY_EXIT_ASSUME_MAKER;

  // Experimental: slopeNorm time-window exit (to avoid SL, only if it pays at least taker fees)
  // Exit on candle close during a time window if current slopeNorm drops below a threshold.
  // Also requires net >= 0 after fees (models taker exit unless EARLY_EXIT_ASSUME_MAKER=true).
  const DEF_EXIT_SLOPENORM = String(process.env.DEF_EXIT_SLOPENORM || '0') === '1';
  const DEF_EXIT_SLOPENORM_THR = Number(process.env.DEF_EXIT_SLOPENORM_THR || 0.065);
  const DEF_EXIT_SLOPENORM_START_MIN = Number(process.env.DEF_EXIT_SLOPENORM_START_MIN || 13);
  const DEF_EXIT_SLOPENORM_END_MIN = Number(process.env.DEF_EXIT_SLOPENORM_END_MIN || 25);
  // Only trigger if MFE% is below this ("no continuation"). 0 disables the filter.
  const DEF_EXIT_SLOPENORM_MFE_MAX_PCT = Number(process.env.DEF_EXIT_SLOPENORM_MFE_MAX_PCT || 0);
  // Profit requirement expressed as "cover fees" multiple:
  // - 1 => net >= 0 (covers fees 1x)
  // - 2 => net >= fee (covers fees 2x)
  const DEF_EXIT_SLOPENORM_FEE_COVER_MULT = Number(process.env.DEF_EXIT_SLOPENORM_FEE_COVER_MULT || 1);
  // Optional: require donchian failure at exit time (close back below donchPrevHigh)
  const DEF_EXIT_SLOPENORM_REQUIRE_DONCH_FAIL = String(process.env.DEF_EXIT_SLOPENORM_REQUIRE_DONCH_FAIL || '0') === '1';
  // Donchian window for EXIT logic (entry still uses cfg.DONCHIAN_N)
  const DEF_EXIT_DONCHIAN_N = Number(process.env.DEF_EXIT_DONCHIAN_N || 20);

  // Experimental: 5-minute reversal exit
  // Close the trade at ~T+5m if BOTH:
  //  - Donchian prev-high (last N closed candles) has decreased vs entry snapshot, AND
  //  - SMA slope (SMA_WINDOW; typically 20) has flipped down (slope < 0)
  const DEF_EXIT_REVERSAL_5M = String(process.env.DEF_EXIT_REVERSAL_5M || '0') === '1';
  const DEF_EXIT_REVERSAL_5M_MINUTES = Number(process.env.DEF_EXIT_REVERSAL_5M_MINUTES || 5);
  // If >0, only trigger the reversal-exit if MFE% is below this threshold (i.e., "no continuation").
  const DEF_EXIT_REVERSAL_MFE_MAX_PCT = Number(process.env.DEF_EXIT_REVERSAL_MFE_MAX_PCT || 0);

  function computeRegimeFromWindow(window) {
    try {
      const closesSeries = mr.getClosedCloses(window);
      if (!Array.isArray(closesSeries) || closesSeries.length < Math.max(35, (SMA_WINDOW + 2))) return null;
      const { sma, smaPrev } = mr.computeSmaPair(closesSeries, SMA_WINDOW);
      const smaSlopeAbs = (sma != null && smaPrev != null) ? (sma - smaPrev) : 0;
      const lastClose = closesSeries[closesSeries.length - 1];
      const priceAboveSMA = (sma != null) ? (lastClose > sma) : true;
      const { atrPct } = mr.computeAtr(window, ATR_WINDOW);
      const atrPctNum = Number.isFinite(atrPct) ? atrPct : 0;
      const realizedVol = mr.computeRealizedVol(closesSeries, 30);
      return mr.detectMarketRegime({ atrPct: atrPctNum, realizedVol, smaSlope: smaSlopeAbs, lastClose, priceAboveSma: priceAboveSMA });
    } catch {
      return null;
    }
  }

  /** open trade or null */
  // Debug funnel (local, ES module safe)
  const useDebugFunnel = String(process.env.DEBUG_FUNNEL || '0') === '1';
  const funnel = useDebugFunnel ? {
    total: 0,
    atr: 0,
    vol: 0,
    body: 0,
    explosive: 0,
    momentum: 0,
    slopeAbs: 0,
    trend: 0,
    donch: 0,
    adx: 0,
    diBull: 0,
    st: 0,
    weakBody: 0,
    weakOpen: 0,
    regime: 0,
    curLow: 0,
    netProfit: 0,
    passed: 0,
    blockedHour: 0
  } : null;

  let open = null;
  const closes = [];
  // Adaptive ATR state (per symbol series)
  const rt = { atrHistory: [], lastAtrCandleTs: null };

  // Need room for current candle
  for (let idx = Math.max(SMA_WINDOW + 5, ATR_WINDOW + 10); idx < klines.length - 2; idx++) {
    if (debugFunnel) debugFunnel.total++;
    if (funnel) funnel.total++;
    const sliceStart = Math.max(0, idx - 500);
    const window = klines.slice(sliceStart, idx + 2); // includes current candle at idx+1

    const lastClosed = window[window.length - 2];
    const current = window[window.length - 1];

    const candle = {
      ts: Number(lastClosed[0]),
      open: Number(lastClosed[1]),
      high: Number(lastClosed[2]),
      low: Number(lastClosed[3]),
      close: Number(lastClosed[4]),
      volume: Number(lastClosed[5]),
      quoteVolume: Number(lastClosed[7]),
    };

    if (![candle.ts, candle.open, candle.high, candle.low, candle.close].every(Number.isFinite)) continue;

    // --- 1) monitor open trade on the *current* candle range ---
    if (open) {
      const ageS = (Number(current[0]) - open.openedAtMs) / 1000;
      if (ageS >= minHoldS) {
        const curHigh = Number(current[2]);
        const curLow = Number(current[3]);
        const curClose = Number(current[4]);

        // Track MFE using candle high
        if (open._maxPriceSinceOpen == null || !Number.isFinite(open._maxPriceSinceOpen)) open._maxPriceSinceOpen = open.entryPrice;
        if (Number.isFinite(curHigh) && Number.isFinite(open._maxPriceSinceOpen)) {
          open._maxPriceSinceOpen = Math.max(open._maxPriceSinceOpen, curHigh);
        }
        // Track MAE using candle low
        if (open._minPriceSinceOpen == null || !Number.isFinite(open._minPriceSinceOpen)) open._minPriceSinceOpen = open.entryPrice;
        if (Number.isFinite(curLow) && Number.isFinite(open._minPriceSinceOpen)) {
          open._minPriceSinceOpen = Math.min(open._minPriceSinceOpen, curLow);
        }

        let didClose = false;
        if (ageS > timeStopMinutes * 60) {
          // time stop closes at candle close
          const exit = Number.isFinite(curClose) ? curClose : open.entryPrice;
          const remQty = open.qtyA + open.qtyB;
          const gross = open.realizedGross + remQty * (exit - open.entryPrice);
          const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, TIME_STOP_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
          closes.push({
            symbol,
            entry_meta: open.entry_meta,
            open_time_iso: nowIsoFromMs(open.openedAtMs),
            close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
            entry: open.entryPrice,
            exit,
            close_reason: open.tp1Hit ? 'time_stop_after_TP1' : 'time_stop',
            gross,
            net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
            tp1_hit: !!open.tp1Hit,
            duration_s: ageS,
            mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
            mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
          });
          open = null;
          didClose = true;
        }

        if (!didClose) {
          // Fail-fast exit (after minHold)
          if (FAIL_FAST_ENABLED && FAIL_FAST_MINUTES > 0 && FAIL_FAST_MIN_MFE_PCT > 0) {
            const ageMin = ageS / 60;
            const mfePct = (Number.isFinite(open._maxPriceSinceOpen) && Number.isFinite(open.entryPrice) && open.entryPrice > 0)
              ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice)
              : 0;
            const inProfitNow = Number.isFinite(curClose) ? (curClose > open.entryPrice) : false;
            const ddPct = (Number.isFinite(curClose) && Number.isFinite(open.entryPrice) && open.entryPrice > 0)
              ? ((open.entryPrice - curClose) / open.entryPrice)
              : 0;
            const ddOk = (Number.isFinite(FAIL_FAST_MIN_DD_PCT) && FAIL_FAST_MIN_DD_PCT > 0)
              ? (ddPct >= FAIL_FAST_MIN_DD_PCT)
              : true;

            if (ageMin >= FAIL_FAST_MINUTES && !inProfitNow && mfePct < FAIL_FAST_MIN_MFE_PCT && ddOk) {
              const exit = Number.isFinite(curClose) ? curClose : open.entryPrice;
              const gross = open.qty * (exit - open.entryPrice);
              const fee = feeEstimate(open.entryPrice, exit, open.qty, feeRateMaker, EARLY_EXIT_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
              closes.push({
                symbol,
                entry_meta: open.entry_meta,
                open_time_iso: nowIsoFromMs(open.openedAtMs),
                close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
                entry: open.entryPrice,
                exit,
                close_reason: 'fail_fast_mfe',
                gross,
                net: gross - fee,
                duration_s: ageS,
                mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
                mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
              });
              open = null;
              didClose = true;
            }
          }
        }

        if (!didClose) {
          // Experimental: slopeNorm time-window exit (after minHold)
          if (DEF_EXIT_SLOPENORM) {
            const ageMin = ageS / 60;
            if (ageMin >= DEF_EXIT_SLOPENORM_START_MIN && ageMin <= DEF_EXIT_SLOPENORM_END_MIN) {
              const regNow = computeRegimeFromWindow(window);
              const slopeNormNow = regNow?.slopeNorm;

              let donchFail = true;
              if (DEF_EXIT_SLOPENORM_REQUIRE_DONCH_FAIL && USE_DONCHIAN_FILTER) {
                try {
                  const closed = window.slice(0, Math.max(0, window.length - 1));
                  const nExit = (Number.isFinite(DEF_EXIT_DONCHIAN_N) && DEF_EXIT_DONCHIAN_N > 1) ? DEF_EXIT_DONCHIAN_N : DONCHIAN_N;
                  if (closed.length >= nExit + 1) {
                    const slice = closed.slice(-(nExit + 1), -1);
                    const donchPrevHighNow = Math.max(...slice.map(k => Number(k?.[2])).filter(Number.isFinite));
                    const c = Number.isFinite(curClose) ? curClose : null;
                    donchFail = (Number.isFinite(donchPrevHighNow) && Number.isFinite(c)) ? (c < donchPrevHighNow) : true;
                  }
                } catch {
                  donchFail = true;
                }
              }

              if (Number.isFinite(slopeNormNow) && slopeNormNow < DEF_EXIT_SLOPENORM_THR && donchFail) {
                const exit = Number.isFinite(curClose) ? curClose : open.entryPrice;
                const remQty = open.qtyA + open.qtyB;
                const gross = open.realizedGross + remQty * (exit - open.entryPrice);
                const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, EARLY_EXIT_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
                const net = open.realizedNet + (remQty * (exit - open.entryPrice) - fee);

                const mfePct = (Number.isFinite(open._maxPriceSinceOpen) && Number.isFinite(open.entryPrice) && open.entryPrice > 0)
                  ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice)
                  : 0;
                const mfeOk = (Number.isFinite(DEF_EXIT_SLOPENORM_MFE_MAX_PCT) && DEF_EXIT_SLOPENORM_MFE_MAX_PCT > 0)
                  ? (mfePct <= DEF_EXIT_SLOPENORM_MFE_MAX_PCT)
                  : true;

                const mult = Number.isFinite(DEF_EXIT_SLOPENORM_FEE_COVER_MULT) ? DEF_EXIT_SLOPENORM_FEE_COVER_MULT : 1;
                const netMin = (mult <= 1) ? 0 : (fee * (mult - 1));

                // Only take the early exit if it satisfies MFE filter and fee-coverage target.
                if (mfeOk && net >= netMin) {
                  closes.push({
                    symbol,
                    entry_meta: open.entry_meta,
                    open_time_iso: nowIsoFromMs(open.openedAtMs),
                    close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
                    entry: open.entryPrice,
                    exit,
                    close_reason: 'early_exit_slopenorm',
                    gross,
                    net,
                    tp1_hit: !!open.tp1Hit,
                    duration_s: ageS,
                    mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
                    mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
                  });
                  open = null;
                  didClose = true;
                }
              }
            }
          }
        }

        if (!didClose) {
          // Experimental 5m reversal exit (after minHold)
          if (DEF_EXIT_REVERSAL_5M && DEF_EXIT_REVERSAL_5M_MINUTES > 0 && !open._rev5mChecked) {
            const ageMin = ageS / 60;
            if (ageMin >= DEF_EXIT_REVERSAL_5M_MINUTES) {
              open._rev5mChecked = true;
              // Donchian breakout failure NOW: price closes back below donchPrevHigh
              let donchPrevHighNow = null;
              try {
                const closed = window.slice(0, Math.max(0, window.length - 1));
                if (closed.length >= DONCHIAN_N + 1) {
                  const slice = closed.slice(-(DONCHIAN_N + 1), -1);
                  donchPrevHighNow = Math.max(...slice.map(k => Number(k?.[2])).filter(Number.isFinite));
                  if (!Number.isFinite(donchPrevHighNow)) donchPrevHighNow = null;
                }
              } catch { donchPrevHighNow = null; }

              // SMA slope norm NOW (SMA_WINDOW, typically 20): |slope| / ATR_abs
              let smaSlopeAbsNow = 0;
              let slopeNormNow = 0;
              try {
                const closesNow = mr.getClosedCloses(window);
                const { sma: smaNow, smaPrev: smaPrevNow } = mr.computeSmaPair(closesNow, SMA_WINDOW);
                smaSlopeAbsNow = (smaNow != null && smaPrevNow != null) ? (smaNow - smaPrevNow) : 0;
                const { atrPct: atrPctNow } = mr.computeAtr(window, ATR_WINDOW);
                const atrAbsNow = (Number.isFinite(atrPctNow) ? atrPctNow : 0) * (Number.isFinite(curClose) && curClose > 0 ? curClose : 1);
                slopeNormNow = atrAbsNow > 0 ? (Math.abs(smaSlopeAbsNow) / atrAbsNow) : 0;
              } catch { smaSlopeAbsNow = 0; slopeNormNow = 0; }

              const minSlopeNormEffEntry = Number(open.entry_meta?.minSlopeNormEff ?? 0);
              const donchFailed = (donchPrevHighNow != null && Number.isFinite(curClose)) ? (curClose < donchPrevHighNow) : false;
              const slopeNormFailed = (Number.isFinite(minSlopeNormEffEntry) && minSlopeNormEffEntry > 0) ? (slopeNormNow < minSlopeNormEffEntry) : false;

              const redNow = Number.isFinite(curClose) ? (curClose < open.entryPrice) : false;
              const mfePctNow = (Number.isFinite(open._maxPriceSinceOpen) && Number.isFinite(open.entryPrice) && open.entryPrice > 0)
                ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice)
                : 0;
              const mfeOk = (Number.isFinite(DEF_EXIT_REVERSAL_MFE_MAX_PCT) && DEF_EXIT_REVERSAL_MFE_MAX_PCT > 0)
                ? (mfePctNow < DEF_EXIT_REVERSAL_MFE_MAX_PCT)
                : true;

              if (donchFailed && slopeNormFailed && redNow && mfeOk) {
                const exit = Number.isFinite(curClose) ? curClose : open.entryPrice;
                const remQty = open.qtyA + open.qtyB;
                const gross = open.realizedGross + remQty * (exit - open.entryPrice);
                const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, EARLY_EXIT_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
                closes.push({
                  symbol,
                  entry_meta: open.entry_meta,
                  open_time_iso: nowIsoFromMs(open.openedAtMs),
                  close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
                  entry: open.entryPrice,
                  exit,
                  close_reason: 'rev5m_donchFail_slopeNormFail',
                  gross,
                  net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
                  tp1_hit: !!open.tp1Hit,
                  duration_s: ageS,
                  rev5m: { donchPrevHighNow, curClose, minSlopeNormEffEntry, smaSlopeAbsNow, slopeNormNow },
                });
                open = null;
                didClose = true;
              }
            }
          }
        }

        if (!didClose) {
          // Exit on supertrend flip
          if (EXIT_ON_SUPERTREND_FLIP) {
            const stNow = mr.computeSupertrend(window, SUPERTREND_ATR_WINDOW, SUPERTREND_MULT);
            if (stNow && stNow.dir === -1) {
              const mfePct = (Number.isFinite(open._maxPriceSinceOpen) && Number.isFinite(open.entryPrice) && open.entryPrice > 0)
                ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice)
                : 0;
              const inProfitNow = Number.isFinite(curClose) ? (curClose > open.entryPrice) : false;
              const allow = (!ST_FLIP_REQUIRE_ADVANTAGE)
                ? true
                : (inProfitNow || (Number.isFinite(mfePct) && mfePct >= ST_FLIP_MIN_MFE_PCT));

              if (allow) {
                const exit = Number.isFinite(curClose) ? curClose : open.entryPrice;
                const gross = open.qty * (exit - open.entryPrice);
                const fee = feeEstimate(open.entryPrice, exit, open.qty, feeRateMaker, EARLY_EXIT_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
                closes.push({
                  symbol,
                  entry_meta: open.entry_meta,
                  open_time_iso: nowIsoFromMs(open.openedAtMs),
                  close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
                  entry: open.entryPrice,
                  exit,
                  close_reason: 'st_flip',
                  gross,
                  net: gross - fee,
                  duration_s: ageS,
                  mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
                  mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
                });
                open = null;
                didClose = true;
              }
            }
          }
        }

        // TP1 partial (wick-based, maker exit)
        if (!didClose && !open.tp1AsTp && !open.tp1Hit && open.qtyA > 0 && Number.isFinite(curHigh) && curHigh >= open.tp1) {
          const exit = open.tp1;
          const grossA = open.qtyA * (exit - open.entryPrice);
          const feeA = feeEstimate(open.entryPrice, exit, open.qtyA, feeRateMaker, feeRateMaker);
          open.realizedGross += grossA;
          open.realizedNet += (grossA - feeA);
          open.tp1Hit = true;
          open.qtyA = 0;
        }

        if (!didClose) {
          // Emergency SL (airbag) based on wick (conservative)
          let emerg = open.stopLossEmergency;
          if (DEF_DISABLE_EMERGENCY_WICK_ALWAYS) emerg = null;
          const regNow = (DEF_EMERGENCY_SL_PCT_HV_CHOPPY > 0 || DEF_DISABLE_EMERGENCY_WICK_HV_CHOPPY || DEF_DISABLE_EMERGENCY_WICK_HIGHVOL) ? computeRegimeFromWindow(window) : null;
          const isHighVol = !!(regNow && regNow.volRegime === 'HIGH_VOL');
          const isHvChoppy = !!(regNow && regNow.volRegime === 'HIGH_VOL' && regNow.microRegime === 'CHOPPY');

          if (DEF_DISABLE_EMERGENCY_WICK_HIGHVOL && isHighVol) {
            emerg = null;
          } else if (DEF_DISABLE_EMERGENCY_WICK_HV_CHOPPY && isHvChoppy) {
            emerg = null;
          } else if (Number.isFinite(DEF_EMERGENCY_SL_PCT_HV_CHOPPY) && DEF_EMERGENCY_SL_PCT_HV_CHOPPY > 0 && isHvChoppy) {
            emerg = open.entryPrice * (1 - DEF_EMERGENCY_SL_PCT_HV_CHOPPY);
          }
          if (emerg != null && Number.isFinite(curLow) && curLow <= emerg) {
            const exit = emerg;
            const remQty = open.qtyA + open.qtyB;
            const gross = open.realizedGross + remQty * (exit - open.entryPrice);
            const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, feeRateTaker);
            closes.push({
              symbol,
              entry_meta: open.entry_meta,
              open_time_iso: nowIsoFromMs(open.openedAtMs),
              close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
              entry: open.entryPrice,
              exit,
              close_reason: open.tp1Hit ? 'emergency_sl_after_TP1' : 'emergency_sl',
              gross,
              net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
              tp1_hit: !!open.tp1Hit,
              duration_s: ageS,
              mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
              mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
            });
            open = null;
            didClose = true;
          }
        }

        if (!didClose) {
          // Soft/classic SL policy using candle CLOSE as market proxy.
          const curClose2 = Number(current[4]);
          const slRes = evalSlPolicy({
            slPolicy: SL_POLICY,
            market: Number.isFinite(curClose2) ? curClose2 : open.entryPrice,
            slClassic: open.stopLossClassic,
            slEmergency: open.stopLossEmergency,
            nowMs: Number(current[0]),
            breachStartMs: open._slBreachStartMs,
            slConfirmSeconds: SL_CONFIRM_SECONDS,
          });
          open._slBreachStartMs = slRes.breachStartMs;

          if (slRes.closeReason === 'SL') {
            const ageMin = ageS / 60;
            if (Number.isFinite(DEF_NO_SL_BEFORE_MIN) && DEF_NO_SL_BEFORE_MIN > 0 && ageMin < DEF_NO_SL_BEFORE_MIN) {
              // Ignore early SL attempts; keep the trade open.
            } else {
            const exit = Number.isFinite(curClose2) ? curClose2 : open.stopLossClassic;
            const remQty = open.qtyA + open.qtyB;
            const gross = open.realizedGross + remQty * (exit - open.entryPrice);
            const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, feeRateTaker);
            closes.push({
              symbol,
              entry_meta: open.entry_meta,
              open_time_iso: nowIsoFromMs(open.openedAtMs),
              close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
              entry: open.entryPrice,
              exit,
              close_reason: open.tp1Hit ? 'SL_after_TP1' : 'SL',
              gross,
              net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
              tp1_hit: !!open.tp1Hit,
              duration_s: ageS,
            });
            open = null;
            didClose = true;
          }
        }

        }

        if (!didClose) {
          // Experimental: late-window smaller TP target (wick-based)
          if (DEF_LATE_TP_DIV_ENABLED && Number.isFinite(DEF_LATE_TP_DIV) && DEF_LATE_TP_DIV > 1) {
            const ageMin = ageS / 60;
            if (ageMin >= DEF_LATE_TP_DIV_START_MIN && ageMin <= DEF_LATE_TP_DIV_END_MIN) {
              const lateTp = open.entryPrice + (open.takeProfit - open.entryPrice) / DEF_LATE_TP_DIV;
              if (Number.isFinite(curHigh) && Number.isFinite(lateTp) && lateTp < open.takeProfit && curHigh >= lateTp) {
                const exit = lateTp;
                const remQty = open.qtyA + open.qtyB;
                const gross = open.realizedGross + remQty * (exit - open.entryPrice);
                const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, DEF_LATE_TP_ASSUME_MAKER ? feeRateMaker : feeRateTaker);
                closes.push({
                  symbol,
                  entry_meta: open.entry_meta,
                  open_time_iso: nowIsoFromMs(open.openedAtMs),
                  close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
                  entry: open.entryPrice,
                  exit,
                  close_reason: `late_tp_div_${DEF_LATE_TP_DIV}`,
                  gross,
                  net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
                  tp1_hit: !!open.tp1Hit,
                  duration_s: ageS,
                  mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
                  mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
                });
                open = null;
                didClose = true;
              }
            }
          }
        }

        if (!didClose) {
          // TP2 check (wick-based, maker exit)
          if (Number.isFinite(curHigh) && curHigh >= open.takeProfit) {
            const exit = open.takeProfit;
            const remQty = open.qtyA + open.qtyB;
            const gross = open.realizedGross + remQty * (exit - open.entryPrice);
            const fee = feeEstimate(open.entryPrice, exit, remQty, feeRateMaker, feeRateMaker);
            closes.push({
              symbol,
              entry_meta: open.entry_meta,
              open_time_iso: nowIsoFromMs(open.openedAtMs),
              close_time_iso: nowIsoFromMs(Number(current[0]) + 60_000),
              entry: open.entryPrice,
              exit,
              close_reason: open.tp1AsTp ? 'TP' : (open.tp1Hit ? 'TP2_after_TP1' : 'TP'),
              gross,
              net: open.realizedNet + (remQty * (exit - open.entryPrice) - fee),
              tp1_hit: (open.tp1AsTp ? true : !!open.tp1Hit),
              duration_s: ageS,
              mfe_pct: open._maxPriceSinceOpen != null && open.entryPrice > 0 ? ((open._maxPriceSinceOpen - open.entryPrice) / open.entryPrice) : 0,
              mae_pct: open._minPriceSinceOpen != null && open.entryPrice > 0 ? ((open.entryPrice - open._minPriceSinceOpen) / open.entryPrice) : 0,
            });
            open = null;
          }
        }
      }
    }

    // --- 2) entry evaluation (only if no open position) ---
    if (open) continue;

    // Filter blocked hours UTC
    if (Array.isArray(cfg.BLOCKED_HOURS_UTC) && cfg.BLOCKED_HOURS_UTC.length > 0) {
      const entryHour = new Date(Number(current[0])).getUTCHours();
      if (cfg.BLOCKED_HOURS_UTC.includes(entryHour)) {
        if (funnel) funnel.blockedHour = (funnel.blockedHour || 0) + 1;
        continue;
      }
    }

    // closes series from closed candles
    const closesSeries = mr.getClosedCloses(window);
    if (!Array.isArray(closesSeries) || closesSeries.length < 3) continue;

    const last = closesSeries[closesSeries.length - 1];
    const prev = closesSeries[closesSeries.length - 2] || last;
    const momentum_pct = prev ? (last - prev) / prev : 0;

    const body = Math.abs(candle.close - candle.open);
    const range = candle.high - candle.low;
    const bodyRatio = range > 0 ? body / range : 0;
    const weakBody = bodyRatio < minCandleBody;

    let weakOpen = false;
    if (window.length >= 6) {
      const prevClose1 = Number(window[window.length - 3][4]);
      const prevClose2 = Number(window[window.length - 4][4]);
      if (Number.isFinite(prevClose1) && Number.isFinite(prevClose2)) {
        weakOpen = candle.open < prevClose1 && candle.open < prevClose2;
      }
    }

    let explosive = false;
    try {
      const kA = window[window.length - 2];
      const kB = window[window.length - 3];
      const ranges = [kA, kB].map((k) => {
        const hi = Number(k[2]);
        const lo = Number(k[3]);
        const cl = Number(k[4]);
        if (!Number.isFinite(hi) || !Number.isFinite(lo) || !Number.isFinite(cl) || cl <= 0) return 0;
        return (hi - lo) / cl;
      });
      explosive = (Number.isFinite(explosiveCandlePct) && explosiveCandlePct > 0)
        ? ranges.some((r) => r > explosiveCandlePct)
        : false;
    } catch {
      explosive = false;
    }

    const quoteVol = Number.isFinite(candle.quoteVolume) ? candle.quoteVolume : (candle.close * candle.volume);
    const volMetric = Number.isFinite(quoteVol) ? quoteVol : candle.volume;
    const volumeOk = (MIN_VOLUME <= 0) ? true : (Number.isFinite(volMetric) && volMetric >= MIN_VOLUME);

    const { sma, smaPrev } = mr.computeSmaPair(closesSeries, SMA_WINDOW);
    const smaSlopeAbs = (sma != null && smaPrev != null) ? (sma - smaPrev) : 0;
    const smaSlopePct = (sma != null && smaPrev != null && smaPrev !== 0) ? (sma - smaPrev) / smaPrev : 0;
    const priceNearSMA = (sma != null) ? (candle.close >= sma * (1 - smaTol)) : true;
    const priceAboveSMA = (sma != null) ? (candle.close > sma) : true;

    const { atrPct } = mr.computeAtr(window, ATR_WINDOW);
    const atrPctNum = Number.isFinite(atrPct) ? atrPct : 0;

    // adaptive atr history
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

    // Defensive hard skip on ATR% (entry only)
    if (Number.isFinite(DEF_MIN_ATR_PCT_HARD) && DEF_MIN_ATR_PCT_HARD > 0 && atrPctNum < DEF_MIN_ATR_PCT_HARD) {
      if (funnel) funnel.atr++;
      continue;
    }
    if (Number.isFinite(DEF_MAX_ATR_PCT) && DEF_MAX_ATR_PCT > 0 && atrPctNum > DEF_MAX_ATR_PCT) {
      if (funnel) funnel.atr++;
      continue;
    }

    const realizedVol = mr.computeRealizedVol(closesSeries, 30);
    const regimeInfo = mr.detectMarketRegime({
      atrPct: atrPctNum,
      realizedVol,
      smaSlope: smaSlopeAbs,
      lastClose: candle.close,
      priceAboveSma: priceAboveSMA,
    });

    // --- Defensive hard skips (entry only) ---
    if (DEF_SKIP_LOW_VOL_HARD && regimeInfo.volRegime === 'LOW_VOL') {
      if (funnel) funnel.regime++;
      continue;
    }
    if (DEF_SKIP_CHOPPY_LOWVOL && regimeInfo.volRegime === 'LOW_VOL' && regimeInfo.microRegime === 'CHOPPY') {
      if (funnel) funnel.regime++;
      continue;
    }
    if (DEF_SKIP_CHOPPY_HIGHVOL && regimeInfo.volRegime === 'HIGH_VOL' && regimeInfo.microRegime === 'CHOPPY') {
      if (funnel) funnel.regime++;
      continue;
    }
    if (DEF_SKIP_CHOPPY_ALWAYS && regimeInfo.microRegime === 'CHOPPY') {
      if (funnel) funnel.regime++;
      continue;
    }

    // afterSkips no se usa; omitimos

    const dynamicMinSlopeAbs = MIN_SMA_SLOPE * regimeInfo.kMinSlope;
    const dynamicMinSlopePct = (MIN_SMA_SLOPE_PCT != null && Number.isFinite(MIN_SMA_SLOPE_PCT))
      ? (MIN_SMA_SLOPE_PCT * regimeInfo.kMinSlope)
      : null;

    let trendUp = (dynamicMinSlopePct != null)
      ? (smaSlopePct > dynamicMinSlopePct)
      : (smaSlopeAbs > dynamicMinSlopeAbs);

    if (TREND_GATE_MODE === 'no_trend') trendUp = true;

    const slopeNormOk = (minSlopeNormEff > 0)
      ? (Number(regimeInfo.slopeNorm) >= minSlopeNormEff)
      : true;

    const dynamicMinMom = MIN_MOMENTUM_PCT * regimeInfo.kMinMomentum;
    const effectiveMinMom = dynamicMinMom;
    const momentumOk = (momentum_pct >= effectiveMinMom) && (momentum_pct <= MAX_MOMENTUM_PCT);

    // Extra indicators (match live service gating; no effect if USE_* is false)
    const adxInfo = mr.computeAdx(window, ADX_WINDOW);

    let donchPrevHigh = null;
    try {
      const closed = window.slice(0, Math.max(0, window.length - 1)); // exclude in-progress
      if (closed.length >= DONCHIAN_N + 1) {
        const slice = closed.slice(-(DONCHIAN_N + 1), -1); // exclude current closed candle
        donchPrevHigh = Math.max(...slice.map(k => Number(k?.[2])).filter(Number.isFinite));
        if (!Number.isFinite(donchPrevHigh)) donchPrevHigh = null;
      }
    } catch { donchPrevHigh = null; }

    const st = mr.computeSupertrend(window, SUPERTREND_ATR_WINDOW, SUPERTREND_MULT);

    const adxOk = (!USE_ADX_FILTER || !(MIN_ADX > 0))
      ? true
      : (adxInfo.adx != null && Number.isFinite(adxInfo.adx) && adxInfo.adx >= MIN_ADX);

    const diBullOk = (!USE_ADX_FILTER || !ADX_REQUIRE_DI_BULL)
      ? true
      : (adxInfo.diPlus != null && adxInfo.diMinus != null && adxInfo.diPlus > adxInfo.diMinus);

    const donchOk = (!USE_DONCHIAN_FILTER)
      ? true
      : (donchPrevHigh != null && candle.close > donchPrevHigh);

    const stOk = (!USE_SUPERTREND_FILTER)
      ? true
      : (st.dir === 1);

    const shouldEnter = atrOk && slopeNormOk && momentumOk && trendUp && priceNearSMA && priceAboveSMA && volumeOk && adxOk && diBullOk && donchOk && stOk && !weakBody && !weakOpen && !explosive;

    // Funnel counters (desired fields only)
    if (funnel) {
      if (!atrOk) funnel.atr++;
      if (!momentumOk) funnel.momentum++;
      if (!trendUp) {
        if (dynamicMinSlopePct != null) {
          funnel.trend++;
        } else {
          funnel.slopeAbs++;
        }
      }
      if (!volumeOk) funnel.vol++;
      if (!adxOk) funnel.adx++;
      if (!diBullOk) funnel.diBull++;
      if (!donchOk) funnel.donch++;
      if (!stOk) funnel.st++;
      if (weakBody) {
        funnel.body++;
        funnel.weakBody++;
      }
      if (weakOpen) funnel.weakOpen++;
      if (explosive) funnel.explosive++;
    }

    if (!shouldEnter) continue;

    if (funnel) funnel.passed++;

    // Build TP/SL
    const entryPrice = candle.close;
    const rr = (k_sl > 0) ? (k_tp / k_sl) : 2.0;
    let riskDist = entryPrice * (k_sl * atrPctNum);

    if (Number.isFinite(MIN_SL_PCT) && MIN_SL_PCT > 0) {
      const minDist = entryPrice * MIN_SL_PCT;
      if (riskDist < minDist) riskDist = minDist;
    } else if (MIN_SL_USD > 0 && riskDist < MIN_SL_USD) {
      riskDist = MIN_SL_USD;
    }

    const stopLossClassic = entryPrice - riskDist;
    const stopLossEmergency = (Number.isFinite(EMERGENCY_SL_PCT) && EMERGENCY_SL_PCT > 0)
      ? (entryPrice * (1 - EMERGENCY_SL_PCT))
      : stopLossClassic;

    // Which SL is "armed" as hard stop in state (match live)
    const stopLoss = (SL_POLICY === 'SOFT_CLASSIC_WITH_EMERGENCY' || SL_POLICY === 'EMERGENCY_ONLY')
      ? stopLossEmergency
      : stopLossClassic;

    const tp1AsTp = false; // keep TP ATR/RR; do NOT force TP1 as TP in this harness
    const tp1 = entryPrice * (1 + TP1_PCT);
    let takeProfit = entryPrice + riskDist * rr;
    if (tp1AsTp) takeProfit = tp1;

    const qty = Number((tradeUSD / entryPrice).toFixed(8));

    // Skip if current candle already pierced ARMED SL (match live)
    const currentLow = Number(current?.[3]);
    if (Number.isFinite(currentLow) && currentLow <= stopLoss) {
      if (funnel) funnel.curLow++;
      continue;
    }

    const grossProfitAtTp = qty * (takeProfit - entryPrice);
    // Entry assumed maker (bot uses maker-entry attempts); TP assumed maker for this check.
    const feeEstAtTp = feeEstimate(entryPrice, takeProfit, qty, feeRateMaker, feeRateMaker);
    const netProfitAtTp = grossProfitAtTp - feeEstAtTp;
    if (netProfitAtTp <= 0) {
      if (funnel) funnel.netProfit++;
      continue;
    }

    // Partial TP sizing (if TP1 is the only TP, disable partial logic)
    const qtyA = tp1AsTp ? 0 : Number((qty * TP1_FRAC).toFixed(8));
    const qtyB = tp1AsTp ? qty : Number((qty - qtyA).toFixed(8));

    const entry_meta = {
      ts: Number(current?.[0]),
      open_time_iso: nowIsoFromMs(Number(current?.[0])),
      atr_pct: atrPctNum,
      realized_vol: realizedVol,
      volRegime: regimeInfo?.volRegime,
      microRegime: regimeInfo?.microRegime,
      slopeNorm: regimeInfo?.slopeNorm,
      kMinMomentum: regimeInfo?.kMinMomentum,
      kMinSlope: regimeInfo?.kMinSlope,
      sma: sma,
      smaSlopeAbs: smaSlopeAbs,
      smaSlopePct: smaSlopePct,
      minSlopeAbsEff: dynamicMinSlopeAbs,
      minSlopePctEff: dynamicMinSlopePct,
      minSlopeNormEff: minSlopeNormEff,
      momentum_pct: momentum_pct,
      minMomEff: effectiveMinMom,
      maxMomEff: MAX_MOMENTUM_PCT,
      priceNearSMA: !!priceNearSMA,
      priceAboveSMA: !!priceAboveSMA,
      volumeOk: !!volumeOk,
      donchPrevHigh: donchPrevHigh,
      donchOk: !!donchOk,
      adx: adxInfo?.adx,
      diPlus: adxInfo?.diPlus,
      diMinus: adxInfo?.diMinus,
      adxOk: !!adxOk,
      diBullOk: !!diBullOk,
      stDir: st?.dir,
      stOk: !!stOk,
      trendUp: !!trendUp,
    };

    open = {
      symbol,
      entryPrice,
      stopLoss,
      stopLossClassic,
      stopLossEmergency,
      takeProfit,
      tp1,
      tp1AsTp,
      qty,
      qtyA,
      qtyB,
      tp1Hit: false,
      realizedGross: 0,
      realizedNet: 0,
      entry_meta,
      openedAtMs: Number(current[0]),
      _slBreachStartMs: null,
      _maxPriceSinceOpen: entryPrice,
      _minPriceSinceOpen: entryPrice,
      _rev5mChecked: false,
    };
    if (debugFunnel) debugFunnel.opened++;
  }

  return closes;
}

function summarize(closes) {
  const n = closes.length;
  const gross = closes.reduce((a, x) => a + x.gross, 0);
  const net = closes.reduce((a, x) => a + x.net, 0);
  const wins = closes.filter((x) => x.net > 0);
  const losses = closes.filter((x) => x.net <= 0);
  const winrate = n ? (wins.length / n) * 100 : 0;
  const avgWin = wins.length ? wins.reduce((a, x) => a + x.net, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((a, x) => a + x.net, 0) / losses.length : 0;
  const pf = Math.abs(losses.reduce((a, x) => a + x.net, 0)) > 0
    ? wins.reduce((a, x) => a + x.net, 0) / Math.abs(losses.reduce((a, x) => a + x.net, 0))
    : (wins.length ? Infinity : 0);
  return { n, gross, net, winrate, avgWin, avgLoss, pf };
}

async function main() {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const skillDir = path.resolve(__dirname, '..');

  // Debug funnel (opt-in)
  const DEBUG_FUNNEL = String(process.env.DEBUG_FUNNEL || '0') === '1';
  const funnel = DEBUG_FUNNEL ? {
    total: 0,
    atr: 0, atrHard: 0, atrMax: 0,
    vol: 0,
    body: 0, explosive: 0,
    momentum: 0, momentumMax: 0,
    priceNear: 0, priceAbove: 0,
    slopeNorm: 0, slopeAbs: 0, trend: 0,
    donch: 0,
    adx: 0, diBull: 0,
    st: 0,
    weakBody: 0, weakOpen: 0,
    afterSkips: 0, shouldEnter: 0,
    curLow: 0, netProfit: 0,
    opened: 0
  } : null;

  // --- CLI args ---
  const argv = process.argv.slice(2);
  let days = 7;
  let configPath = process.env.CONFIG_PATH || path.join(skillDir, 'config.json');
  let endMsOverride = process.env.END_MS ? Number(process.env.END_MS) : null;
  let startMsOverride = process.env.START_MS ? Number(process.env.START_MS) : null;
  let dataSource = process.env.DATA_SOURCE || 'mexc';
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--days') days = Number(argv[++i] || 7);
    else if (a === '--config') configPath = String(argv[++i] || configPath);
    else if (a === '--end-ms') endMsOverride = Number(argv[++i]);
    else if (a === '--start-ms') startMsOverride = Number(argv[++i]);
    else if (a === '--data-source') dataSource = String(argv[++i] || dataSource);
    else if (/^\d+$/.test(a)) days = Number(a);
  }

  const cfg = loadConfig(configPath);

  const symbols = Array.isArray(cfg.SYMBOLS) && cfg.SYMBOLS.length ? cfg.SYMBOLS : [cfg.SYMBOL].filter(Boolean);
  // Fee model: maker/taker (prefer these over legacy FEE_RATE)
  const feeRateMaker = Number.isFinite(Number(cfg.FEE_RATE_MAKER)) ? Number(cfg.FEE_RATE_MAKER) : Number(cfg.FEE_RATE ?? 0.0003);
  const feeRateTaker = Number.isFinite(Number(cfg.FEE_RATE_TAKER)) ? Number(cfg.FEE_RATE_TAKER) : Number(cfg.FEE_RATE ?? 0.0003);

  // --- Data window ---
  const end = Number.isFinite(endMsOverride) ? endMsOverride : Date.now();
  const start = Number.isFinite(startMsOverride)
    ? startMsOverride
    : (end - days * 24 * 60 * 60 * 1000);
  const startAligned = Math.floor(start / 60_000) * 60_000;
  const endAligned = Math.floor(end / 60_000) * 60_000;

  // --- Backtest asset dirs ---
  const backtestDir = path.join(skillDir, 'scripts', 'backtest');
  const cacheDir = path.join(backtestDir, 'cache');
  const reportsDir = path.join(backtestDir, 'reports');
  fs.mkdirSync(cacheDir, { recursive: true });
  fs.mkdirSync(reportsDir, { recursive: true });

  console.log(`BACKTEST_PARTIAL_TP (${String(dataSource).toUpperCase()} 1m) last ${days} days`);
  console.log('symbols:', symbols.join(', '));
  console.log('configPath:', configPath);
  console.log('config:', {
    ATR_ADAPTIVE_ENABLED: !!cfg.ATR_ADAPTIVE_ENABLED,
    ATR_ADAPTIVE_PCTL: cfg.ATR_ADAPTIVE_PCTL,
    MIN_MOMENTUM_PCT: cfg.MIN_MOMENTUM_PCT,
    MAX_MOMENTUM_PCT: cfg.MAX_MOMENTUM_PCT,
    MIN_SMA_SLOPE_PCT: cfg.MIN_SMA_SLOPE_PCT,
    MIN_SLOPE_NORM: cfg.MIN_SLOPE_NORM,
    MIN_SLOPE_NORM_BY_SYMBOL: cfg.MIN_SLOPE_NORM_BY_SYMBOL,
    K_TP: cfg.K_TP,
    K_SL: cfg.K_SL,
    MIN_SL_PCT: cfg.MIN_SL_PCT,
    MIN_VOLUME: cfg.MIN_VOLUME,
    MIN_BODY_CANDLE: cfg.MIN_BODY_CANDLE,
    EXPLOSIVE_CANDLE_PCT: cfg.EXPLOSIVE_CANDLE_PCT,
    SMA_WINDOW: cfg.SMA_WINDOW,
    ATR_WINDOW: cfg.ATR_WINDOW,
  });

  const klinesBySymbol = {};
  for (const sym of symbols) {
    const cacheName = `${String(dataSource).toLowerCase()}_1m_${sym}_${startAligned}_${endAligned}.json`;
    const cachePath = path.join(cacheDir, cacheName);

    if (fs.existsSync(cachePath)) {
      console.log(`cache hit: ${sym} -> ${cacheName}`);
      const kl = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
      klinesBySymbol[sym] = kl;
      console.log(`  klines: ${kl.length}`);
      continue;
    }

    console.log(`fetching ${sym}...`);
    const src = String(dataSource).toLowerCase();
    const kl = (src === 'binance')
      ? await fetchKlinesBinance(sym, startAligned, endAligned)
      : await fetchKlinesMexc(sym, startAligned, endAligned);
    console.log(`  klines: ${kl.length}`);
    fs.writeFileSync(cachePath, JSON.stringify(kl));
    console.log(`  saved cache: ${cachePath}`);
    klinesBySymbol[sym] = kl;
  }

  const allCloses = [];
  for (const sym of symbols) {
    const closes = backtestSymbolSeries({ 
      symbol: sym, 
      klines: klinesBySymbol[sym], 
      cfg, 
      feeRateMaker, 
      feeRateTaker,
      debugFunnel: DEBUG_FUNNEL ? funnel : null 
    });
    for (const c of closes) allCloses.push({ ...c, symbol: sym });
  }

  allCloses.sort((a, b) => a.close_time_iso.localeCompare(b.close_time_iso));

  const sum = summarize(allCloses);
  console.log('\nRESULTS (net):');
  console.log(sum);

  // Per symbol
  for (const sym of symbols) {
    const s = summarize(allCloses.filter((x) => x.symbol === sym));
    console.log(`  ${sym}:`, s);
  }

  if (DEBUG_FUNNEL) {
    console.log('\nFUNNEL:', JSON.stringify(funnel, null, 2));
  }

  console.log('\nCLOSES (last 10):');
  for (const c of allCloses.slice(-10)) {
    console.log(`${c.close_time_iso} ${c.symbol} ${c.close_reason} net=${c.net.toFixed(2)} gross=${c.gross.toFixed(2)} dur=${(c.duration_s/60).toFixed(1)}m`);
  }

  // Save JSON output
  const stamp = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15) + 'Z';
  const outPath = path.join(reportsDir, `backtest_${days}d_${stamp}.json`);
  fs.writeFileSync(outPath, JSON.stringify({ startedAt: new Date().toISOString(), days, symbols, configPath, config: cfg, summary: sum, closes: allCloses }, null, 2));
  console.log(`\nSaved report: ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
