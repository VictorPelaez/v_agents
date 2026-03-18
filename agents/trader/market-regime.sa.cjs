'use strict';

/**
 * market-regime.sa.cjs
 *
 * Small, dependency-free helper module for:
 * - ATR% calculation
 * - SMA + SMA slope
 * - realized volatility (std of log-returns)
 * - simple, stable market regime classification
 *
 * Design goals:
 * - Robust to bad/missing data
 * - Deterministic + well-commented
 * - Avoid overfitting: use a small set of conservative heuristics
 */

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  return Math.max(lo, Math.min(hi, x));
}

function mean(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  if (!Array.isArray(arr) || arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.reduce((acc, x) => acc + Math.pow(x - m, 2), 0) / (arr.length - 1);
  return Math.sqrt(v);
}

function safeNum(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

/**
 * Extracts the last CLOSED candle index for Binance-style klines.
 * Klines array ends with the *current in-progress* candle.
 */
function lastClosedIndex(klines) {
  if (!Array.isArray(klines) || klines.length < 2) return -1;
  return klines.length - 2;
}

function getClosedCloses(klines) {
  if (!Array.isArray(klines) || klines.length < 2) return [];
  // exclude current in-progress candle (last element)
  const closed = klines.slice(0, klines.length - 1);
  const closes = [];
  for (const k of closed) {
    const c = safeNum(k?.[4]);
    if (c == null) return []; // hard fail -> caller can handle
    closes.push(c);
  }
  return closes;
}

/**
 * SMA over the last `window` closed closes.
 * Returns { sma, smaPrev } where:
 * - sma is SMA of last `window` closes
 * - smaPrev is SMA of the window shifted by 1 close (previous SMA)
 */
function computeSmaPair(closes, window) {
  if (!Array.isArray(closes) || closes.length < 2) return { sma: null, smaPrev: null };
  const w = Math.max(2, parseInt(window || 60, 10));

  // If we don't have enough samples, fall back to full-length average.
  if (closes.length < w + 1) {
    const sma = mean(closes);
    return { sma, smaPrev: null };
  }

  const lastWindow = closes.slice(-w);
  const prevWindow = closes.slice(-w - 1, -1);
  return { sma: mean(lastWindow), smaPrev: mean(prevWindow) };
}

/**
 * ATR% computed from closed klines.
 * Uses classic True Range definition.
 * Returns { atr, atrPct } in price-units and percent-of-price.
 */
function computeAtr(klines, atrWindow) {
  const w = Math.max(2, parseInt(atrWindow || 14, 10));
  const idxLastClosed = lastClosedIndex(klines);
  if (idxLastClosed < 0) return { atr: 0, atrPct: 0 };

  // Need at least w + 1 closed candles to compute TR with prevClose.
  const closed = Array.isArray(klines) ? klines.slice(0, klines.length - 1) : [];
  if (closed.length < w + 1) return { atr: 0, atrPct: 0 };

  const trs = [];
  // Compute TR for the last w candles.
  // We start at closed.length - w (inclusive) so that i-1 exists.
  for (let i = closed.length - w; i < closed.length; i++) {
    const high = safeNum(closed[i]?.[2]);
    const low = safeNum(closed[i]?.[3]);
    const prevClose = safeNum(closed[i - 1]?.[4]);
    if (high == null || low == null || prevClose == null) return { atr: 0, atrPct: 0 };

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trs.push(tr);
  }

  const atr = mean(trs);
  const lastClose = safeNum(closed[closed.length - 1]?.[4]) || 1;
  const atrPct = lastClose > 0 ? atr / lastClose : 0;
  return { atr, atrPct };
}

/**
 * Realized volatility: std of log returns over last N closed closes.
 * Returns a *fraction* (e.g. 0.001 = 0.1%).
 */
function computeRealizedVol(closes, window) {
  const w = Math.max(5, parseInt(window || 30, 10));
  if (!Array.isArray(closes) || closes.length < w + 1) return 0;

  const slice = closes.slice(-w - 1);
  const rets = [];
  for (let i = 1; i < slice.length; i++) {
    const a = slice[i - 1];
    const b = slice[i];
    if (!(a > 0 && b > 0)) continue;
    rets.push(Math.log(b / a));
  }
  return std(rets);
}

/**
 * Market regime classifier.
 *
 * Returns:
 * - volRegime: LOW_VOL | MID_VOL | HIGH_VOL
 * - microRegime: TRENDING | CHOPPY | NORMAL
 * - multipliers: conservative scalars to adapt thresholds
 */
function detectMarketRegime({ atrPct, realizedVol, smaSlope, lastClose, priceAboveSma }) {
  const a = Number.isFinite(atrPct) ? atrPct : 0;
  const rv = Number.isFinite(realizedVol) ? realizedVol : 0;
  const slope = Number.isFinite(smaSlope) ? smaSlope : 0;
  const px = Number.isFinite(lastClose) && lastClose > 0 ? lastClose : 1;

  // ATR in price units (approx) to normalize slope.
  const atrAbs = a * px;
  const slopeNorm = atrAbs > 0 ? Math.abs(slope) / atrAbs : 0;

  // --- Volatility regime ---
  // Use both ATR% and realized vol so we don't overreact to one noisy measure.
  let volRegime = 'MID_VOL';
  if (a < 0.00055 && rv < 0.00045) volRegime = 'LOW_VOL';
  if (a > 0.00110 || rv > 0.00100) volRegime = 'HIGH_VOL';

  // --- Micro regime (trend/chop) ---
  // NOTE (2026-03-18): thresholds tuned for 1m trading. The previous (0.30/0.80)
  // labeled almost everything as CHOPPY because SMA(60) slope is very smooth.
  const CHOPPY_SLOPE_NORM = 0.12;
  const TRENDING_SLOPE_NORM = 0.45;

  let microRegime = 'NORMAL';
  // CHOPPY: slope too small relative to ATR (no directional edge)
  if (slopeNorm < CHOPPY_SLOPE_NORM) microRegime = 'CHOPPY';
  // TRENDING: strong slope relative to ATR; only meaningful if slope aligns with long bias
  if (slopeNorm > TRENDING_SLOPE_NORM && slope > 0 && !!priceAboveSma) microRegime = 'TRENDING';

  // --- Multipliers (bounded) ---
  // kMinMomentum / kMinSlope scale the thresholds.
  // >1 => stricter (fewer trades), <1 => looser.
  let kMinMomentum = 1.0;
  let kMinSlope = 1.0;

  if (volRegime === 'LOW_VOL') {
    kMinMomentum *= 0.90;
    kMinSlope *= 0.95;
  }
  if (volRegime === 'HIGH_VOL') {
    kMinMomentum *= 1.15;
    kMinSlope *= 1.10;
  }

  if (microRegime === 'CHOPPY') {
    kMinMomentum *= 1.10;
    kMinSlope *= 1.10;
  }
  if (microRegime === 'TRENDING') {
    kMinMomentum *= 0.95;
    kMinSlope *= 0.95;
  }

  kMinMomentum = clamp(kMinMomentum, 0.80, 1.30);
  kMinSlope = clamp(kMinSlope, 0.80, 1.30);

  return {
    volRegime,
    microRegime,
    // Return full precision; callers can round for logging.
    slopeNorm,
    kMinMomentum,
    kMinSlope
  };
}

/* -----------------------------
 * Extra indicators (Phase A)
 * ----------------------------- */

// Donchian channel on CLOSED candles (exclude current in-progress)
function computeDonchian(klines, window) {
  const n = Math.max(2, parseInt(window || 20, 10));
  const closed = Array.isArray(klines) ? klines.slice(0, Math.max(0, klines.length - 1)) : [];
  if (closed.length < n) return { high: null, low: null };
  const slice = closed.slice(-n);
  let hi = -Infinity;
  let lo = Infinity;
  for (const k of slice) {
    const h = safeNum(k?.[2]);
    const l = safeNum(k?.[3]);
    if (h == null || l == null) return { high: null, low: null };
    if (h > hi) hi = h;
    if (l < lo) lo = l;
  }
  return { high: hi, low: lo };
}

// ADX (+DI/-DI) on CLOSED candles, Wilder's smoothing (simplified, stable)
function computeAdx(klines, window) {
  const n = Math.max(5, parseInt(window || 14, 10));
  const closed = Array.isArray(klines) ? klines.slice(0, Math.max(0, klines.length - 1)) : [];
  // Need at least n+1 candles to form n periods of deltas
  if (closed.length < n + 1) return { adx: null, diPlus: null, diMinus: null };

  // Build arrays for last n periods
  const trs = [];
  const dmPlus = [];
  const dmMinus = [];

  for (let i = closed.length - n; i < closed.length; i++) {
    const cur = closed[i];
    const prev = closed[i - 1];
    const high = safeNum(cur?.[2]);
    const low = safeNum(cur?.[3]);
    const prevHigh = safeNum(prev?.[2]);
    const prevLow = safeNum(prev?.[3]);
    const prevClose = safeNum(prev?.[4]);
    if ([high, low, prevHigh, prevLow, prevClose].some(v => v == null)) {
      return { adx: null, diPlus: null, diMinus: null };
    }

    const upMove = high - prevHigh;
    const downMove = prevLow - low;

    const plus = (upMove > downMove && upMove > 0) ? upMove : 0;
    const minus = (downMove > upMove && downMove > 0) ? downMove : 0;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );

    trs.push(tr);
    dmPlus.push(plus);
    dmMinus.push(minus);
  }

  const trSum = trs.reduce((a, b) => a + b, 0);
  if (!(trSum > 0)) return { adx: 0, diPlus: 0, diMinus: 0 };

  const diPlus = 100 * (dmPlus.reduce((a, b) => a + b, 0) / trSum);
  const diMinus = 100 * (dmMinus.reduce((a, b) => a + b, 0) / trSum);
  const dx = (diPlus + diMinus) > 0 ? (100 * Math.abs(diPlus - diMinus) / (diPlus + diMinus)) : 0;

  // NOTE: Full ADX is an EMA of DX over n periods.
  // For Phase A logging we use a stable approximation: report DX as "adx".
  // If we adopt it for decisions later, we should implement the full Wilder smoothing.
  return { adx: dx, diPlus, diMinus };
}

// Supertrend (basic) on CLOSED candles; returns { value, dir }
// dir: 1 bullish, -1 bearish
function computeSupertrend(klines, atrWindow, multiplier) {
  const m = Number.isFinite(Number(multiplier)) ? Number(multiplier) : 3;
  const w = Math.max(2, parseInt(atrWindow || 10, 10));
  const closed = Array.isArray(klines) ? klines.slice(0, Math.max(0, klines.length - 1)) : [];
  if (closed.length < w + 2) return { value: null, dir: null };

  // Compute ATR (simple mean of TR) over last w periods
  const trs = [];
  for (let i = closed.length - w; i < closed.length; i++) {
    const cur = closed[i];
    const prev = closed[i - 1];
    const high = safeNum(cur?.[2]);
    const low = safeNum(cur?.[3]);
    const prevClose = safeNum(prev?.[4]);
    if ([high, low, prevClose].some(v => v == null)) return { value: null, dir: null };
    trs.push(Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose)));
  }
  const atr = mean(trs);

  const last = closed[closed.length - 1];
  const high = safeNum(last?.[2]);
  const low = safeNum(last?.[3]);
  const close = safeNum(last?.[4]);
  if ([high, low, close].some(v => v == null)) return { value: null, dir: null };

  const hl2 = (high + low) / 2;
  const upperBasic = hl2 + m * atr;
  const lowerBasic = hl2 - m * atr;

  // For Phase A (logging), we approximate final bands without full recursion.
  // Direction is inferred from close vs mid-band.
  const dir = close >= hl2 ? 1 : -1;
  const value = (dir === 1) ? lowerBasic : upperBasic;

  return { value, dir };
}

module.exports = {
  clamp,
  getClosedCloses,
  computeSmaPair,
  computeAtr,
  computeRealizedVol,
  detectMarketRegime,
  computeDonchian,
  computeAdx,
  computeSupertrend
};
