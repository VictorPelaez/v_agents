'use strict';

/** Move-only refactor: market data fetch + simple symbol ranking. */

function createMarketData(ctx) {
  const {
    getApiBase,
    httpGetWithRetry,
    activeApiKey,
    exchange,
    state,
  } = ctx || {};

  if (!getApiBase) throw new Error('createMarketData: getApiBase required');
  if (!httpGetWithRetry) throw new Error('createMarketData: httpGetWithRetry required');
  if (!state) throw new Error('createMarketData: state required');

  async function getTicker(symbol, timeoutMs) {
    const base = getApiBase();
    const url = `${base}/api/v3/ticker/price?symbol=${symbol}`;

    try {
      const headers = activeApiKey ? { 'X-MBX-APIKEY': activeApiKey } : undefined;
      const r = await httpGetWithRetry(url, headers ? { headers } : {}, 4, 250, timeoutMs);
      const px = Number(r?.data?.price);
      return Number.isFinite(px) ? px : null;
    } catch (e) {
      console.error('getTicker failed', exchange, symbol, e.message);
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
      const headers = activeApiKey ? { 'X-MBX-APIKEY': activeApiKey } : undefined;
      const r = await httpGetWithRetry(url, headers ? { headers } : {}, 4, 250, timeoutMs);
      if (!Array.isArray(r?.data)) return null;
      return r.data;
    } catch (e) {
      console.error('getRecentKlines failed', exchange, symbol, e.message);
      return null;
    }
  }

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

  return {
    getTicker,
    getTickerCached,
    getRecentKlines,
    rankSymbolsByRecentReturn,
  };
}

module.exports = {
  createMarketData,
};
