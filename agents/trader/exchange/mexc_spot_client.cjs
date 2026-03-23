'use strict';

/**
 * High-level MEXC Spot v3 client wrapper.
 *
 * Goals:
 * - Keep trading-bot code small/maintainable
 * - Centralize signing, REST calls, order status parsing
 * - Cache exchangeInfo rules (tickSize/stepSize/minQty/minNotional)
 * - Provide waitForFill polling with transient-error tolerance
 */

const { mexcPublic, mexcSigned } = require('./mexc_spot_v3.cjs');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function pickNum(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k];
    const n = (v == null) ? NaN : Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function floorToStep(x, step) {
  const n = Number(x);
  const s = Number(step);
  if (!Number.isFinite(n) || !Number.isFinite(s) || s <= 0) return n;
  const inv = 1 / s;
  return Math.floor(n * inv) / inv;
}

function orderStatus(order) {
  return String(order?.status || order?.state || '').toUpperCase();
}

function isOrderFilled(order) {
  const st = orderStatus(order);
  return st === 'FILLED' || st === 'DONE';
}

function isOrderCanceled(order) {
  const st = orderStatus(order);
  return st === 'CANCELED' || st === 'CANCELLED' || st === 'EXPIRED' || st === 'REJECTED';
}

function orderAvgFillPrice(order) {
  const avg = pickNum(order, 'avgPrice', 'avg_price');
  if (avg != null) return avg;

  const quote = pickNum(order, 'cummulativeQuoteQty', 'cumulativeQuoteQty', 'cummulativeQuoteQuantity', 'quoteQty');
  const exec = pickNum(order, 'executedQty', 'executedQuantity', 'cumulativeQuantity');
  if (quote != null && exec != null && exec > 0) return quote / exec;
  return null;
}

function parseSymbolRules(exchangeInfo, symbol) {
  try {
    const syms = exchangeInfo?.symbols || exchangeInfo?.data?.symbols || [];
    const s = Array.isArray(syms)
      ? (syms.find((z) => String(z?.symbol || '').toUpperCase() === String(symbol || '').toUpperCase()) || syms[0])
      : null;
    const filters = s?.filters || [];
    const pf = Array.isArray(filters) ? filters.find((f) => String(f?.filterType || f?.filter_type) === 'PRICE_FILTER') : null;
    const lf = Array.isArray(filters) ? filters.find((f) => String(f?.filterType || f?.filter_type) === 'LOT_SIZE') : null;
    const mn = Array.isArray(filters) ? filters.find((f) => String(f?.filterType || f?.filter_type) === 'MIN_NOTIONAL') : null;

    const tickSize = Number(pf?.tickSize);
    const stepSize = Number(lf?.stepSize);
    const minQty = Number(lf?.minQty);
    const minNotional = Number(mn?.minNotional);

    return {
      tickSize: Number.isFinite(tickSize) ? tickSize : null,
      stepSize: Number.isFinite(stepSize) ? stepSize : null,
      minQty: Number.isFinite(minQty) ? minQty : null,
      minNotional: Number.isFinite(minNotional) ? minNotional : null,
    };
  } catch (_) {
    return { tickSize: null, stepSize: null, minQty: null, minNotional: null };
  }
}

function createMexcSpotClient(opts) {
  const baseUrl = opts?.baseUrl;
  const apiKey = opts?.apiKey;
  const apiSecret = opts?.apiSecret;
  const defaultTimeoutMs = Number.isFinite(Number(opts?.timeoutMs)) ? Number(opts.timeoutMs) : 2500;

  if (!baseUrl) throw new Error('createMexcSpotClient: baseUrl required');
  if (!apiKey || !apiSecret) throw new Error('createMexcSpotClient: apiKey/apiSecret required');

  const symbolRulesCache = new Map();

  async function publicGet(path, params, timeoutMs) {
    return mexcPublic({ baseUrl, method: 'GET', path, params: params || {}, timeoutMs: timeoutMs || defaultTimeoutMs });
  }

  async function signed(method, path, params, timeoutMs) {
    return mexcSigned({ baseUrl, method, path, params: params || {}, apiKey, apiSecret, timeoutMs: timeoutMs || defaultTimeoutMs });
  }

  async function bookTicker(symbol, timeoutMs) {
    return publicGet('/api/v3/ticker/bookTicker', { symbol }, timeoutMs);
  }

  async function exchangeInfo(symbol, timeoutMs) {
    return publicGet('/api/v3/exchangeInfo', symbol ? { symbol } : {}, timeoutMs);
  }

  async function getSymbolRules(symbol, timeoutMs) {
    const sym = String(symbol || '').toUpperCase();
    if (symbolRulesCache.has(sym)) return symbolRulesCache.get(sym);
    const info = await exchangeInfo(sym, timeoutMs);
    const rules = parseSymbolRules(info, sym);
    symbolRulesCache.set(sym, rules);
    return rules;
  }

  async function normalizeQuantity(symbol, quantity, timeoutMs) {
    const rules = await getSymbolRules(symbol, timeoutMs);
    let q = Number(quantity);
    if (rules.stepSize) q = floorToStep(q, rules.stepSize);
    return { qty: q, rules };
  }

  async function normalizeLimit(symbol, quantity, price, timeoutMs) {
    const { qty, rules } = await normalizeQuantity(symbol, quantity, timeoutMs);
    let p = Number(price);
    if (rules.tickSize) p = floorToStep(p, rules.tickSize);
    return { qty, price: p, rules };
  }

  async function placeOrder(params, timeoutMs) {
    return signed('POST', '/api/v3/order', params, timeoutMs);
  }

  async function getOrder(params, timeoutMs) {
    return signed('GET', '/api/v3/order', params, timeoutMs);
  }

  async function cancelOrder(params, timeoutMs) {
    return signed('DELETE', '/api/v3/order', params, timeoutMs);
  }

  async function openOrders(params, timeoutMs) {
    return signed('GET', '/api/v3/openOrders', params, timeoutMs);
  }

  async function myTrades(params, timeoutMs) {
    return signed('GET', '/api/v3/myTrades', params, timeoutMs);
  }

  async function klines(params, timeoutMs) {
    return publicGet('/api/v3/klines', params, timeoutMs);
  }

  async function waitForFill({ symbol, orderId, origClientOrderId, timeoutMs = 15000, pollMs = 500, httpTimeoutMs }) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        const ord = await getOrder({ symbol, orderId, origClientOrderId }, httpTimeoutMs || defaultTimeoutMs);
        if (isOrderFilled(ord)) return ord;
        if (isOrderCanceled(ord)) return ord;
      } catch (_) {
        // ignore transient errors
      }
      await sleep(Math.max(50, pollMs));
    }
    return null;
  }

  return {
    baseUrl,
    bookTicker,
    exchangeInfo,
    getSymbolRules,
    normalizeQuantity,
    normalizeLimit,
    placeOrder,
    getOrder,
    cancelOrder,
    openOrders,
    myTrades,
    klines,
    waitForFill,
    // helpers
    isOrderFilled,
    isOrderCanceled,
    orderAvgFillPrice,
  };
}

module.exports = {
  createMexcSpotClient,
  floorToStep,
};
