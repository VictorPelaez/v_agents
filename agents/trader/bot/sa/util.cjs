'use strict';

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function pickNum(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k];
    const n = (v == null) ? NaN : Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
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

module.exports = {
  sleep,
  pickNum,
  isValidNumber,
  percentile,
};
