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

module.exports = {
  sleep,
  pickNum,
};
