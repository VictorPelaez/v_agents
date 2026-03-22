'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const path = require('path');

const { evalSlPolicy } = require('../bot/sl_policy.cjs');
const { pickNum } = require('../bot/sa/util.cjs');
const { getYmd, fmtDateUtc1, nowIso, makeGetJournalPath } = require('../bot/sa/time.cjs');
const { computeAtr, computeRealizedVol, detectMarketRegime, computeSmaPair } = require('../market-regime.sa.cjs');

test('sl_policy: emergency triggers immediately in SOFT_CLASSIC_WITH_EMERGENCY', () => {
  const r = evalSlPolicy({
    slPolicy: 'SOFT_CLASSIC_WITH_EMERGENCY',
    market: 90,
    slClassic: 95,
    slEmergency: 91,
    nowMs: 1000,
    breachStartMs: null,
    slConfirmSeconds: 240,
  });
  assert.equal(r.closeReason, 'emergency_sl');
  assert.equal(r.emergency, true);
  assert.equal(r.breachStartMs, null);
});

test('sl_policy: classic confirmation waits until confirmMs elapsed', () => {
  // first breach -> start timer
  const r1 = evalSlPolicy({
    slPolicy: 'SOFT_CLASSIC_WITH_EMERGENCY',
    market: 94,
    slClassic: 95,
    slEmergency: 80,
    nowMs: 1000,
    breachStartMs: null,
    slConfirmSeconds: 10,
  });
  assert.equal(r1.closeReason, null);
  assert.equal(r1.emergency, false);
  assert.equal(r1.breachStartMs, 1000);

  // still within confirmation window
  const r2 = evalSlPolicy({
    slPolicy: 'SOFT_CLASSIC_WITH_EMERGENCY',
    market: 94,
    slClassic: 95,
    slEmergency: 80,
    nowMs: 1000 + 9000,
    breachStartMs: r1.breachStartMs,
    slConfirmSeconds: 10,
  });
  assert.equal(r2.closeReason, null);
  assert.equal(r2.breachStartMs, 1000);

  // elapsed -> SL
  const r3 = evalSlPolicy({
    slPolicy: 'SOFT_CLASSIC_WITH_EMERGENCY',
    market: 94,
    slClassic: 95,
    slEmergency: 80,
    nowMs: 1000 + 10000,
    breachStartMs: r2.breachStartMs,
    slConfirmSeconds: 10,
  });
  assert.equal(r3.closeReason, 'SL');
  assert.equal(r3.emergency, false);
  assert.equal(r3.breachStartMs, null);
});

test('util.pickNum: picks first finite numeric field', () => {
  assert.equal(pickNum({ a: 'x', b: '2.5' }, 'a', 'b', 'c'), 2.5);
  assert.equal(pickNum({ a: null, b: undefined, c: 3 }, 'a', 'b', 'c'), 3);
  assert.equal(pickNum({ a: 'NaN' }, 'a'), null);
});

test('time helpers: getYmd + fmtDateUtc1 basic behavior', () => {
  const d = new Date(Date.UTC(2026, 2, 22, 9, 11, 0)); // 2026-03-22 09:11:00Z
  assert.equal(getYmd(d.getTime()), '20260322');
  assert.equal(fmtDateUtc1(d), '2026-03-22 10:11:00');
  assert.ok(/\d{4}-\d{2}-\d{2}T/.test(nowIso()));
});

test('time.makeGetJournalPath: stable naming', () => {
  const base = '/tmp/base';
  const getJournalPath = makeGetJournalPath(base);
  const d = new Date(Date.UTC(2026, 2, 22, 0, 0, 0));
  const p = getJournalPath(d.getTime());
  assert.equal(p, path.join(base, 'trade_journal_20260322.jsonl'));
});

test('market-regime: computeSmaPair + computeAtr basic sanity', () => {
  const closes = [100, 101, 102, 103, 104, 105];
  const { sma, smaPrev } = computeSmaPair(closes, 3);
  assert.equal(sma, (103 + 104 + 105) / 3);
  assert.equal(smaPrev, (102 + 103 + 104) / 3);

  // klines: [openTime, open, high, low, close, volume, ...]
  const klines = [];
  // build 20 candles + one in-progress
  let t = Date.UTC(2026, 2, 22, 0, 0, 0);
  for (let i = 0; i < 21; i++) {
    const o = 100 + i;
    const h = o + 2;
    const l = o - 2;
    const c = o + 1;
    klines.push([t, String(o), String(h), String(l), String(c), '1']);
    t += 60 * 1000;
  }
  const { atrPct } = computeAtr(klines, 14);
  assert.ok(atrPct > 0);

  const rv = computeRealizedVol(closes.concat([106, 107, 108, 109, 110]), 5);
  assert.ok(rv >= 0);

  const reg = detectMarketRegime({ atrPct: 0.0004, realizedVol: 0.0003, smaSlope: 0.1, lastClose: 70000, priceAboveSma: true });
  assert.ok(reg && typeof reg === 'object');
  assert.ok(['LOW_VOL', 'MID_VOL', 'HIGH_VOL'].includes(reg.volRegime));
});
