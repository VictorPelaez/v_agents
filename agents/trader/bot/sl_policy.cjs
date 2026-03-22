'use strict';

/**
 * SL policy evaluation helpers.
 *
 * This module is deliberately small and pure(ish) so it is easy to test and reason about.
 * It does NOT place orders; it only decides whether a trade should be closed.
 */

function evalSlPolicy(params) {
  const {
    slPolicy,
    market,
    slClassic,
    slEmergency,
    nowMs,
    breachStartMs,
    slConfirmSeconds,
  } = params || {};

  const policy = String(slPolicy || 'CLASSIC').toUpperCase();
  const m = Number(market);
  const now = Number.isFinite(Number(nowMs)) ? Number(nowMs) : Date.now();
  const confirmMs = Math.max(0, Number(slConfirmSeconds) || 0) * 1000;

  let nextBreachStartMs = breachStartMs || null;

  // Emergency (airbag)
  if ((policy === 'SOFT_CLASSIC_WITH_EMERGENCY' || policy === 'EMERGENCY_ONLY')) {
    const se = (slEmergency == null) ? null : Number(slEmergency);
    if (Number.isFinite(se) && Number.isFinite(m) && m <= se) {
      return {
        closeReason: 'emergency_sl',
        emergency: true,
        breachStartMs: null,
      };
    }
  }

  // Classic SL
  if (policy === 'CLASSIC') {
    const sc = (slClassic == null) ? null : Number(slClassic);
    if (Number.isFinite(sc) && Number.isFinite(m) && m <= sc) {
      return { closeReason: 'SL', emergency: false, breachStartMs: null };
    }
    return { closeReason: null, emergency: false, breachStartMs: null };
  }

  // Soft classic with confirmation
  if (policy === 'SOFT_CLASSIC_WITH_EMERGENCY') {
    const sc = (slClassic == null) ? null : Number(slClassic);

    if (Number.isFinite(sc) && Number.isFinite(m) && m <= sc) {
      if (!nextBreachStartMs) nextBreachStartMs = now;
      if (confirmMs <= 0 || (now - nextBreachStartMs) >= confirmMs) {
        return { closeReason: 'SL', emergency: false, breachStartMs: null };
      }
      return { closeReason: null, emergency: false, breachStartMs: nextBreachStartMs };
    }

    // Price recovered above classic SL: reset timer
    return { closeReason: null, emergency: false, breachStartMs: null };
  }

  // Emergency only (but if we get here, emergency didn't trigger)
  if (policy === 'EMERGENCY_ONLY') {
    return { closeReason: null, emergency: false, breachStartMs: null };
  }

  // Unknown policy -> safest fallback: behave as classic
  const sc = (slClassic == null) ? null : Number(slClassic);
  if (Number.isFinite(sc) && Number.isFinite(m) && m <= sc) {
    return { closeReason: 'SL', emergency: false, breachStartMs: null };
  }
  return { closeReason: null, emergency: false, breachStartMs: null };
}

module.exports = {
  evalSlPolicy,
};
