'use strict';

/** Move-only refactor from service.sa: trade-rate limiting helpers. */

function createRateLimiter(ctx) {
  const { state, getYmd } = ctx || {};
  if (!state) throw new Error('createRateLimiter: state required');
  if (!getYmd) throw new Error('createRateLimiter: getYmd required');

  function pruneRecentOpens(nowMs, windowMs) {
    const w = Math.max(1000, windowMs || 3600000);
    state.recentOpenTimes = state.recentOpenTimes.filter(ts => (nowMs - ts) <= w);
  }

  function noteOpenTimestamp(ms) {
    if (!Number.isFinite(ms)) return;
    const nowMs = Date.now();
    state.recentOpenTimes.push(ms);
    pruneRecentOpens(nowMs, 3600000);
  }

  function countOpensLastHour() {
    const nowMs = Date.now();
    pruneRecentOpens(nowMs, 3600000);
    return state.recentOpenTimes.length;
  }

  function countOpensTodayUtc() {
    const ymd = getYmd(Date.now());
    return state.openCountsByDay.get(ymd) || 0;
  }

  return {
    pruneRecentOpens,
    noteOpenTimestamp,
    countOpensLastHour,
    countOpensTodayUtc,
  };
}

module.exports = {
  createRateLimiter,
};
