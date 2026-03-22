'use strict';

/**
 * Journal + open-positions snapshot state management.
 *
 * IMPORTANT: This is a move-only refactor from trading-bot.live.service.sa.cjs.
 * No logic changes intended.
 */

function createJournalStateManager(ctx) {
  const {
    state,
    label,
    getJournalPath,
    nowIso,
    loadJsonl,
    appendJsonl,
    writeJsonFileAtomic,
    openPositionsPath,
    crypto,
    noteOpenTimestamp,
    getYmd,
  } = ctx || {};

  if (!state) throw new Error('createJournalStateManager: state required');
  if (!label) throw new Error('createJournalStateManager: label required');
  if (!getJournalPath) throw new Error('createJournalStateManager: getJournalPath required');
  if (!nowIso) throw new Error('createJournalStateManager: nowIso required');
  if (!loadJsonl) throw new Error('createJournalStateManager: loadJsonl required');
  if (!appendJsonl) throw new Error('createJournalStateManager: appendJsonl required');
  if (!writeJsonFileAtomic) throw new Error('createJournalStateManager: writeJsonFileAtomic required');
  if (!openPositionsPath) throw new Error('createJournalStateManager: openPositionsPath required');
  if (!crypto) throw new Error('createJournalStateManager: crypto required');
  if (!noteOpenTimestamp) throw new Error('createJournalStateManager: noteOpenTimestamp required');
  if (!getYmd) throw new Error('createJournalStateManager: getYmd required');

  let snapshotTimer = null;
  let snapshotDirty = false;
  let lastSnapshotHash = '';

  function ensureOpenTradesSymbolMap(symbol) {
    if (!state.openTradesBySymbol.has(symbol)) state.openTradesBySymbol.set(symbol, new Map());
    return state.openTradesBySymbol.get(symbol);
  }

  function markSnapshotDirty() {
    snapshotDirty = true;
  }

  function applyJournalEvent(event) {
    if (!event || !event.event_key) return;
    state.seenEventKeys.add(event.event_key);

    if (event.type === 'OPEN') {
      const t = event.trade;
      const trade = {
        id: t.id,
        symbol: t.symbol || 'BTCUSDT',
        entryPrice: Number(t.entry_price ?? 0),
        stopLoss: t.sl != null ? Number(t.sl) : null,
        stopLossClassic: (t.sl_classic != null) ? Number(t.sl_classic) : null,
        stopLossEmergency: (t.sl_emergency != null) ? Number(t.sl_emergency) : null,
        takeProfit: t.tp != null ? Number(t.tp) : null,
        openedAt: t.open_time_iso,
        signalCandleTs: t.signal_candle_ts,
        size: Number(t.qty ?? 0),
        type: t.side || 'LONG',
        reasonTag: t.reason_tag || '',
        reasonDetails: t.reason_details || {},
        exposureUSD: t.exposure_usd ?? null,
        mode: t.mode || null,
        executionEntry: t.execution_entry || null,
        feeRateEntry: (t.fee_rate_entry != null) ? Number(t.fee_rate_entry) : null,
        entryOrderId: t.entry_order_id || null,
        entryClientOrderId: t.entry_client_order_id || null,
        tpOrderId: t.tp_order_id || null,
        tpClientOrderId: t.tp_client_order_id || null,
        slPolicy: t.sl_policy || null,
        slConfirmSeconds: (t.sl_confirm_seconds != null) ? Number(t.sl_confirm_seconds) : null
      };

      const symbol = trade.symbol;
      const signalKey = t.signal_key;

      state.openTradesById.set(String(trade.id), trade);
      ensureOpenTradesSymbolMap(symbol).set(String(trade.id), trade);

      state.openTradeIdBySignalKey.set(signalKey, String(trade.id));
      state.recentSignalSeenAt.set(signalKey, Date.now());

      // rebuild trade-rate limiting window from journal
      const ms = Date.parse(trade.openedAt);
      if (Number.isFinite(ms)) {
        noteOpenTimestamp(ms);
        const ymd = getYmd(ms);
        const prev = state.openCountsByDay.get(ymd) || 0;
        state.openCountsByDay.set(ymd, prev + 1);
      }

      markSnapshotDirty();
    }

    if (event.type === 'CLOSE') {
      const t = event.trade;
      const id = String(t.id ?? '');
      const symbol = t.symbol || 'BTCUSDT';
      const signalKey = t.signal_key || '';

      state.openTradesById.delete(id);
      const m = state.openTradesBySymbol.get(symbol);
      if (m) m.delete(id);

      if (signalKey) state.openTradeIdBySignalKey.delete(signalKey);
      if (signalKey) state.recentSignalSeenAt.set(signalKey, Date.now());

      markSnapshotDirty();
    }

    // Additive: patch/update event (used mainly for LIVE reconciliation, e.g., attaching tp_order_id).
    if (event.type === 'UPDATE') {
      const t = event.trade || {};
      const id = String(t.id ?? '');
      if (!id) return;

      const tr = state.openTradesById.get(id);
      if (!tr) return;

      if (t.tp_order_id != null) tr.tpOrderId = t.tp_order_id;
      if (t.tp_client_order_id != null) tr.tpClientOrderId = t.tp_client_order_id;
      if (t.entry_order_id != null) tr.entryOrderId = t.entry_order_id;
      if (t.entry_client_order_id != null) tr.entryClientOrderId = t.entry_client_order_id;

      // allow updating SL levels if we ever recompute
      if (t.sl_classic != null) tr.stopLossClassic = Number(t.sl_classic);
      if (t.sl_emergency != null) tr.stopLossEmergency = Number(t.sl_emergency);

      markSnapshotDirty();
    }
  }

  function persistJournalEvent(event) {
    if (state.seenEventKeys.has(event.event_key)) return true;
    const ok = appendJsonl(getJournalPath(event.ts || nowIso()), event);
    if (!ok) return false;
    applyJournalEvent(event);
    return true;
  }

  function getOpenTradesArray(symbol) {
    if (symbol) {
      const m = state.openTradesBySymbol.get(symbol);
      return m ? Array.from(m.values()) : [];
    }
    return Array.from(state.openTradesById.values());
  }

  function computeSnapshotHash(trades) {
    const s = JSON.stringify(trades.map(t => ({
      id: t.id,
      symbol: t.symbol,
      entryPrice: t.entryPrice,
      stopLoss: t.stopLoss,
      takeProfit: t.takeProfit,
      openedAt: t.openedAt,
      size: t.size,
      reasonTag: t.reasonTag
    })));
    return crypto.createHash('sha1').update(s).digest('hex');
  }

  function flushOpenPositionsSnapshot(force = false) {
    try {
      const trades = getOpenTradesArray();
      const h = computeSnapshotHash(trades);
      if (!force && !snapshotDirty && h === lastSnapshotHash) return true;

      const ok = writeJsonFileAtomic(
        openPositionsPath,
        trades.map(t => ({
          id: t.id,
          symbol: t.symbol,
          entryPrice: t.entryPrice,
          stopLoss: t.stopLoss,
          stopLossClassic: t.stopLossClassic ?? null,
          stopLossEmergency: t.stopLossEmergency ?? null,
          takeProfit: t.takeProfit,
          mode: t.mode ?? null,
          slPolicy: t.slPolicy ?? null,
          slConfirmSeconds: t.slConfirmSeconds ?? null,
          openedAt: t.openedAt,
          signalCandleTs: t.signalCandleTs,
          size: t.size,
          type: t.type,
          reasonTag: t.reasonTag,
          reasonDetails: t.reasonDetails,
          exposureUSD: t.exposureUSD
        }))
      );

      if (ok) {
        snapshotDirty = false;
        lastSnapshotHash = h;
      }

      return ok;
    } catch (e) {
      console.error('flushOpenPositionsSnapshot err:', e.message);
      return false;
    }
  }

  function startSnapshotTimer(intervalMs = 5000) {
    snapshotTimer = setInterval(() => {
      try {
        flushOpenPositionsSnapshot(false);
      } catch (e) {
        console.error('snapshot timer err:', e.message);
      }
    }, Math.max(1000, intervalMs));
  }

  function stopSnapshotTimer() {
    try {
      if (snapshotTimer) clearInterval(snapshotTimer);
    } catch (_) {}
    snapshotTimer = null;
  }

  function rebuildStateFromJournal() {
    state.openTradesById.clear();
    state.openTradesBySymbol.clear();
    state.openTradeIdBySignalKey.clear();
    state.seenEventKeys.clear();
    state.recentSignalSeenAt.clear();
    state.recentOpenTimes = [];
    state.openCountsByDay.clear();

    // reset OPEN_TRADE spam guard
    state.lastOpenTradesLogHash = '';
    state.lastOpenTradesLogCount = 0;

    const events = loadJsonl(getJournalPath(nowIso()));
    for (const ev of events) applyJournalEvent(ev);

    flushOpenPositionsSnapshot(true);
  }

  function cleanupRecentSignals(maxAgeMs = 10 * 60 * 1000) {
    const now = Date.now();
    for (const [k, ts] of state.recentSignalSeenAt.entries()) {
      if ((now - ts) > maxAgeMs) state.recentSignalSeenAt.delete(k);
    }
  }

  function hasOpenSignal(signalKey) {
    return state.openTradeIdBySignalKey.has(signalKey);
  }

  function canOpenSignal(signalKey, cooldownMs) {
    const now = Date.now();
    const lastSeen = state.recentSignalSeenAt.get(signalKey) || 0;
    return (now - lastSeen) >= cooldownMs;
  }

  return {
    // journal
    applyJournalEvent,
    persistJournalEvent,
    rebuildStateFromJournal,

    // snapshot
    flushOpenPositionsSnapshot,
    startSnapshotTimer,
    stopSnapshotTimer,

    // misc
    getOpenTradesArray,
    cleanupRecentSignals,
    hasOpenSignal,
    canOpenSignal,

    // for visibility/testing
    _ensureOpenTradesSymbolMap: ensureOpenTradesSymbolMap,
  };
}

module.exports = {
  createJournalStateManager,
};
