'use strict';

/**
 * LIVE execution engine for MEXC spot.
 *
 * Responsibilities:
 * - openTradeLive: maker entry with fallback taker (market)
 * - closeTradeLiveMarket: cancel TP then market exit
 * - reconcileLiveTpOrders: attach/recreate TP orders + emit CLOSE if TP filled while down
 * - persistTradeUpdate: emit UPDATE events to journal to patch open trades
 *
 * This module is intentionally dependency-injected: bot passes in journal/state helpers.
 */

function pickNum(obj, ...keys) {
  for (const k of keys) {
    const v = obj?.[k];
    const n = (v == null) ? NaN : Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

// MEXC constraint (per error 700008): newClientOrderId must match ^[0-9a-zA-Z_-]{1,32}$
function sanitizeClientOrderId(id) {
  const s = String(id ?? '')
    .replace(/[^0-9a-zA-Z_-]/g, '_')
    .slice(0, 32);
  // Ensure non-empty
  return s.length ? s : 'cid_' + Date.now().toString().slice(-8);
}

function createLiveExecutorMexc(ctx) {
  const {
    mexc,
    label,
    symbols,
    candleBucketMs,
    httpTimeoutMs,
    makerEntryTimeoutMs,
    makerEntryMaxAttempts: makerEntryMaxAttemptsCfg,
    makerEntryRetrySleepMs: makerEntryRetrySleepMsCfg,
    makerEntryOnly,
    orderPollMs,
    tpOnExchange,
    feeRateMaker,
    feeRateTaker,
    // state/journal wiring
    state,
    nowIso,
    getSignalKey,
    hasOpenSignal,
    canOpenSignal,
    getOpenEventKey,
    persistJournalEvent,
    normalizeOpenTrade,
    closeTrade,
    getTickerCached,
    verbose,
  } = ctx || {};

  if (!mexc) throw new Error('createLiveExecutorMexc: mexc client required');
  if (!label) throw new Error('createLiveExecutorMexc: label required');

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Maker entry retries (in addition to MAKER_ENTRY_TIMEOUT_MS).
  // Priority: ctx/config -> env -> default.
  const makerEntryMaxAttempts = Number(
    (makerEntryMaxAttemptsCfg != null ? makerEntryMaxAttemptsCfg : (process.env.MAKER_ENTRY_MAX_ATTEMPTS || 3))
  );
  const makerEntryRetrySleepMs = Number(
    (makerEntryRetrySleepMsCfg != null ? makerEntryRetrySleepMsCfg : (process.env.MAKER_ENTRY_RETRY_SLEEP_MS || 750))
  );

  function persistTradeUpdate(patch) {
    try {
      const id = patch?.id;
      if (!id) return false;
      const event = {
        ts: nowIso(),
        type: 'UPDATE',
        event_key: `update:${label}:${id}:${Date.now()}`,
        trade: patch,
      };
      return persistJournalEvent(event);
    } catch (_) {
      return false;
    }
  }

  async function openTradeLive(trade, signalCooldownMs) {
    const sym = trade.symbol;
    const signalKey = getSignalKey(trade, candleBucketMs);

    if (hasOpenSignal(signalKey)) return false;
    if (!canOpenSignal(signalKey, signalCooldownMs)) return false;

    // Mark as seen immediately to avoid duplicate opens while the entry order is pending.
    state.recentSignalSeenAt.set(signalKey, Date.now());

    // Normalize qty to stepSize
    const { qty: normQty, rules } = await mexc.normalizeQuantity(sym, trade.size, httpTimeoutMs);
    trade.size = normQty;

    if (rules.minQty && trade.size < rules.minQty) {
      console.error('LIVE entry: qty below minQty', { symbol: sym, qty: trade.size, minQty: rules.minQty });
      return false;
    }

    let entryExec = 'maker';
    let entryFeeRate = feeRateMaker;
    let entryClientId = sanitizeClientOrderId(`open_${label}_${trade.id}`);
    const entryClientIdBase = entryClientId;

    let entryOrder = null;
    let entryOrderId = null;

    // Partial-fill aware entry handling (maker)
    let makerOrd = null;
    let makerExecQty = 0;
    let makerAvg = null;

    // 1) Entry: post-only LIMIT near best bid (MEXC Spot v3)
    // NOTE: MEXC does not reliably support Binance-style type=LIMIT_MAKER; use LIMIT + timeInForce=GTX (post-only).
    const maxAttempts = Math.max(1, Number.isFinite(makerEntryMaxAttempts) ? makerEntryMaxAttempts : 1);
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      // Make client id unique per attempt (and keep it MEXC-legal)
      entryClientId = sanitizeClientOrderId(`${entryClientIdBase}_a${attempt}`);
      entryOrder = null;
      entryOrderId = null;
      makerOrd = null;
      makerExecQty = 0;
      makerAvg = null;

      try {
        const bt = await mexc.bookTicker(sym, httpTimeoutMs);
        const bid = pickNum(bt, 'bidPrice', 'bid');
        const _makerOffset = Number.isFinite(ctx.makerEntryPriceOffsetPct) ? ctx.makerEntryPriceOffsetPct : 0;
        let price = (bid != null && bid > 0)
          ? bid * (1 - _makerOffset)
          : trade.entryPrice * (1 - _makerOffset);
        const norm = await mexc.normalizeLimit(sym, trade.size, price, httpTimeoutMs);

        entryOrder = await mexc.placeOrder({
          symbol: sym,
          side: 'BUY',
          type: 'LIMIT',
          quantity: norm.qty,
          price: norm.price,
          timeInForce: 'GTX',
          newClientOrderId: entryClientId,
        }, httpTimeoutMs);

        entryOrderId = entryOrder?.orderId || entryOrder?.order_id || null;
      } catch (e) {
        const status = e?.response?.status;
        const data = e?.response?.data;
        console.error('LIVE entry maker place failed:', { symbol: sym, status, data, message: e?.message });
      }

      if (entryOrderId) {
        makerOrd = await mexc.waitForFill({
          symbol: sym,
          orderId: entryOrderId,
          origClientOrderId: entryClientId,
          timeoutMs: makerEntryTimeoutMs,
          pollMs: orderPollMs,
          httpTimeoutMs,
        });

        try {
          const ordNow = makerOrd || await mexc.getOrder({ symbol: sym, orderId: entryOrderId, origClientOrderId: entryClientId }, httpTimeoutMs);
          if (ordNow) makerOrd = ordNow;
        } catch (_) {}

        makerExecQty = pickNum(makerOrd, 'executedQty', 'executedQuantity', 'cumulativeQuantity') || 0;
        makerAvg = mexc.orderAvgFillPrice(makerOrd);
      }

      const filledEnoughMaker = makerExecQty >= (trade.size * 0.999999);
      if (filledEnoughMaker) break;

      // Cancel any remaining maker quantity before retrying
      try {
        if (entryOrderId) await mexc.cancelOrder({ symbol: sym, orderId: entryOrderId }, httpTimeoutMs);
      } catch (_) {}

      // Sleep a bit and retry (refresh bid)
      if (attempt < maxAttempts) await sleep(Math.max(0, makerEntryRetrySleepMs));
    }

    let mktOrd = null;
    let mktExecQty = 0;
    let mktAvg = null;

    const filledEnough = makerExecQty >= (trade.size * 0.999999);

    if (!filledEnough) {
      // Cancel any remaining maker quantity
      try {
        if (entryOrderId) {
          await mexc.cancelOrder({ symbol: sym, orderId: entryOrderId }, httpTimeoutMs);
        }
      } catch (_) {}

      // If configured, do NOT fallback to taker/market. Abort the entry.
      if (makerEntryOnly) {
        if (verbose) console.log('LIVE entry: maker-only enabled; not filled -> abort', { symbol: sym, wantedQty: trade.size, makerExecQty });
        return false;
      }

      let remaining = Math.max(0, trade.size - makerExecQty);
      remaining = (await mexc.normalizeQuantity(sym, remaining, httpTimeoutMs)).qty;

      if (remaining > 0 && (!rules.minQty || remaining >= rules.minQty)) {
        const mktClientId = sanitizeClientOrderId(`${entryClientId}_mkt`);
        const mkt = await mexc.placeOrder({
          symbol: sym,
          side: 'BUY',
          type: 'MARKET',
          quantity: remaining,
          newClientOrderId: mktClientId,
        }, httpTimeoutMs);

        const mktOrderId = mkt?.orderId || mkt?.order_id || null;
        mktOrd = mktOrderId
          ? await mexc.waitForFill({ symbol: sym, orderId: mktOrderId, origClientOrderId: mktClientId, timeoutMs: 15000, pollMs: orderPollMs, httpTimeoutMs })
          : null;

        mktExecQty = pickNum(mktOrd, 'executedQty', 'executedQuantity', 'cumulativeQuantity') || 0;
        mktAvg = mexc.orderAvgFillPrice(mktOrd);

        entryOrderId = mktOrderId || entryOrderId;
      }

      entryExec = (makerExecQty > 0 && mktExecQty > 0) ? 'maker+fallback_taker' : 'fallback_taker';
      entryFeeRate = feeRateTaker;
    } else {
      entryExec = 'maker';
      entryFeeRate = feeRateMaker;
    }

    const execQty = makerExecQty + mktExecQty;
    const quoteMaker = (makerAvg != null ? makerAvg : trade.entryPrice) * makerExecQty;
    const quoteMkt = (mktAvg != null ? mktAvg : trade.entryPrice) * mktExecQty;
    const avgEntry = execQty > 0 ? ((quoteMaker + quoteMkt) / execQty) : null;

    if (!Number.isFinite(execQty) || execQty <= 0 || !Number.isFinite(avgEntry) || avgEntry <= 0) {
      console.error('LIVE entry failed (no fills)', { symbol: sym, id: trade.id, entryOrderId });
      return false;
    }

    // Update trade with actual execution details
    trade.mode = 'live';
    trade.executionEntry = entryExec;
    trade.feeRateEntry = entryFeeRate;
    trade.entryOrderId = entryOrderId;
    trade.entryClientOrderId = entryClientId;

    trade.size = execQty;
    trade.entryPrice = avgEntry;
    trade.openedAt = nowIso();

    // 2) Place TP order on exchange (LIMIT sell).
    if (tpOnExchange && trade.takeProfit && Number.isFinite(trade.takeProfit)) {
      try {
        const tpClientId = sanitizeClientOrderId(`tp_${label}_${trade.id}`);
        const tpNorm = await mexc.normalizeLimit(sym, trade.size, trade.takeProfit, httpTimeoutMs);
        const tp = await mexc.placeOrder({
          symbol: sym,
          side: 'SELL',
          type: 'LIMIT',
          quantity: tpNorm.qty,
          price: tpNorm.price,
          timeInForce: 'GTC',
          newClientOrderId: tpClientId,
        }, httpTimeoutMs);

        trade.tpOrderId = tp?.orderId || tp?.order_id || null;
        trade.tpClientOrderId = tpClientId;
      } catch (e) {
        console.error('LIVE TP place failed:', e.message);
      }
    }

    const event = {
      ts: nowIso(),
      type: 'OPEN',
      event_key: getOpenEventKey(trade, candleBucketMs),
      trade: normalizeOpenTrade(trade, candleBucketMs),
    };

    const ok = persistJournalEvent(event);
    if (!ok) return false;

    console.log(
      `OPEN (${trade.mode || 'live'}):`, trade.openedAt,
      'id=', trade.id,
      'symbol=', trade.symbol,
      'entry=', trade.entryPrice,
      'exec=', trade.executionEntry,
      'entryOrderId=', trade.entryOrderId,
      'tpOrderId=', trade.tpOrderId || '—'
    );

    return true;
  }

  async function computeFeeUsdRealForOrderId(symbol, orderId) {
    try {
      const fills = await mexc.myTrades({ symbol, orderId }, httpTimeoutMs);
      const arr = Array.isArray(fills) ? fills : (fills?.data || []);
      if (!Array.isArray(arr) || arr.length === 0) return { feeUsd: null, feeByAsset: null, ts: null };

      const feeByAsset = {};
      let t0 = null;
      for (const t of arr) {
        const asset = String(t?.commissionAsset || '').toUpperCase();
        const c = Number(t?.commission);
        const tt = Number(t?.time);
        if (!asset || !Number.isFinite(c)) continue;
        feeByAsset[asset] = (feeByAsset[asset] || 0) + c;
        if (Number.isFinite(tt)) t0 = (t0 == null) ? tt : Math.min(t0, tt);
      }

      let feeUsd = 0;
      for (const [asset, amt] of Object.entries(feeByAsset)) {
        if (!Number.isFinite(amt) || amt <= 0) continue;
        if (asset === 'USDT') {
          feeUsd += amt;
          continue;
        }
        const pair = `${asset}USDT`;
        // Use 1m kline around the earliest fill time.
        const ts = (t0 != null) ? t0 : Date.now();
        const kl = await mexc.klines({ symbol: pair, interval: '1m', startTime: ts - 60000, endTime: ts + 60000, limit: 3 }, httpTimeoutMs);
        const k = Array.isArray(kl) && kl.length ? kl[kl.length - 1] : null;
        const px = k ? Number(k[4]) : NaN; // close
        if (Number.isFinite(px) && px > 0) {
          feeUsd += amt * px;
        } else {
          // Can't convert -> return null to avoid writing incorrect "real" fees.
          return { feeUsd: null, feeByAsset, ts };
        }
      }

      return { feeUsd, feeByAsset, ts: t0 };
    } catch (e) {
      if (verbose) console.error('computeFeeUsdRealForOrderId failed:', e.message);
      return { feeUsd: null, feeByAsset: null, ts: null };
    }
  }

  async function closeTradeLiveMarket(trade, closeReason) {
    const sym = trade.symbol;
    let qty = (await mexc.normalizeQuantity(sym, trade.size, httpTimeoutMs)).qty;

    // Cancel TP if present
    try {
      if (trade.tpOrderId) {
        await mexc.cancelOrder({ symbol: sym, orderId: trade.tpOrderId }, httpTimeoutMs);
      }
    } catch (_) {}

    const clientId = sanitizeClientOrderId(`c_${label}_${Date.now()}_${trade.id}_mkt`);
    const mkt = await mexc.placeOrder({
      symbol: sym,
      side: 'SELL',
      type: 'MARKET',
      quantity: qty,
      newClientOrderId: clientId,
    }, httpTimeoutMs);

    const orderId = mkt?.orderId || mkt?.order_id || null;
    const filled = orderId
      ? await mexc.waitForFill({ symbol: sym, orderId, origClientOrderId: clientId, timeoutMs: 15000, pollMs: orderPollMs, httpTimeoutMs })
      : null;

    const avgExit = mexc.orderAvgFillPrice(filled) ?? (await getTickerCached(sym, httpTimeoutMs, 0));

    // Real fee (best-effort): sum entry+exit commissions converted to USDT using 1m kline of commission asset.
    try {
      const entryOrderId = trade.entryOrderId;
      const exitOrderId = orderId;
      const entryFee = entryOrderId ? await computeFeeUsdRealForOrderId(sym, entryOrderId) : { feeUsd: null };
      const exitFee = exitOrderId ? await computeFeeUsdRealForOrderId(sym, exitOrderId) : { feeUsd: null };
      const feeUsdReal = (Number.isFinite(entryFee.feeUsd) ? entryFee.feeUsd : 0) + (Number.isFinite(exitFee.feeUsd) ? exitFee.feeUsd : 0);
      if (feeUsdReal > 0) {
        trade.feeUsdReal = feeUsdReal;
        // Precompute profit after real fees (closeTrade will compute profit the same way).
        const qtyNum = Number(trade.size || 0);
        const realizedGross = Number(trade.realizedGross || 0);
        const profit = realizedGross + (avgExit - trade.entryPrice) * qtyNum;
        trade.profitAfterFeesReal = profit - feeUsdReal;
      }
    } catch (_) {}

    trade.mode = 'live';
    trade.executionExit = 'taker';
    trade.feeRateExit = feeRateTaker;

    return closeTrade(trade, avgExit, closeReason, candleBucketMs, feeRateTaker);
  }

  // Maker-first exit (best-effort) with fallback to market for remaining qty.
  // Use ONLY when it's acceptable that the maker leg might not fill quickly (e.g., time_stop).
  async function closeTradeLiveMakerFirst(trade, closeReason, opts) {
    const sym = trade.symbol;
    const makerTimeoutMs = Number(opts?.makerTimeoutMs ?? ctx?.makerCloseTimeoutMs ?? 8000);

    // DEBUG: log offset
    if (verbose) {
      console.error('[DEBUG closeTradeLiveMakerFirst] makerCloseOffsetPct:', ctx?.makerCloseOffsetPct, 'makerEntryOffset:', ctx?.makerEntryPriceOffsetPct);
    }

    let qty = (await mexc.normalizeQuantity(sym, trade.size, httpTimeoutMs)).qty;

    // Cancel TP if present
    try {
      if (trade.tpOrderId) {
        await mexc.cancelOrder({ symbol: sym, orderId: trade.tpOrderId }, httpTimeoutMs);
      }
    } catch (_) {}

    // 1) Maker attempts (LIMIT_MAKER) with fresh bookTicker each time
    // Close-maker params (prefer dedicated config; fallback to entry params)
    const makerCloseOffsetPct = Number.isFinite(ctx?.makerCloseOffsetPct)
      ? ctx.makerCloseOffsetPct
      : (Number.isFinite(ctx?.makerEntryPriceOffsetPct) ? ctx.makerEntryPriceOffsetPct : 0.0002);

    const makerMaxAttempts = Math.max(1, Number(
      opts?.makerMaxAttempts ?? ctx?.makerCloseMaxAttempts ?? makerEntryMaxAttempts ?? 1
    ));

    const makerRetrySleepMs = Math.max(0, Number(
      opts?.makerRetrySleepMs ?? ctx?.makerCloseRetrySleepMs ?? makerEntryRetrySleepMs ?? 0
    ));

    let makerExecQty = 0;
    let makerAvg = null;
    let makerOrderId = null;

    for (let attempt = 1; attempt <= makerMaxAttempts; attempt++) {
      // IMPORTANT: clientOrderId must be unique per attempt (MEXC rejects duplicates with 400)
      const makerClientId = sanitizeClientOrderId(`c_${label}_${Date.now()}_${trade.id}_mk_a${attempt}`);
      makerOrderId = null;
      let makerOrd = null;

      try {
        const bt = await mexc.bookTicker(sym, httpTimeoutMs);
        const ask = pickNum(bt, 'askPrice', 'ask');
        const px0 = (ask != null && ask > 0) ? ask : (await getTickerCached(sym, httpTimeoutMs, 0)) || null;

        // SELL maker: place slightly BELOW ask to sit at/inside spread without crossing bid
        // (Using +offset makes it less likely to fill and can leave positions hanging.)
        const px = (px0 != null && px0 > 0) ? (px0 * (1 - makerCloseOffsetPct)) : null;

        if (px != null && px > 0) {
          const norm = await mexc.normalizeLimit(sym, qty, px, httpTimeoutMs);
          if (verbose) {
            console.error('LIVE close maker attempt:', {
              attempt,
              maxAttempts: makerMaxAttempts,
              symbol: sym,
              reason: closeReason,
              qty,
              ask: px0,
              px,
              normQty: norm.qty,
              normPrice: norm.price,
              clientId: makerClientId,
            });
          }

          const placed = await mexc.placeOrder({
            symbol: sym,
            side: 'SELL',
            type: 'LIMIT_MAKER',
            quantity: norm.qty,
            price: norm.price,
            newClientOrderId: makerClientId,
          }, httpTimeoutMs);
          makerOrderId = placed?.orderId || placed?.order_id || null;
        }
      } catch (e) {
        const resp = e?.response?.data;
        if (verbose) {
          console.error('LIVE close maker place failed:', e.message, resp ? { resp } : '');
        } else {
          console.error('LIVE close maker place failed:', e.message);
        }
      }

      if (makerOrderId) {
        makerOrd = await mexc.waitForFill({
          symbol: sym,
          orderId: makerOrderId,
          origClientOrderId: makerClientId,
          timeoutMs: makerTimeoutMs,
          pollMs: orderPollMs,
          httpTimeoutMs,
        });

        try {
          const ordNow = makerOrd || await mexc.getOrder({ symbol: sym, orderId: makerOrderId, origClientOrderId: makerClientId }, httpTimeoutMs);
          if (ordNow) makerOrd = ordNow;
        } catch (_) {}

        makerExecQty = pickNum(makerOrd, 'executedQty', 'executedQuantity', 'cumulativeQuantity') || 0;
        makerAvg = mexc.orderAvgFillPrice(makerOrd);

        // Cancel remainder of this attempt before retrying
        try {
          await mexc.cancelOrder({ symbol: sym, orderId: makerOrderId }, httpTimeoutMs);
        } catch (_) {}
      }

      const filledEnough = makerExecQty >= (qty * 0.999999);
      if (filledEnough) break;

      if (attempt < makerMaxAttempts) {
        if (makerRetrySleepMs > 0) await sleep(makerRetrySleepMs);
      }
    }

    const filledEnough = makerExecQty >= (qty * 0.999999);
    if (filledEnough) {
      const avgExit = (makerAvg != null) ? makerAvg : (await getTickerCached(sym, httpTimeoutMs, 0));

      // Real fee (best-effort): entry+exit commissions converted to USDT
      try {
        const entryOrderId = trade.entryOrderId;
        const exitOrderId = makerOrderId;
        const entryFee = entryOrderId ? await computeFeeUsdRealForOrderId(sym, entryOrderId) : { feeUsd: null };
        const exitFee = exitOrderId ? await computeFeeUsdRealForOrderId(sym, exitOrderId) : { feeUsd: null };
        const feeUsdReal = (Number.isFinite(entryFee.feeUsd) ? entryFee.feeUsd : 0) + (Number.isFinite(exitFee.feeUsd) ? exitFee.feeUsd : 0);
        if (feeUsdReal > 0) {
          trade.feeUsdReal = feeUsdReal;
          const qtyNum = Number(trade.size || 0);
          const realizedGross = Number(trade.realizedGross || 0);
          const profit = realizedGross + (avgExit - trade.entryPrice) * qtyNum;
          trade.profitAfterFeesReal = profit - feeUsdReal;
        }
      } catch (_) {}

      trade.mode = 'live';
      trade.executionExit = 'maker';
      trade.feeRateExit = feeRateMaker;
      return closeTrade(trade, avgExit, closeReason, candleBucketMs, feeRateMaker);
    }

    // 2) Maker-only mode: NO market fallback.
    // If we couldn't fully close as maker, leave the position open and let the next ticks/time_stop retries handle it.
    console.error('LIVE close maker-only: not fully filled; leaving trade open', {
      symbol: sym,
      reason: closeReason,
      qty,
      makerExecQty,
      makerMaxAttempts,
      makerTimeoutMs,
    });

    trade.mode = 'live';
    trade.executionExit = 'maker_unfilled';
    trade.feeRateExit = feeRateMaker;
    return null;
  }

  async function reconcileLiveTpOrders() {
    if (!tpOnExchange) return;

    for (const sym of (symbols || [])) {
      const openTrades = ctx.getOpenTradesArray(sym);

      // Pull open orders once per symbol
      let openOrders = [];
      try {
        const oo = await mexc.openOrders({ symbol: sym }, httpTimeoutMs);
        if (Array.isArray(oo)) openOrders = oo;
        else if (Array.isArray(oo?.data)) openOrders = oo.data;
        else if (Array.isArray(oo?.orders)) openOrders = oo.orders;
      } catch (e) {
        console.error('LIVE reconcile: openOrders fetch failed:', e.message);
        continue;
      }

      // If no open trades in journal/state, cancel any stray TP orders created by this bot.
      if (!openTrades.length) {
        for (const o of openOrders) {
          const clientId = String(o?.clientOrderId || o?.client_order_id || o?.origClientOrderId || o?.orig_client_order_id || '');
          const orderId = o?.orderId || o?.order_id || null;
          if (orderId && clientId.startsWith(`tp_${label}_`)) {
            try {
              await mexc.cancelOrder({ symbol: sym, orderId }, httpTimeoutMs);
              console.log('LIVE reconcile: canceled stray TP order', { symbol: sym, orderId, clientId });
            } catch (_) {}
          }
        }
        continue;
      }

      // For each open trade, ensure we have a TP order attached.
      for (const tr of openTrades) {
        const expectedTpClientId = sanitizeClientOrderId(`tp_${label}_${tr.id}`);

        // Attach tpOrderId if missing (look in openOrders)
        if (!tr.tpOrderId) {
          const found = openOrders.find((o) => {
            const clientId = String(o?.clientOrderId || o?.client_order_id || o?.origClientOrderId || o?.orig_client_order_id || '');
            return clientId === expectedTpClientId;
          });
          if (found) {
            tr.tpOrderId = found?.orderId || found?.order_id || null;
            tr.tpClientOrderId = expectedTpClientId;
            persistTradeUpdate({ id: tr.id, symbol: sym, tp_order_id: tr.tpOrderId, tp_client_order_id: tr.tpClientOrderId });
            console.log('LIVE reconcile: attached existing TP order to trade', { id: tr.id, symbol: sym, tpOrderId: tr.tpOrderId });
          }
        }

        // If we have a TP order id, check if it already filled while we were down.
        if (tr.tpOrderId) {
          try {
            const tpOrd = await mexc.getOrder({ symbol: sym, orderId: tr.tpOrderId }, httpTimeoutMs);
            if (mexc.isOrderFilled(tpOrd)) {
              const exitP = mexc.orderAvgFillPrice(tpOrd) ?? null;
              tr.mode = 'live';
              tr.executionExit = 'maker';
              tr.feeRateExit = feeRateMaker;
              await closeTrade(tr, exitP, 'TP', candleBucketMs, feeRateMaker);
              console.log('LIVE reconcile: TP already filled, journal close emitted', { id: tr.id, symbol: sym, tpOrderId: tr.tpOrderId });
              continue;
            }
          } catch (_) {}
        }

        // If still no TP order, place a new one.
        if (!tr.tpOrderId && tr.takeProfit && Number.isFinite(tr.takeProfit)) {
          try {
            const tpNorm = await mexc.normalizeLimit(sym, tr.size, tr.takeProfit, httpTimeoutMs);

            const tp = await mexc.placeOrder({
              symbol: sym,
              side: 'SELL',
              type: 'LIMIT',
              quantity: tpNorm.qty,
              price: tpNorm.price,
              timeInForce: 'GTC',
              newClientOrderId: expectedTpClientId,
            }, httpTimeoutMs);

            tr.tpOrderId = tp?.orderId || tp?.order_id || null;
            tr.tpClientOrderId = expectedTpClientId;
            persistTradeUpdate({ id: tr.id, symbol: sym, tp_order_id: tr.tpOrderId, tp_client_order_id: tr.tpClientOrderId });
            console.log('LIVE reconcile: placed missing TP order', { id: tr.id, symbol: sym, tpOrderId: tr.tpOrderId });
          } catch (e) {
            console.error('LIVE reconcile: TP place failed:', e.message);
          }
        }
      }
    }
  }

  return {
    openTradeLive,
    closeTradeLiveMarket,
    closeTradeLiveMakerFirst,
    reconcileLiveTpOrders,
    persistTradeUpdate,
  };
}

module.exports = {
  createLiveExecutorMexc,
  _sanitizeClientOrderId: sanitizeClientOrderId,
};
