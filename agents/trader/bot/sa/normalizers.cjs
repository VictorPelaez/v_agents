'use strict';

/** Move-only refactor: trade normalizers for journal OPEN/CLOSE events. */

function createNormalizers(ctx) {
  const { label, nowIso, getSignalKey } = ctx || {};
  if (!label) throw new Error('createNormalizers: label required');
  if (!nowIso) throw new Error('createNormalizers: nowIso required');
  if (!getSignalKey) throw new Error('createNormalizers: getSignalKey required');

  function normalizeOpenTrade(trade, candleBucketMs) {
    return {
      id: trade.id,
      label: label,
      // Additive: execution context (paper/live + maker/taker). Safe for old parsers.
      mode: trade.mode || null,
      execution_entry: trade.executionEntry || trade.execution_entry || null,
      fee_rate_entry: (trade.feeRateEntry != null) ? Number(trade.feeRateEntry) : null,
      sl_policy: trade.slPolicy || null,
      sl_confirm_seconds: (trade.slConfirmSeconds != null) ? Number(trade.slConfirmSeconds) : null,
      entry_order_id: trade.entryOrderId || null,
      entry_client_order_id: trade.entryClientOrderId || null,
      tp_order_id: trade.tpOrderId || null,
      tp_client_order_id: trade.tpClientOrderId || null,
      symbol: trade.symbol || 'BTCUSDT',
      open_time_iso: trade.openedAt || nowIso(),
      signal_candle_ts: trade.signalCandleTs || null,
      side: trade.type || 'LONG',
      qty: trade.size ?? 0,
      entry_price: trade.entryPrice ?? null,
      sl: (trade.stopLossClassic != null)
        ? trade.stopLossClassic
        : (trade.stopLoss != null ? trade.stopLoss : null),
      sl_classic: (trade.stopLossClassic != null) ? trade.stopLossClassic : null,
      sl_emergency: (trade.stopLossEmergency != null) ? trade.stopLossEmergency : null,
      tp: trade.takeProfit ?? null,
      // Scale-out TP (additive)
      scale_tp_enabled: !!trade.scaleTpEnabled,
      tp1: (trade.tp1Price != null) ? trade.tp1Price : null,
      tp2: (trade.tp2Price != null) ? trade.tp2Price : (trade.takeProfit ?? null),
      tp1_frac: (trade.tp1Frac != null) ? trade.tp1Frac : null,
      tp1_hit: !!trade.tp1Hit,
      realized_gross: (trade.realizedGross != null) ? Number(trade.realizedGross) : null,
      realized_fee_usd_est: (trade.realizedFeeUsdEst != null) ? Number(trade.realizedFeeUsdEst) : null,
      reason_tag: trade.reasonTag || '',
      reason_details: trade.reasonDetails || {},
      exposure_usd: trade.exposureUSD ?? null,
      signal_key: getSignalKey(trade, candleBucketMs)
    };
  }

  function normalizeCloseTrade(trade, closeReason, candleBucketMs) {
    const openedAt = trade.openedAt || '';
    const closedAt = trade.closedAt || nowIso();
    const durationS = openedAt
      ? Math.round((new Date(closedAt).getTime() - new Date(openedAt).getTime()) / 1000)
      : null;

    return {
      id: trade.id,
      label: label,
      // Additive: execution context (paper/live + maker/taker). Safe for old parsers.
      mode: trade.mode || null,
      execution_entry: trade.executionEntry || trade.execution_entry || null,
      execution_exit: trade.executionExit || trade.execution_exit || null,
      fee_rate_entry: (trade.feeRateEntry != null) ? Number(trade.feeRateEntry) : null,
      fee_rate_exit: (trade.feeRateExit != null) ? Number(trade.feeRateExit) : null,
      sl_policy: trade.slPolicy || null,
      sl_confirm_seconds: (trade.slConfirmSeconds != null) ? Number(trade.slConfirmSeconds) : null,
      entry_order_id: trade.entryOrderId || null,
      entry_client_order_id: trade.entryClientOrderId || null,
      tp_order_id: trade.tpOrderId || null,
      tp_client_order_id: trade.tpClientOrderId || null,
      fee_usd_real: (trade.feeUsdReal != null) ? Number(trade.feeUsdReal) : null,
      profit_after_fees_real: (trade.profitAfterFeesReal != null) ? Number(trade.profitAfterFeesReal) : null,
      symbol: trade.symbol || 'BTCUSDT',
      open_time_iso: openedAt,
      close_time_iso: closedAt,
      duration_s: durationS,
      side: trade.type || 'LONG',
      qty: trade.size ?? 0,
      entry_price: trade.entryPrice ?? null,
      exit_price: trade.exitPrice ?? null,
      sl: (trade.stopLossClassic != null)
        ? trade.stopLossClassic
        : (trade.stopLoss != null ? trade.stopLoss : null),
      sl_classic: (trade.stopLossClassic != null) ? trade.stopLossClassic : null,
      sl_emergency: (trade.stopLossEmergency != null) ? trade.stopLossEmergency : null,
      tp: trade.takeProfit ?? null,
      // Scale-out TP (additive)
      scale_tp_enabled: !!trade.scaleTpEnabled,
      tp1: (trade.tp1Price != null) ? trade.tp1Price : null,
      tp2: (trade.tp2Price != null) ? trade.tp2Price : (trade.takeProfit ?? null),
      tp1_frac: (trade.tp1Frac != null) ? trade.tp1Frac : null,
      tp1_hit: !!trade.tp1Hit,
      realized_gross: (trade.realizedGross != null) ? Number(trade.realizedGross) : null,
      realized_fee_usd_est: (trade.realizedFeeUsdEst != null) ? Number(trade.realizedFeeUsdEst) : null,

      profit: trade.profit ?? null,
      profit_pct: trade.profit_pct ?? null,
      fee_usd_est: trade.feeUsdEst ?? null,
      profit_after_fees_est: (trade.profit != null && trade.feeUsdEst != null) ? (trade.profit - trade.feeUsdEst) : null,
      reason_tag: trade.reasonTag || '',
      close_reason: closeReason || '',
      reason_details: trade.reasonDetails || {},
      exposure_usd: trade.exposureUSD ?? null,
      signal_key: getSignalKey(trade, candleBucketMs)
    };
  }

  return { normalizeOpenTrade, normalizeCloseTrade };
}

module.exports = {
  createNormalizers,
};
