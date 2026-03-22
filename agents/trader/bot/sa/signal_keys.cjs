'use strict';

function getSignalBucketMs(signalCandleTs, bucketMs) {
  if (!signalCandleTs) return 0;
  return Math.floor(Number(signalCandleTs) / bucketMs);
}

function createSignalKeys(ctx) {
  const { label } = ctx || {};
  if (!label) throw new Error('createSignalKeys: label required');

  function getSignalKey(trade, candleBucketMs) {
    const symbol = trade.symbol || 'BTCUSDT';
    const side = trade.type || trade.side || 'LONG';
    const reasonTag = trade.reasonTag || trade.reason_tag || '';
    const bucket = getSignalBucketMs(trade.signalCandleTs || trade.signal_candle_ts, candleBucketMs);
    return `signal:${label}:${symbol}:${side}:${bucket}:${reasonTag}`;
  }

  function getOpenEventKey(trade, candleBucketMs) {
    return `open:${getSignalKey(trade, candleBucketMs)}`;
  }

  function getCloseEventKey(trade, closeReason) {
    return `close:${label}:${String(trade.id)}:${closeReason || 'UNKNOWN'}`;
  }

  return {
    getSignalBucketMs,
    getSignalKey,
    getOpenEventKey,
    getCloseEventKey,
  };
}

module.exports = {
  createSignalKeys,
};
