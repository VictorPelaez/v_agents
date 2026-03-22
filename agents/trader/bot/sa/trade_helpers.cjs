'use strict';

function buildCloseReason(trade, market) {
  if (trade.stopLoss && market <= trade.stopLoss) return 'SL';
  if (trade.takeProfit && market >= trade.takeProfit) return 'TP';
  return 'OTHER';
}

module.exports = {
  buildCloseReason,
};
