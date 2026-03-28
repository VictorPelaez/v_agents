'use strict';

// MARKET BIAS (logging only)
// Returns a coarse market state label for ITER_SUMMARY logging.
function getMarketBias({ regime, microRegime, slopeNorm, trendUp, priceAboveSMA, adx, diPlus, diMinus }) {
  // Bearish
  if (!priceAboveSMA && slopeNorm < 0.10) return 'BEARISH';
  if (adx > 40 && diMinus > diPlus) return 'STRONG_BEARISH';

  // Strong bullish
  if (microRegime === 'TRENDING' && trendUp && priceAboveSMA && adx > 40 && diPlus > diMinus) {
    return 'STRONG_BULLISH';
  }

  // Moderate bullish
  if (trendUp && priceAboveSMA && slopeNorm >= 0.10) return 'BULLISH';

  // Neutral / choppy
  if (microRegime === 'CHOPPY') return 'NEUTRAL_CHOPPY';
  if (regime === 'LOW_VOL') return 'NEUTRAL_LOW_VOL';

  return 'NEUTRAL';
}

module.exports = { getMarketBias };
