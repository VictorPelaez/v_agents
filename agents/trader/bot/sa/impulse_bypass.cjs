/**
 * impulse_bypass.cjs
 *
 * Helper para IMPULSE3_BYPASS_EXPLOSIVE_N:
 * - Mantiene contador `rt.impulseCount`.
 * - Evalúa si debe bypassear el bloqueo `explosive` tras N velas consecutivas con criteria.
 *
 * Uso:
 *   const { evaluateImpulseBypass } = require('./bot/sa/impulse_bypass.cjs');
 *   const { explosiveBlock, impulseCount } = evaluateImpulseBypass({
 *     rt, cfg, MIN_MOMENTUM_PCT, momentum_pct, volumeOk, priceAboveSMA, candleExplosive
 *   });
 */

function evaluateImpulseBypass({
  rt,
  cfg,
  MIN_MOMENTUM_PCT,
  momentum_pct,
  volumeOk,
  priceAboveSMA,
  candleExplosive
}) {
  const impulseBypassN = Number.isFinite(cfg.IMPULSE3_BYPASS_EXPLOSIVE_N) ? Number(cfg.IMPULSE3_BYPASS_EXPLOSIVE_N) : 0;

  // Si no está activado (N <= 1), resetea contador y no bypassea.
  if (impulseBypassN <= 1) {
    rt.impulseCount = 0;
    return { explosiveBlock: !!candleExplosive, impulseCount: 0 };
  }

  const impulseCriteria = !!candleExplosive && !!volumeOk && !!priceAboveSMA && (momentum_pct >= MIN_MOMENTUM_PCT);
  if (impulseCriteria) {
    rt.impulseCount = (rt.impulseCount || 0) + 1;
    if (rt.impulseCount > impulseBypassN) rt.impulseCount = impulseBypassN;
  } else {
    rt.impulseCount = 0;
  }

  const explosiveBlock = candleExplosive && !(rt.impulseCount >= impulseBypassN);
  return { explosiveBlock, impulseCount: rt.impulseCount };
}

module.exports = { evaluateImpulseBypass };
