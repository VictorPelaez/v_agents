'use strict';

function getYmd(dateValue) {
  const dt = new Date(dateValue || Date.now());
  return String(dt.getUTCFullYear()) +
    String(dt.getUTCMonth() + 1).padStart(2, '0') +
    String(dt.getUTCDate()).padStart(2, '0');
}

function fmtDateUtc1(d) {
  // UTC+1 formatting (used in logs & iteration summaries)
  const dt = new Date(d.getTime() + 60 * 60 * 1000);
  const Y = dt.getUTCFullYear();
  const M = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const D = String(dt.getUTCDate()).padStart(2, '0');
  const h = String(dt.getUTCHours()).padStart(2, '0');
  const m = String(dt.getUTCMinutes()).padStart(2, '0');
  const s = String(dt.getUTCSeconds()).padStart(2, '0');
  return `${Y}-${M}-${D} ${h}:${m}:${s}`;
}

function nowIso() {
  return new Date().toISOString();
}

function makeGetJournalPath(baseDir) {
  return function getJournalPath(dateValue) {
    return require('path').join(baseDir, `trade_journal_${getYmd(dateValue)}.jsonl`);
  };
}

module.exports = {
  getYmd,
  fmtDateUtc1,
  nowIso,
  makeGetJournalPath,
};
