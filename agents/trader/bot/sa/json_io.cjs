'use strict';

const fs = require('fs');
const path = require('path');

function createJsonIo(ctx) {
  const { ensureDir } = ctx || {};
  if (!ensureDir) throw new Error('createJsonIo: ensureDir required');

  function writeJsonFileAtomic(filePath, value) {
    try {
      ensureDir(path.dirname(filePath));
      const tmpPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
      fs.writeFileSync(tmpPath, JSON.stringify(value, null, 2), 'utf8');
      fs.renameSync(tmpPath, filePath);
      return true;
    } catch (e) {
      console.error('writeJsonFileAtomic err:', filePath, e.message);
      return false;
    }
  }

  function appendJsonl(filePath, record) {
    try {
      ensureDir(path.dirname(filePath));
      fs.appendFileSync(filePath, JSON.stringify(record) + '\n', 'utf8');
      return true;
    } catch (e) {
      console.error('appendJsonl err:', filePath, e.message);
      return false;
    }
  }

  function loadJsonl(filePath) {
    try {
      if (!fs.existsSync(filePath)) return [];
      const raw = fs.readFileSync(filePath, 'utf8');
      if (!raw.trim()) return [];
      return raw
        .split(/\r?\n/)
        .filter(Boolean)
        .map(line => {
          try {
            return JSON.parse(line);
          } catch (e) {
            return null;
          }
        })
        .filter(Boolean);
    } catch (e) {
      console.error('loadJsonl err:', filePath, e.message);
      return [];
    }
  }

  return {
    writeJsonFileAtomic,
    appendJsonl,
    loadJsonl,
  };
}

module.exports = {
  createJsonIo,
};
