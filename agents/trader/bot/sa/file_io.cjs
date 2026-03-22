'use strict';

const fs = require('fs');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    console.error('readJsonFile err:', filePath, e.message);
    return fallback;
  }
}

module.exports = {
  ensureDir,
  readJsonFile,
};
