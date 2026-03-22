'use strict';

const fs = require('fs');

function isPidAlive(pid) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e && e.code === 'EPERM';
  }
}

function createFileLock(ctx) {
  const { ensureDir, baseDir, lockPath, nowIso } = ctx || {};
  if (!ensureDir) throw new Error('createFileLock: ensureDir required');
  if (!baseDir) throw new Error('createFileLock: baseDir required');
  if (!lockPath) throw new Error('createFileLock: lockPath required');
  if (!nowIso) throw new Error('createFileLock: nowIso required');

  let lockFd = null;

  function acquireLock() {
    ensureDir(baseDir);

    if (fs.existsSync(lockPath)) {
      try {
        const raw = fs.readFileSync(lockPath, 'utf8');
        const info = JSON.parse(raw || '{}');
        const pid = Number(info.pid || 0);
        if (pid && isPidAlive(pid)) {
          console.error('lock exists, process seems alive:', { lock: lockPath, pid });
          return false;
        }
        console.warn('stale lock detected, removing:', { lock: lockPath, pid });
        try { fs.unlinkSync(lockPath); } catch (_) {}
      } catch (e) {
        console.warn('lock exists but unreadable, removing:', { lock: lockPath, err: e.message });
        try { fs.unlinkSync(lockPath); } catch (_) {}
      }
    }

    try {
      lockFd = fs.openSync(lockPath, 'wx');
      fs.writeFileSync(lockFd, JSON.stringify({ pid: process.pid, started_at: nowIso() }), 'utf8');
      return true;
    } catch (e) {
      if (e && e.code === 'EEXIST') {
        console.error('lock exists, another instance may be running:', lockPath);
        return false;
      }
      console.error('acquireLock err:', e.message);
      return false;
    }
  }

  function releaseLock() {
    try {
      if (lockFd !== null) fs.closeSync(lockFd);
    } catch (_) {}
    lockFd = null;
    try {
      if (fs.existsSync(lockPath)) fs.unlinkSync(lockPath);
    } catch (e) {
      console.error('releaseLock err:', e.message);
    }
  }

  return {
    acquireLock,
    releaseLock,
  };
}

module.exports = {
  createFileLock,
};
