'use strict';

/** Move-only refactor: HTTP retry + backoff helpers (axios GET). */

function createHttpRetry(ctx) {
  const { axios, sleep } = ctx || {};
  if (!axios) throw new Error('createHttpRetry: axios required');
  if (!sleep) throw new Error('createHttpRetry: sleep required');

  function jitter(ms) {
    const j = 0.15;
    return Math.round(ms * (1 - j + Math.random() * 2 * j));
  }

  async function httpGetWithRetry(url, opts = {}, retries = 4, baseDelayMs = 250, timeoutMs = 2500) {
    let lastErr = null;

    for (let i = 0; i < retries; i++) {
      try {
        const resp = await axios.get(url, {
          timeout: timeoutMs,
          validateStatus: () => true,
          ...opts
        });

        const status = resp?.status || 0;
        if (status >= 200 && status < 300) return resp;

        if (status === 429 || status === 418) {
          const delay = jitter(baseDelayMs * Math.pow(2, i));
          console.warn('HTTP rate limited', { status, url: url.split('?')[0], delay });
          await sleep(delay);
          continue;
        }

        if (status >= 500 && status < 600) {
          const delay = jitter(baseDelayMs * Math.pow(2, i));
          console.warn('HTTP server error', { status, url: url.split('?')[0], delay });
          await sleep(delay);
          continue;
        }

        lastErr = new Error(`HTTP ${status}`);
        console.error('HTTP non-2xx', { status, url: url.split('?')[0], data: resp?.data });
        break;

      } catch (e) {
        lastErr = e;
        const delay = jitter(baseDelayMs * Math.pow(2, i));
        console.warn('HTTP error', { attempt: i + 1, url: url.split('?')[0], err: e.message, delay });
        if (i < retries - 1) await sleep(delay);
      }
    }

    throw lastErr || new Error('httpGetWithRetry failed');
  }

  return { jitter, httpGetWithRetry };
}

module.exports = {
  createHttpRetry,
};
