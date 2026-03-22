'use strict';

// Minimal MEXC Spot v3 REST helper (signed endpoints).
// Docs: https://mexcdevelop.github.io/apidocs/spot_v3_en/

const crypto = require('node:crypto');
const axios = require('axios');

const DEFAULT_BASE = 'https://api.mexc.com';

function toQuery(params) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v === undefined || v === null || v === '') continue;
    p.set(k, String(v));
  }
  // Important: signature uses the exact query string order.
  // We sort keys to ensure deterministic signing.
  const entries = Array.from(p.entries()).sort(([a], [b]) => a.localeCompare(b));
  const s = new URLSearchParams(entries);
  return s.toString();
}

function signQuery(queryString, apiSecret) {
  return crypto.createHmac('sha256', apiSecret).update(queryString).digest('hex');
}

async function mexcPublic({ baseUrl = DEFAULT_BASE, method = 'GET', path, params = {}, timeoutMs = 2500 }) {
  const qs = toQuery(params);
  const url = qs ? `${baseUrl}${path}?${qs}` : `${baseUrl}${path}`;
  const res = await axios.request({ method, url, timeout: timeoutMs });
  return res.data;
}

async function mexcSigned({
  baseUrl = DEFAULT_BASE,
  method = 'GET',
  path,
  params = {},
  apiKey,
  apiSecret,
  recvWindow = 5000,
  timeoutMs = 2500,
}) {
  if (!apiKey || !apiSecret) throw new Error('mexcSigned: apiKey/apiSecret required');

  const signedParams = {
    ...params,
    recvWindow,
    timestamp: Date.now(),
  };

  const qs = toQuery(signedParams);
  const sig = signQuery(qs, apiSecret);
  const body = `${qs}&signature=${sig}`;

  let url = `${baseUrl}${path}`;

  const headers = {
    'X-MEXC-APIKEY': apiKey,
    'Content-Type': 'application/json',
  };

  const req = {
    method,
    url,
    headers,
    timeout: timeoutMs,
  };

  // MEXC behavior is inconsistent across deployments; some endpoints reject form bodies.
  // We send signed params in the query string for all methods (including POST), which works across more gateways.
  const m = String(method || 'GET').toUpperCase();
  url = `${url}?${body}`;
  req.url = url;
  if (!(m === 'GET' || m === 'DELETE')) {
    // keep body empty
    req.data = undefined;
  }

  const res = await axios.request(req);

  return res.data;
}

module.exports = {
  mexcPublic,
  mexcSigned,
  toQuery,
  signQuery,
};
