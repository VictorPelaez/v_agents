#!/usr/bin/env node
const fs=require('fs'); const path=require('path'); const axios=require('axios'); const child_process = require('child_process');
const astBase = process.env.BINANCE_API_BASE || 'https://api.binance.com';
let BINANCE_API_KEY = process.env.BINANCE_API_KEY || '';
try{ if(!BINANCE_API_KEY){ const keyPath = path.join(__dirname,'API_KEYS.md'); if(fs.existsSync(keyPath)){ BINANCE_API_KEY = fs.readFileSync(keyPath,'utf8').split(/\r?\n/)[0].trim(); } } }catch(e){ BINANCE_API_KEY=''; }
const LABEL = process.env.LABEL || 'V4.2';
let lastTradeCandle = null;
const CSV_PATH = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'TRADE_LOG_ad.csv');
let pythonScriptPath = process.env.PYTHON_SCRIPT_PATH || path.join(__dirname,'tools','csv_writer.py');


function writeTradeCsv(entry){
  // Unified CSV writer: send command to csv_writer_daemon via FIFO (non-blocking append)
  try{
    const payload = {
      id: entry.id,
      label: LABEL,
      symbol: 'BTC',
      open_time_iso: entry.openedAt || '',
      close_time_iso: entry.closedAt || entry.timestamp || '',
      duration_s: entry.duration_s || Math.round((new Date(entry.closedAt||entry.timestamp||new Date()).getTime() - new Date(entry.openedAt||new Date()).getTime())/1000),
      side: entry.type || 'LONG',
      qty: entry.size || entry.qty || '',
      entry_price: entry.entryPrice || entry.entry_price || '',
      exit_price: entry.exitPrice || entry.exit_price || '',
      sl: entry.stopLoss || entry.sl || '',
      tp: entry.takeProfit || entry.tp || '',
      profit: entry.profit || '',
      profit_pct: entry.profit_pct || '',
      reason_details: entry.reasonDetails || ''
    };
    const action = entry.action || 'close';
    // write a single-line JSON command to FIFO
    const fifoPath = process.env.CSV_FIFO_PATH || path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'csv_cmd.fifo');
    try{
      const stream = fs.createWriteStream(fifoPath,{flags:'a'});
      const cmdObj = { cmd: action, data: payload };
      stream.write(JSON.stringify(cmdObj) + '\n');
      stream.end();
    }catch(e){
      // fallback to spawn if FIFO fails
      try{
        const p = child_process.spawn('python3',[pythonScriptPath,action,JSON.stringify(payload)], {stdio:['ignore','pipe','pipe']});
        p.stdout.on('data',(d)=>{ console.log('csv_writer stdout:', d.toString().trim()); });
        p.stderr.on('data',(d)=>{ console.error('csv_writer stderr:', d.toString().trim()); });
        p.on('exit',(code,signal)=>{ if(code!==0) console.error('csv_writer exited non-zero',code,signal); });
      }catch(err){ console.error('writeTradeCsv fallback spawn err', err.message) }
    }
  }catch(e){ console.error('writeTradeCsv err', e.message) }
}

// --- Persistencia directa en JSON para reducir lectura de logs y coste de contexto ---
// const OPEN_TRADES_PATH = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'open_trades.json');
// const CLOSE_TRADES_PATH = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'close_trades.json');

// function safeReadJson(p){ try{ if(fs.existsSync(p)){ return JSON.parse(fs.readFileSync(p,'utf8')) || []; } }catch(e){} return []; }
// function safeWriteJson(p,obj){ try{ fs.writeFileSync(p, JSON.stringify(obj, null, 2)); return true;}catch(e){ console.error('safeWriteJson err', e.message); return false; } }

function persistOpenTrade(t){
  try{
    // rotate by month: open_trades_YYYYMM.json
    const dt = new Date(t.openedAt || Date.now());
    const ymd = dt.getUTCFullYear().toString()
          + String(dt.getUTCMonth()+1).padStart(2,'0')
          + String(dt.getUTCDate()).padStart(2,'0'); // añade día
        
    const dir = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
   
    const file = path.join(dir, `open_trades_${ymd}.json`);
    // Crear archivo con [] si no existe
    if (!fs.existsSync(file)) { fs.writeFileSync(file, JSON.stringify([], null, 2)); }
        
    // const file = path.join(dir, `open_trades_${ym}.json`);
    let arr = [];
    try{ if(fs.existsSync(file)){ arr = JSON.parse(fs.readFileSync(file,'utf8')) || []; } }catch(e){ arr = []; }
    const rec = {
      id: t.id,
      label: LABEL,
      symbol: 'BTC',
      open_time_iso: t.openedAt || new Date().toISOString(),
      side: t.type || 'LONG',
      qty: t.size || t.qty || 0,
      entry_price: t.entryPrice || t.entry_price || null,
      sl: t.stopLoss || t.sl || null,
      tp: t.takeProfit || t.tp || null,
      reason_tag: t.reasonTag || '',
      reason_details: t.reasonDetails || {},
      exposure_usd: t.exposureUSD || null
    };
    arr.push(rec);
    fs.writeFileSync(file, JSON.stringify(arr,null,2));
    return true;
  }catch(e){ console.error('persistOpenTrade err', e && e.message? e.message : e); return false; }
}

function persistCloseTrade(t){
  try{
    const dt = new Date(t.closedAt || Date.now());
    // const ym = dt.getUTCFullYear().toString() + String(dt.getUTCMonth()+1).padStart(2,'0');
    const ymd = dt.getUTCFullYear().toString()
          + String(dt.getUTCMonth()+1).padStart(2,'0')
          + String(dt.getUTCDate()).padStart(2,'0');

    const dir = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        
    const file = path.join(dir, `close_trades_${ymd}.json`);
    // Crear archivo con [] si no existe
    if (!fs.existsSync(file)) { fs.writeFileSync(file, JSON.stringify([], null, 2)); }    

    let arr = [];
    try{ if(fs.existsSync(file)){ arr = JSON.parse(fs.readFileSync(file,'utf8')) || []; } }catch(e){ arr = []; }
    const rec = {
      id: t.id,
      label: LABEL,
      symbol: 'BTC',
      open_time_iso: t.openedAt || '',
      close_time_iso: t.closedAt || new Date().toISOString(),
      duration_s: t.duration_s || (t.closedAt && t.openedAt ? Math.round((new Date(t.closedAt).getTime()-new Date(t.openedAt).getTime())/1000) : null),
      side: t.type || 'LONG',
      qty: t.size || t.qty || 0,
      entry_price: t.entryPrice || t.entry_price || null,
      exit_price: t.exitPrice || t.exit_price || null,
      sl: t.stopLoss || t.sl || null,
      tp: t.takeProfit || t.tp || null,
      profit: t.profit || null,
      profit_pct: t.profit_pct || null,
      reason_tag: t.reasonTag || '',
      reason_details: t.reasonDetails || {},
      exposure_usd: t.exposureUSD || null
    };
    arr.push(rec);
    fs.writeFileSync(file, JSON.stringify(arr,null,2));
    return true;
  }catch(e){ console.error('persistCloseTrade err', e && e.message? e.message : e); return false; }
}

async function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

(async ()=>{
  console.log('starting live service (paper decisions)');
  // load per-skill config (already read above for MAX_POSITIONS); reuse cfg if available
  let skillCfg={};
  try{
    const cfgPath = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'config.json');
    if(fs.existsSync(cfgPath)) skillCfg = JSON.parse(fs.readFileSync(cfgPath,'utf8'));
  }catch(e){ console.error('skill cfg read err',e.message) }
  const symbol = process.env.SYMBOL || skillCfg.SYMBOL || 'BTCUSDT';
  const skillCapital = parseFloat(process.env.CAPITAL || skillCfg.CAPITAL || 1000);
  // Use a single canonical risk variable: skillCfg.RISK_PCT (or env RISK_PCT); default 0.01
  const riskPct = parseFloat(process.env.RISK_PCT || skillCfg.RISK_PCT || 0.01);

  const monitorInterval = parseInt(process.env.MONITOR_INTERVAL_MS || skillCfg.MONITOR_INTERVAL_MS || 1000,10);
  const defaultSide = process.env.DEFAULT_SIDE || skillCfg.DEFAULT_SIDE || 'LONG';
  // CSV_PATH and python script path configurable
  const configuredCsvPath = (process.env.CSV_PATH || skillCfg.CSV_PATH) || '';
  // CSV_PATH default (if not configured) falls back to skills folder
  const CSV_PATH = configuredCsvPath || path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'TRADE_LOG_ad.csv');

  const openTrades=[]; // {id,entryPrice,stopLoss,takeProfit,openedAt,size,reason}
  const lastOpenBySymbol = {}; // symbol -> timestamp (ms) to enforce cooldown between opens
  let lastDecision = {}; // stores last decision snapshot (momentum, priceAboveSMA, shouldEnter)
  // Backfill: load persisted open trades at startup so monitor can pick up previously-opened positions
  try{
    const opensPath = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'open_trades.json');
    if(fs.existsSync(opensPath)){
      const data = JSON.parse(fs.readFileSync(opensPath,'utf8'));
      if(Array.isArray(data)){
        for(const o of data){
          try{
            const t = {
              id: o.id || (Date.now()),
              entryPrice: parseFloat(o.entry_price) || 0,
              stopLoss: o.sl ? parseFloat(o.sl) : null,
              takeProfit: o.tp ? parseFloat(o.tp) : null,
              openedAt: o.open_time || new Date().toISOString(),
              size: o.qty || 0,
              type: 'LONG',
              reasonTag: 'backfill',
              reasonDetails: {}
            };
            openTrades.push(t);
            console.log('backfilled open trade', t.id, t.entryPrice, 'SL', t.stopLoss, 'TP', t.takeProfit);
          }catch(e){ console.error('backfill parse err',e.message) }
        }
      }
    }
  }catch(e){ console.error('backfill err', e.message) }
  // load skill config (per-skill config.json) if present
  let maxPositionsDefault = 2;
  try{
    const cfgPath = path.join(__dirname,'skills',`live-forward-${LABEL.toLowerCase()}`,'config.json');
    if(fs.existsSync(cfgPath)){
      const cfg = JSON.parse(fs.readFileSync(cfgPath,'utf8'));
      if(cfg && cfg.MAX_POSITIONS) maxPositionsDefault = parseInt(cfg.MAX_POSITIONS,10);
    }
  }catch(e){ console.error('config load err', e.message) }
  const maxPositions = parseInt(process.env.MAX_POSITIONS || maxPositionsDefault,10);
  // Use skillCapital and riskPct computed above (from env or skill config)
  const minHoldS = parseInt(process.env.MIN_HOLD_SECONDS || skillCfg.MIN_HOLD_SECONDS || '60',10);
  const timeStopMinutes = parseInt(process.env.TIME_STOP_MINUTES || skillCfg.TIME_STOP_MINUTES || 10,10);


  // helper: http GET with retries
  async function httpGetWithRetry(url, opts={}, retries=3, delayMs=300){
    for(let i=0;i<retries;i++){
      try{
        const r = await axios.get(url, opts);
        return r;
      }catch(e){
        console.error('httpGetWithRetry attempt',i+1,'failed',e.message);
        if(i<retries-1) await sleep(delayMs);
      }
    }
    throw new Error('httpGetWithRetry failed after '+retries+' attempts');
  }
  async function getTicker(){
    const url=`${astBase}/api/v3/ticker/price?symbol=${symbol}`;
    try{
      const r=await httpGetWithRetry(url,{headers: (BINANCE_API_KEY? {'X-MBX-APIKEY': BINANCE_API_KEY} : {})});
      if(r.data && (r.data.price || r.data.price===0)) return +r.data.price;
    }catch(e){ console.error('getTicker failed', e.message); }
    return null;
  }

  // helper: fetch recent klines (candles) from Binance
  async function getRecentKlines(limit, interval='1m'){
    try{
      const url = `${astBase}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
      const r = await httpGetWithRetry(url,{headers: (BINANCE_API_KEY? {'X-MBX-APIKEY': BINANCE_API_KEY} : {})});
      if(r && r.data) return r.data;
    }catch(e){ console.error('getRecentKlines failed', e.message); }
    return null;
  }

  while(true){
      const SMA_WINDOW = parseInt(process.env.SMA_WINDOW || skillCfg.SMA_WINDOW || 60,10);
      const limit = SMA_WINDOW + 2; // enough candles for SMA window + previous closed candle + one extra
      const klines = await getRecentKlines(limit);
      // compute candle, SMA, momentum and ATR from the same klines array (single API call)
      let momentum_pct = 0; let sma = null; let smaPrev = null; let atr_pct = 0;
      let candle = null;
      if(klines && Array.isArray(klines) && klines.length>=2){
        // last closed candle is the penultimate entry
        const lastClosed = klines[klines.length-2];
        candle = { ts: lastClosed[0], open:+lastClosed[1], high:+lastClosed[2], low:+lastClosed[3], close:+lastClosed[4] };
        const closes = klines.map(c=>+c[4]);
        const last = closes[closes.length-2]; // last closed
        const prev = closes[closes.length-3] || last;
        momentum_pct = prev ? (last - prev)/prev : 0;
        // SMA over window
        if(closes.length >= SMA_WINDOW){
          const lastWindow = closes.slice(-SMA_WINDOW-1, -1); // take the last SMA_WINDOW closed closes
          const sumWindow = lastWindow.reduce((a,b)=>a+b,0);
          sma = sumWindow / lastWindow.length;
          const prevWindow = closes.slice(-SMA_WINDOW-2, -2);
          const sumPrev = prevWindow.reduce((a,b)=>a+b,0);
          smaPrev = prevWindow.length? (sumPrev / prevWindow.length) : null;
        } else {
          const sum = closes.reduce((a,b)=>a+b,0); sma = sum/closes.length; smaPrev = null;
        }
        // ATR using trs over klines
        const trs = [];
        for(let i=1;i<klines.length;i++){
          const high = +klines[i][2];
          const low = +klines[i][3];
          const prevClose = +klines[i-1][4];
          const tr = Math.max(high-low, Math.abs(high-prevClose), Math.abs(low-prevClose));
          trs.push(tr);
        }
        const sumTr = trs.reduce((a,b)=>a+b,0);
        const atr = trs.length? (sumTr/trs.length) : 0;
        atr_pct = atr / (last || 1);
      }
      // base momentum and configurable reduction on green runs
      const BASE_MIN_MOM = parseFloat(process.env.MIN_MOMENTUM_PCT || skillCfg.MIN_MOMENTUM_PCT || 0.005);
      const MOM_REDUCTION_PCT = parseFloat(process.env.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || skillCfg.MOMENTUM_REDUCTION_PCT_ON_GREEN_RUN || 0.0);
      const GREEN_KLINES = parseInt(process.env.GREEN_KLINES_FOR_REDUCTION || skillCfg.GREEN_KLINES_FOR_REDUCTION || 0,10);
      const smaSlope = (sma !== null && smaPrev !== null) ? (sma - smaPrev) : 0;
      const smaTol = parseFloat(process.env.SMA_TOLERANCE || skillCfg.SMA_TOLERANCE || 0.001);
      const priceNearSMA = (sma!==null && candle) ? (candle.close >= sma * (1 - smaTol)) : true; // tolerance from config
      const trendUp = smaSlope > 0;
      // determine how many consecutive green candles up to GREEN_KLINES
      let green_run = 0;
      if(GREEN_KLINES>0 && Array.isArray(klines)){
        for(let j=klines.length-2;j>0 && green_run < GREEN_KLINES;j--){
          const cur = +klines[j][4]; const op = +klines[j][1];
          if(cur>op) green_run++; else break;
        }
      }
      let effectiveMinMom = BASE_MIN_MOM;
      if(green_run>=GREEN_KLINES && GREEN_KLINES>0 && MOM_REDUCTION_PCT>0){
        effectiveMinMom = BASE_MIN_MOM * (1 - MOM_REDUCTION_PCT);
      }
      const momentumOk = momentum_pct >= effectiveMinMom;
      const k_tp = parseFloat(process.env.K_TP || (skillCfg && skillCfg.K_TP) || '2.0');
      const k_sl = parseFloat(process.env.K_SL || (skillCfg && skillCfg.K_SL) || '1.2');
      // log effective TP/SL factors and their source
      const _k_tp_src = process.env.K_TP ? 'env' : (skillCfg && skillCfg.K_TP ? 'config' : 'default');
      const _k_sl_src = process.env.K_SL ? 'env' : (skillCfg && skillCfg.K_SL ? 'config' : 'default');
      // reduced logging: only report params in ITER_SUMMARY to avoid spamming the log
      // console.log('PARAMS: k_tp=',k_tp,'(source=',_k_tp_src+') k_sl=',k_sl,'(source=',_k_sl_src+')');
      // final logic: require momentum AND (trend up OR price near/above SMA)
      const shouldEnter = momentumOk && (trendUp || priceNearSMA);
      lastDecision = {momentum_pct: Number((momentum_pct).toFixed(6)), sma: sma?Number(sma.toFixed(2)):null, smaSlope: Number(smaSlope.toFixed(6)), priceNearSMA: !!priceNearSMA, trendUp: !!trendUp, shouldEnter: !!shouldEnter, effective_min_momentum: Number(effectiveMinMom.toFixed(6)), green_run: green_run};
      // EVALUATION_DETAIL: dump decision inputs and protections for debugging (throttled)
      const entryCandidate = candle ? candle.close : null;
      // throttle logging: print eval detail only if decision changes or every LOG_INTERVAL_MS
      if(typeof globalThis.__lastEvalLogTs === 'undefined') globalThis.__lastEvalLogTs = 0;
      if(typeof globalThis.__lastDecisionSnap === 'undefined') globalThis.__lastDecisionSnap = '';
      try{
        const snapshot = JSON.stringify({momentum_pct:lastDecision.momentum_pct, sma:lastDecision.sma, smaSlope:lastDecision.smaSlope, shouldEnter:lastDecision.shouldEnter, openTrades_count: openTrades.length});
        const LOG_INTERVAL_MS = parseInt(process.env.EVAL_LOG_INTERVAL_MS || skillCfg.EVAL_LOG_INTERVAL_MS || 60000,10);
        const now = Date.now();
        const shouldLog = (snapshot !== globalThis.__lastDecisionSnap) || (now - globalThis.__lastEvalLogTs > LOG_INTERVAL_MS);
        if(shouldLog){
          globalThis.__lastDecisionSnap = snapshot;
          globalThis.__lastEvalLogTs = now;
          const _entryCandidate = (typeof entryCandidate !== 'undefined') ? entryCandidate : null;
          const evalDetail = {
            ts: new Date().toISOString(),
            entryCandidate: _entryCandidate,
            candle_price: _entryCandidate,
            shouldEnter: !!lastDecision.shouldEnter,
            momentum_pct: lastDecision.momentum_pct,
            sma: lastDecision.sma,
            smaSlope: lastDecision.smaSlope,
            priceNearSMA: lastDecision.priceNearSMA,
            trendUp: lastDecision.trendUp,
            effective_min_momentum: lastDecision.effective_min_momentum,
            green_run: lastDecision.green_run,
            openTrades_count: openTrades.length,
            maxPositions: maxPositions,
            dup_tolerance_pct: parseFloat(process.env.DUP_TOLERANCE_PCT || skillCfg.DUP_TOLERANCE_PCT || 1e-6),
            symbol_cooldown_s: parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || skillCfg.SYMBOL_COOLDOWN_SECONDS || 0,10)
          };
          console.log('EVALUATION_DETAIL_JSON:', JSON.stringify(evalDetail));
        }
      }catch(e){ 
        console.error('eval detail failed', e.stack || e.message);
        try{ console.error('EVAL_TRACE_PRE', {entryCandidate: (typeof entryCandidate!=='undefined'? entryCandidate : '<undef>'), openTradesLen: openTrades.length, lastDecision: lastDecision}); }catch(xx){ console.error('EVAL_TRACE_PRE failed', xx && xx.stack? xx.stack : xx);
        }
       }

      // prevent near-duplicate opens: if an open with almost the same entryPrice already exists, skip
      const DUP_TOLERANCE_PCT = parseFloat(process.env.DUP_TOLERANCE_PCT || skillCfg.DUP_TOLERANCE_PCT || 1e-6);
      const CANDLE_MS = parseInt(process.env.CANDLE_MS || skillCfg.CANDLE_MS || 60000,10);
      const alreadySimilar = (entryCandidate !== null) && openTrades.some(ot => {
        if(!ot.entryPrice) return false;
        const similarPrice = Math.abs(ot.entryPrice - entryCandidate) <= Math.abs(entryCandidate) * DUP_TOLERANCE_PCT;
        if(!similarPrice) return false;
        // only treat as duplicate if opened within the same candle window (prevent duplicate in same candle)
        try{
          const openedTs = new Date(ot.openedAt).getTime();
          return (Date.now() - openedTs) <= CANDLE_MS;
        }catch(e){
          return false;
        }
      });
      const cooldown_s = parseInt(process.env.SYMBOL_COOLDOWN_SECONDS || skillCfg.SYMBOL_COOLDOWN_SECONDS || 0,10);
      const lastOpenTs = lastOpenBySymbol[symbol] || 0;
      const nowTs = Date.now();
      const withinCooldown = (cooldown_s>0) && ((nowTs - lastOpenTs) < cooldown_s*1000);
      
      // Just one trade with this candle      
      const candleMinute = Math.floor(candle.ts / 60000);
      const alreadyOpenedThisCandle = lastTradeCandle === candleMinute;      
      // if(lastTradeCandle === candleTs){console.log("skip: trade already opened in this candle"); return;}

      if(shouldEnter && openTrades.length<maxPositions && !alreadySimilar && !withinCooldown && !alreadyOpenedThisCandle){
          const entryPrice=candle.close; const ktp=k_tp; const ksl=k_sl;
          const effectiveATR = atr_pct; // usar ATR calculado directamente (fórmula simple)
          let stopLoss = entryPrice * (1 - ksl * effectiveATR);
          const takeProfit = entryPrice * (1 + ktp * effectiveATR);
          const MIN_SL_USD = parseFloat(process.env.MIN_SL_USD || skillCfg.MIN_SL_USD || 50);
          // enforce a minimum absolute stop distance in USD
          const stopDistance = entryPrice - stopLoss;
          if(stopDistance < MIN_SL_USD){
            stopLoss = entryPrice - MIN_SL_USD;
          }
          const reasonDet = {
            momentum_pct: Number((momentum_pct).toFixed(6)),
            sma: sma?Number(sma.toFixed(2)):null,
            atr_pct: Number(atr_pct.toFixed(6)),
            base_min_momentum: BASE_MIN_MOM,
            effective_minimum: Number(effectiveMinMom.toFixed(6)),
            green_run_len: green_run,
            reduction_pct: MOM_REDUCTION_PCT
          };
          // mark a concise reason tag
          const reasonTag = (green_run>=GREEN_KLINES && MOM_REDUCTION_PCT>0) ? 'momentum_with_green_run' : 'momentum_standard';
          // proper risk-controlled sizing: risk in USD divided by stop distance (USD)
          const riskUSD = skillCapital * riskPct;
          const stopDistanceUSD = entryPrice - stopLoss;
          if(stopDistanceUSD <= 0){
            console.error('invalid stop distance, skipping trade', {entryPrice, stopLoss});
          }
          const qty = Number((riskUSD / stopDistanceUSD).toFixed(8));

          console.log('CHECK SL:', 'entry=',entryPrice, 'SL=',stopLoss, 'candleLow=', candle ? candle.low : 'null');

          // intrabar validation: if the candle low already touches the stopLoss, skip opening (would have died inside the candle)
          let trade = null;
          if(candle && typeof candle.low !== 'undefined' && candle.low <= stopLoss){
            console.log('trade skipped: SL inside candle (would have triggered before entry)', {entryPrice, stopLoss, candleLow:candle.low});
          } else {lastTradeCandle = candleMinute; const exposureUSD = Number((entryPrice * qty).toFixed(2));
            trade = {id:Date.now(),entryPrice,stopLoss,takeProfit,openedAt:new Date().toISOString(),size:qty,exposureUSD:exposureUSD,type:'LONG',reasonTag:reasonTag,reasonDetails:reasonDet};    
          
            openTrades.push(trade);
            console.log('OPEN (paper):', (trade.openedAt? trade.openedAt : new Date(trade.id).toISOString()), trade.entryPrice,'SL',trade.stopLoss,'TP',trade.takeProfit,'REASON',reasonTag,reasonDet);
            try{
              const payloadOpen = JSON.stringify({id:trade.id,label:LABEL,symbol:'BTC',open_time_iso:trade.openedAt,close_time_iso:'',duration_s:'',side:trade.type,qty:trade.size,entry_price:trade.entryPrice,exit_price:'',sl:trade.stopLoss,tp:trade.takeProfit,profit:'',profit_pct:'',mfe:'',mae:'',reason_tag:trade.reasonTag,reason_details:trade.reasonDetails,tag:LABEL});
              // unified CSV writer call for open (avoid duplicate python invocations)
              try{
                const csvPayload = { id: trade.id, label: LABEL, symbol: 'BTC', open_time_iso: trade.openedAt || new Date().toISOString(), close_time_iso: '', duration_s: '', side: trade.type, qty: trade.size, entry_price: trade.entryPrice, exit_price: '', sl: trade.stopLoss, tp: trade.takeProfit, profit: '', profit_pct: '', reason_details: trade.reasonDetails };
                writeTradeCsv({ ...csvPayload, action: 'open' });
              }catch(e){ console.error('writeTradeCsv open call failed', e.message); }
            }catch(e){console.error('csv open write failed',e.message)}
            try{ persistOpenTrade(trade);}catch(e){ console.error('persistOpenTrade failed', e && e.message ? e.message : e); }
          }
        } 
    // } ERROR

    // monitor open trades by price every 1s for up to minHoldS
    const monitorStart=Date.now();
    console.log('ENTER MONITOR LOOP:', new Date().toISOString(), ' openTrades=', openTrades.length);
    while(Date.now()-monitorStart < 60*1000){ // check up to 1 minute before next candle fetch
      try{
        const market=await getTicker();
        if(market!==null){
          for(let i=openTrades.length-1;i>=0;i--){
            const t=openTrades[i];
            const age=(Date.now()-new Date(t.openedAt).getTime())/1000;
            if(age<minHoldS) continue; // enforce minimal hold
            if(age > timeStopMinutes*60){
              // time-stop forced close
              console.log('debug-time-stop forced close'); 
              const profit = (market - t.entryPrice) * (t.size || 0);
              t.exitPrice=market; t.profit=profit; t.closedAt=new Date().toISOString();
              console.log('CLOSE TIME_STOP:', (t.closedAt? t.closedAt : (t.id? new Date(t.id).toISOString() : '')), t.exitPrice,t.profit,'age_s',Math.round(age));
              try{
              const payloadCloseTime = JSON.stringify({id:t.id,label:LABEL,symbol:'BTC',open_time_iso:t.openedAt,close_time_iso:t.closedAt,duration_s:Math.round((new Date(t.closedAt).getTime()-new Date(t.openedAt).getTime())/1000),side:t.type,qty:t.size,entry_price:t.entryPrice,exit_price:t.exitPrice,sl:t.stopLoss,tp:t.takeProfit,profit:t.profit,profit_pct:'',mfe:'',mae:'',reason_tag:'time_stop',close_reason:'time_stop',reason_details:t.reasonDetails,tag:LABEL});
              // write unified CSV close record (time-stop)
              try{
                writeTradeCsv({ id: t.id, label: LABEL, symbol: 'BTC', open_time_iso: t.openedAt, close_time_iso: t.closedAt || new Date().toISOString(), duration_s: Math.round((new Date(t.closedAt).getTime()-new Date(t.openedAt).getTime())/1000), side: t.type, qty: t.size, entry_price: t.entryPrice, exit_price: t.exitPrice, sl: t.stopLoss, tp: t.takeProfit, profit: t.profit, reason_details: t.reasonDetails, action: 'close' });
              }catch(e){ console.error('writeTradeCsv close time_stop failed', e.message); }
            }catch(e){console.error('csv time_stop write failed',e.message)}
            try{ persistCloseTrade(t); }catch(e){ console.error('persistCloseTrade failed', e && e.message? e.message : e); }
              openTrades.splice(i,1);
            } else if(t.stopLoss && market<=t.stopLoss){
              // close
              console.log('debug-sl close');
              const profit = (market - t.entryPrice) * (t.size || 0);
              t.exitPrice=market; t.profit=profit; t.closedAt=new Date().toISOString();
              console.log('CLOSE SL (paper):', (t.closedAt? t.closedAt : (t.id? new Date(t.id).toISOString() : '')), t.exitPrice,t.profit);
              try{
              const payloadClose = JSON.stringify({id:t.id,label:LABEL,symbol:'BTC',open_time_iso:t.openedAt,close_time_iso:t.closedAt,duration_s:Math.round((new Date(t.closedAt).getTime()-new Date(t.openedAt).getTime())/1000),side:t.type,qty:t.size,entry_price:t.entryPrice,exit_price:t.exitPrice,sl:t.stopLoss,tp:t.takeProfit,profit:t.profit,profit_pct:'',mfe:'',mae:'',reason_tag:t.reasonTag,close_reason:(t.exitPrice<=t.stopLoss? 'SL' : (t.exitPrice>=t.takeProfit? 'TP' : 'OTHER')),reason_details:t.reasonDetails,tag:LABEL});
              // write unified CSV close record (SL/OTHER)
              try{
                writeTradeCsv({ id: t.id, label: LABEL, symbol: 'BTC', open_time_iso: t.openedAt, close_time_iso: t.closedAt || new Date().toISOString(), duration_s: Math.round((new Date(t.closedAt).getTime()-new Date(t.openedAt).getTime())/1000), side: t.type, qty: t.size, entry_price: t.entryPrice, exit_price: t.exitPrice, sl: t.stopLoss, tp: t.takeProfit, profit: t.profit, reason_details: t.reasonDetails, action: 'close' });
              }catch(e){ console.error('writeTradeCsv close failed', e.message); }
            }catch(e){console.error('csv close write failed',e.message)}
            try{ persistCloseTrade(t); }catch(e){ console.error('persistCloseTrade failed', e && e.message? e.message : e); }
              openTrades.splice(i,1);
            } else if(t.takeProfit && market>=t.takeProfit){
              const profit = (market - t.entryPrice) * (t.size || 0);
              t.exitPrice=market; t.profit=profit; t.closedAt=new Date().toISOString();
              console.log('CLOSE TP:', (t.closedAt? t.closedAt : (t.id? new Date(t.id).toISOString() : '')), t.exitPrice,t.profit);              
              // console.log('CLOSE TP (paper):',t.id,t.exitPrice,t.profit);
              writeTradeCsv({...t, action:'close', timestamp:new Date().toISOString()});
              openTrades.splice(i,1);
            }
          }
        }
      }catch(e){console.error('monitor err',e.message)}
      await sleep(500);
    }
    }
    // Restore historic multi-line tracing: ok end iter + detailed lines (momentum, sma, params, openTrades). Print human-readable date (UTC+1)
    function fmtDateUtc1(d){
      const dt = new Date(d.getTime()+60*60*1000);
      const Y = dt.getUTCFullYear();
      const M = String(dt.getUTCMonth()+1).padStart(2,'0');
      const D = String(dt.getUTCDate()).padStart(2,'0');
      const h = String(dt.getUTCHours()).padStart(2,'0');
      const m = String(dt.getUTCMinutes()).padStart(2,'0');
      const s = String(dt.getUTCSeconds()).padStart(2,'0');
      return `${Y}-${M}-${D} ${h}:${m}:${s}`;
    }
    const tsHuman = fmtDateUtc1(new Date());
    // Expanded, human-friendly ITER summary (forced verbose)
    const ld = lastDecision || {};
    try{
      console.log('ITER_SUMMARY:', tsHuman, 'momentum='+(ld.momentum_pct||0),'sma='+(ld.sma||'null'),'smaSlope='+(ld.smaSlope||0),'priceNearSMA='+(ld.priceNearSMA?1:0),'trendUp='+(ld.trendUp?1:0),'shouldEnter='+(ld.shouldEnter?1:0),'effective_min_momentum='+(ld.effective_min_momentum||0),'green_run='+(ld.green_run||0),'openTrades='+openTrades.length);
      console.log('TRACE_DETAILS_JSON:', JSON.stringify({params:{k_tp:k_tp,k_sl:k_sl,k_tp_src:_k_tp_src,k_sl_src:_k_sl_src}, decision: ld, openTradesCount: openTrades.length}));
      // print full openTrades (multi-line) so Watch returns the same style as before
      if(openTrades && openTrades.length>0) {
        for(const ot of openTrades) console.log('OPEN_TRADE:', JSON.stringify(ot));
      }
    }catch(e){ console.error('verbose log failed', e.message) }
})();
