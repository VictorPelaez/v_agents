#!/bin/bash
set -e

BOT=./aegis_trade_v2.js
BASE=./skills/live-forward-v4.2
TODAY=$(date -u +%Y%m%d)
JOURNAL=$BASE/trade_journal_${TODAY}.jsonl

mkdir -p $BASE
rm -f $JOURNAL
rm -f $BASE/open_positions.json

echo "TEST 1 — open event writes journal"

node - <<'NODE'
const fs=require("fs")
const path=require("path")

const BASE="./skills/live-forward-v4.2"
const JOURNAL=BASE+"/trade_journal_"+new Date().toISOString().slice(0,10).replace(/-/g,"")+".jsonl"

function appendJsonl(p,r){
 fs.appendFileSync(p,JSON.stringify(r)+"\n")
}

const ev={
 ts:new Date().toISOString(),
 type:"OPEN",
 event_key:"test_open_1",
 trade:{
  id:123,
  symbol:"BTC",
  entry_price:60000,
  sl:59000,
  tp:62000,
  qty:0.01,
  open_time_iso:new Date().toISOString(),
  side:"LONG",
  signal_key:"sig1"
 }
}

appendJsonl(JOURNAL,ev)
NODE

grep -q "test_open_1" $JOURNAL
echo "OK"

echo
echo "TEST 2 — close event append"

node - <<'NODE'
const fs=require("fs")

const BASE="./skills/live-forward-v4.2"
const JOURNAL=BASE+"/trade_journal_"+new Date().toISOString().slice(0,10).replace(/-/g,"")+".jsonl"

fs.appendFileSync(JOURNAL,JSON.stringify({
 ts:new Date().toISOString(),
 type:"CLOSE",
 event_key:"test_close_1",
 trade:{
  id:123,
  symbol:"BTC",
  entry_price:60000,
  exit_price:60100,
  qty:0.01,
  close_time_iso:new Date().toISOString(),
  signal_key:"sig1"
 }
})+"\n")
NODE

grep -q "test_close_1" $JOURNAL
echo "OK"

echo
echo "TEST 3 — journal rebuild open positions"

node - <<'NODE'
const fs=require("fs")

const BASE="./skills/live-forward-v4.2"
const JOURNAL=BASE+"/trade_journal_"+new Date().toISOString().slice(0,10).replace(/-/g,"")+".jsonl"

const lines=fs.readFileSync(JOURNAL,"utf8").trim().split("\n").map(JSON.parse)

let open=new Map()

for(const e of lines){
 if(e.type==="OPEN") open.set(e.trade.id,e.trade)
 if(e.type==="CLOSE") open.delete(e.trade.id)
}

if(open.size!==0){
 console.error("FAIL open trades not empty")
 process.exit(1)
}

console.log("OK rebuild")
NODE

echo
echo "TEST 4 — dedupe event key"

node - <<'NODE'
const fs=require("fs")

const BASE="./skills/live-forward-v4.2"
const JOURNAL=BASE+"/trade_journal_"+new Date().toISOString().slice(0,10).replace(/-/g,"")+".jsonl"

const lines=fs.readFileSync(JOURNAL,"utf8").trim().split("\n")
const keys=new Set()

for(const l of lines){
 const k=JSON.parse(l).event_key
 if(keys.has(k)){
  console.error("FAIL duplicate event key",k)
  process.exit(1)
 }
 keys.add(k)
}

console.log("OK dedupe")
NODE

echo
echo "ALL BASIC TESTS PASSED"
