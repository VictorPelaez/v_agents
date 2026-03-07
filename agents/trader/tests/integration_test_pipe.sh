#!/bin/bash
# Integration test: simulate open+close via FIFO and verify CSV/JSON updated
set -e
BASE=/root/.openclaw/workspace/agents/trader/skills/live-forward-v4.2
FIFO=$BASE/csv_cmd.fifo
ID_TEST=1999999999999
OPEN_PAY='{"cmd":"open","data":{"id":'$ID_TEST',"label":"V4.2","symbol":"BTC","open_time_iso":"2026-03-07T09:58:00.000Z","close_time_iso":"","duration_s":"","side":"LONG","qty":0.01,"entry_price":67000.0,"exit_price":"","sl":66950.0,"tp":67050.0,"profit":"","profit_pct":"","mfe":"","mae":"","reason_tag":"itest","reason_details":{},"tag":"V4.2"}}'
CLOSE_PAY='{"cmd":"close","data":{"id":'$ID_TEST',"label":"V4.2","symbol":"BTC","open_time_iso":"2026-03-07T09:58:00.000Z","close_time_iso":"2026-03-07T09:58:10.000Z","duration_s":10,"side":"LONG","qty":0.01,"entry_price":67000.0,"exit_price":67010.0,"sl":66950.0,"tp":67050.0,"profit":0.0001492537,"profit_pct":0.0001492537,"mfe":"","mae":"","reason_tag":"itest_close","reason_details":{},"tag":"V4.2"}}'

# send open
echo "$OPEN_PAY" > $FIFO
sleep 0.5
# send close
echo "$CLOSE_PAY" > $FIFO
sleep 1
# assert CSV contains id
if grep -q "$ID_TEST" "$BASE/TRADE_LOG_ad.csv"; then
  echo "CSV entry OK"
else
  echo "CSV missing"; exit 2
fi
# assert JSON close contains id
python3 - <<PY
import json,sys
p='$BASE/close_trades.json'
try:
    data=json.loads(open(p).read())
except Exception:
    data=[]
found=any(str(d.get('id'))=='%s' for d in data)
print('JSON close found:', found)
if not found:
    sys.exit(3)
PY
# cleanup test entries
python3 - <<PY
import json,sys,os
p='$BASE/TRADE_LOG_ad.csv'
lines=open(p).read().splitlines()
lines=[l for l in lines if '1999999999999' not in l]
open(p,'w').write('\n'.join(lines)+'\n')
for j in ['open_trades.json','close_trades.json']:
    p2=os.path.join('$BASE',j)
    if os.path.exists(p2):
        try:
            arr=json.loads(open(p2).read())
        except:
            arr=[]
        arr=[d for d in arr if str(d.get('id'))!='1999999999999']
        open(p2,'w').write(json.dumps(arr,indent=2))
print('cleanup done')
PY

echo 'integration test passed'