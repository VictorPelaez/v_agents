#!/usr/bin/env python3
import sys, json, csv, os, time, traceback
from pathlib import Path
# derive paths relative to the repo workspace (script location)
BASE = Path(__file__).resolve().parent.parent
CSV_PATH = str(BASE / 'skills' / 'live-forward-v4.2' / 'TRADE_LOG_ad.csv')
OPEN_TRADES_JSON = str(BASE / 'skills' / 'live-forward-v4.2' / 'open_trades.json')
HEADER=['id','version','symbol','open_time_iso','close_time_iso','exit_price','close_reason','duration_secs','side','qty','entry_price','tp','sl','profit','profit_pct','reason_tag','reason_details','exposure_usd','tag']
LOG_ERRORS = str(BASE / 'skills' / 'live-forward-v4.2' / 'csv_writer_errors.log')

def log_err(msg):
    try:
        with open(LOG_ERRORS,'a') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')
    except Exception:
        pass

def ensure():
    d=os.path.dirname(CSV_PATH)
    os.makedirs(d,exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(HEADER)

def _write_with_retries(fn, *args, retries=3, delay=0.2, **kwargs):
    last=None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last=e
            log_err(f"I/O attempt {i+1}/{retries} failed: {repr(e)} | trace: {traceback.format_exc()}")
            time.sleep(delay)
    # after retries, re-raise
    raise last

def update_open_trades_from_csv():
    # regenerate open_trades.json from CSV (rows with empty close_time_iso)
    try:
        rows=[]
        with open(CSV_PATH,'r',newline='') as f:
            reader=csv.DictReader(f)
            for r in reader:
                if not r.get('close_time_iso','').strip():
                    rows.append({
                        'id': r.get('id',''),
                        'symbol': r.get('symbol',''),
                        'entry_price': r.get('entry_price',''),
                        'sl': r.get('sl',''),
                        'tp': r.get('tp',''),
                        'open_time': r.get('open_time_iso',''),
                        'qty': r.get('qty',''),
                        'profit': r.get('profit',''),
                        'profit_pct': r.get('profit_pct',''),
                        'reason_tag': r.get('reason_tag',''),
                        'reason_details': r.get('reason_details',''),
                        'exposure_usd': r.get('exposure_usd','')
                    })
        tmp=OPEN_TRADES_JSON+'.tmp'
        def write_json():
            with open(tmp,'w') as jf:
                json.dump(rows,jf,indent=2)
        _write_with_retries(write_json)
        def replace_tmp():
            os.replace(tmp,OPEN_TRADES_JSON)
        _write_with_retries(replace_tmp)
    except Exception as e:
        log_err('update_open_trades_from_csv failed: '+repr(e)+"\n"+traceback.format_exc())

def validate_and_prepare_row(data):
    # Ensure expected types and attempt small autocorrections
    out={}
    out['id']=str(data.get('id',''))
    out['version']=data.get('label','')
    out['symbol']=data.get('symbol','')
    out['open_time_iso']=data.get('open_time_iso',data.get('open_time',''))
    out['close_time_iso']=data.get('close_time_iso','')
    out['duration_secs']=data.get('duration_s','')
    out['side']=data.get('side','') or data.get('type','') or ''
    out['qty']=data.get('qty',data.get('size',''))
    out['entry_price']=data.get('entry_price',data.get('entryPrice',''))
    out['tp']=data.get('tp',data.get('takeProfit',''))
    out['sl']=data.get('sl',data.get('stopLoss',''))
    out['profit']=data.get('profit','')
    out['profit_pct']=data.get('profit_pct','')
    out['reason_tag']=data.get('reason_tag',data.get('reasonTag',''))
    out['reason_details']=data.get('reason_details',data.get('reasonDetails',''))
    out['exposure_usd']=data.get('exposureUSD',data.get('exposure_usd',''))
    out['tag']=data.get('tag',data.get('label',''))

    # Autocorrect common problems: if qty looks like 'LONG' and side empty, swap
    try:
        if isinstance(out['qty'], str) and out['qty'].strip().upper()=='LONG' and not out['side']:
            out['side']='LONG'
            out['qty']=''
            log_err(f"autocorrect: qty contained 'LONG' for id={out['id']}, set side=LONG, cleared qty")
    except Exception as e:
        log_err('autocorrect error: '+repr(e))

    # Ensure numeric fields are strings of numbers or empty
    for k in ('entry_price','tp','sl','qty','profit','profit_pct','duration_secs'):
        v=out.get(k,'')
        if v is None: out[k]=''
    return out


def append_open(data):
    ensure()
    prepared = validate_and_prepare_row(data)
    # build row in HEADER order
    row=[prepared.get(k,'') for k in HEADER]
    def write():
        with open(CSV_PATH,'a',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(row)
    _write_with_retries(write)
    # keep open_trades.json synchronized
    update_open_trades_from_csv()

def append_close_and_update(data):
    ensure()
    # Do NOT append a separate close row (avoid duplicate entries).
    tmp=CSV_PATH+'.tmp'
    # update first matching open row: set close_time_iso, duration, profit fields if present
    def update_file_wrapped():
        updated=False
        with open(CSV_PATH,'r',newline='') as rf, open(tmp,'w',newline='') as wf:
            reader=csv.reader(rf); writer=csv.writer(wf)
            header=next(reader,None)
            writer.writerow(header)
            for r in reader:
                if not r: continue
                if r[0]==str(data.get('id')) and not updated:
                    mapping={h:v for h,v in zip(header,r)}
                    mapping['close_time_iso']=data.get('close_time_iso',mapping.get('close_time_iso',''))
                    # map exit_price and close_reason if provided
                    if 'exit_price' in header:
                        mapping['exit_price']=data.get('exit_price',mapping.get('exit_price',''))
                    if 'close_reason' in header:
                        mapping['close_reason']=data.get('close_reason',mapping.get('close_reason',''))
                    # map duration/profit keys if header contains them
                    if 'duration_secs' in header:
                        mapping['duration_secs']=data.get('duration_s',mapping.get('duration_secs',''))
                    if 'profit' in header:
                        mapping['profit']=data.get('profit',mapping.get('profit',''))
                    # write updated row preserving header order
                    newrow=[mapping.get(h,'') for h in header]
                    writer.writerow(newrow)
                    updated=True
                else:
                    writer.writerow(r)
    _write_with_retries(update_file_wrapped)
    # atomic replace
    def replace_tmp():
        os.replace(tmp,CSV_PATH)
    _write_with_retries(replace_tmp)

    # Deduplicate: if there are multiple rows with same id and a close_time set, keep the first and remove subsequent ones
    tmp2=CSV_PATH+'.dedup'
    def dedup_file():
        seen_closed=set()
        with open(CSV_PATH,'r',newline='') as rf, open(tmp2,'w',newline='') as wf:
            reader=csv.DictReader(rf)
            header=reader.fieldnames
            writer=csv.writer(wf)
            writer.writerow(header)
            for r in reader:
                cid=r.get('id','')
                ctime=r.get('close_time_iso','').strip()
                if ctime:
                    if cid in seen_closed:
                        # skip duplicate closed record
                        continue
                    else:
                        seen_closed.add(cid)
                        writer.writerow([r.get(h,'') for h in header])
                else:
                    writer.writerow([r.get(h,'') for h in header])
    _write_with_retries(dedup_file)
    def replace_dedup():
        os.replace(tmp2,CSV_PATH)
    _write_with_retries(replace_dedup)

    # keep open_trades.json synchronized
    update_open_trades_from_csv()

if __name__=='__main__':
    if len(sys.argv)<3:
        print('usage: csv_writer.py <open|close> <json>')
        sys.exit(1)
    cmd=sys.argv[1]
    data=json.loads(sys.argv[2])
    try:
        if cmd=='open':
            append_open(data)
        elif cmd=='close':
            append_close_and_update(data)
        else:
            print('unknown cmd')
            sys.exit(2)
    except Exception as e:
        log_err('csv_writer fatal: '+repr(e)+"\n"+traceback.format_exc())
        raise
