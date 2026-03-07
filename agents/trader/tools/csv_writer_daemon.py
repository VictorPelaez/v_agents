#!/usr/bin/env python3
import os, sys, json, time, traceback
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
FIFO_PATH = str(BASE / 'skills' / 'live-forward-v4.2' / 'csv_cmd.fifo')
LOG = str(BASE / 'skills' / 'live-forward-v4.2' / 'csv_writer_daemon.log')

def log(msg):
    try:
        with open(LOG,'a') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')
    except:
        pass

# Import the csv_writer functions (they must be in the same module path)
try:
    from agents.trader.tools import csv_writer as cw
except Exception:
    # try relative import
    import csv_writer as cw

# ensure FIFO exists
os.makedirs(os.path.dirname(FIFO_PATH), exist_ok=True)
if not os.path.exists(FIFO_PATH):
    try:
        os.mkfifo(FIFO_PATH, 0o600)
    except Exception as e:
        log('mkfifo failed: '+repr(e))
        sys.exit(1)

log('csv_writer_daemon starting, FIFO='+FIFO_PATH)

# Daemon: open FIFO for reading in blocking mode; if reader ends, reopen
while True:
    try:
        with open(FIFO_PATH,'r') as fifo:
            log('FIFO opened for reading')
            for line in fifo:
                line=line.strip()
                if not line: continue
                try:
                    msg=json.loads(line)
                    cmd=msg.get('cmd')
                    data=msg.get('data',{})
                    if cmd=='open':
                        cw.append_open(data)
                        log('processed open id='+str(data.get('id','')))
                        # also persist to JSON open_trades.json (normalized)
                        try:
                            open_json = Path(BASE / 'skills' / 'live-forward-v4.2' / 'open_trades.json')
                            arr = []
                            if open_json.exists():
                                try:
                                    arr = json.loads(open_json.read_text())
                                except:
                                    arr = []
                            rec = {
                                'id': data.get('id'),
                                'label': data.get('label'),
                                'symbol': data.get('symbol'),
                                'open_time_iso': data.get('open_time_iso'),
                                'side': data.get('side'),
                                'qty': data.get('qty'),
                                'entry_price': data.get('entry_price'),
                                'sl': data.get('sl'),
                                'tp': data.get('tp'),
                                'reason_tag': data.get('reason_tag'),
                                'reason_details': data.get('reason_details',{})
                            }
                            arr.append(rec)
                            open_json.write_text(json.dumps(arr,indent=2))
                            log('persisted open json id='+str(data.get('id','')))
                        except Exception as e:
                            log('persist open json failed: '+repr(e))
                    elif cmd=='close':
                        cw.append_close_and_update(data)
                        log('processed close id='+str(data.get('id','')))
                        # also persist to JSON close_trades.json
                        try:
                            close_json = Path(BASE / 'skills' / 'live-forward-v4.2' / 'close_trades.json')
                            arr = []
                            if close_json.exists():
                                try:
                                    arr = json.loads(close_json.read_text())
                                except:
                                    arr = []
                            rec = {
                                'id': data.get('id'),
                                'label': data.get('label'),
                                'symbol': data.get('symbol'),
                                'open_time_iso': data.get('open_time_iso'),
                                'close_time_iso': data.get('close_time_iso'),
                                'duration_s': data.get('duration_s'),
                                'side': data.get('side'),
                                'qty': data.get('qty'),
                                'entry_price': data.get('entry_price'),
                                'exit_price': data.get('exit_price'),
                                'sl': data.get('sl'),
                                'tp': data.get('tp'),
                                'profit': data.get('profit'),
                                'profit_pct': data.get('profit_pct'),
                                'reason_tag': data.get('reason_tag'),
                                'reason_details': data.get('reason_details',{})
                            }
                            arr.append(rec)
                            close_json.write_text(json.dumps(arr,indent=2))
                            log('persisted close json id='+str(data.get('id','')))
                        except Exception as e:
                            log('persist close json failed: '+repr(e))
                    else:
                        log('unknown cmd: '+str(cmd))
                except Exception as e:
                    log('processing line failed: '+repr(e)+" | line:"+line+" trace:"+traceback.format_exc())
    except Exception as e:
        log('FIFO read loop failed: '+repr(e)+" trace:"+traceback.format_exc())
    time.sleep(1)
