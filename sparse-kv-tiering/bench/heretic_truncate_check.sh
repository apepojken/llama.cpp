#!/bin/bash
# heretic_truncate_check.sh — [TAG_PROMPT_TRUNCATE] the server trims an over-long prompt itself.
# Small context (8192) so the whole thing runs in minutes. A conversation of ~12k tokens is sent to
# a server started WITH --prompt-truncate and to one WITHOUT it.
#   T1 with the flag   : must answer, keep the system prompt (it demands a suffix) and the newest turn
#   T2 with the flag   : one more turn on top -> must reuse the cache instead of re-processing
#   T3 without the flag: the same request must still be rejected (default behaviour unchanged)
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_truncate_check.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
start(){ local k=$1; shift; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 8192 --parallel 1 -b 2048 -ub 512 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 -lv 4 "$@" > "$BD/tc_srv_$k.log" 2>&1 &
  PID=$!; local ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'error[^\n]*' "$BD/tc_srv_$k.log" | head -2 | tr '\n' ' ')"; return 1; }
  log "  server up ($k)"
}
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
build_reqs(){ python3 - "$BD" "$PORT" <<'PY'
import json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=600).read())
filler=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
sysmsg=("You are a terse assistant. Whatever else you are asked, your reply must end with the exact "
        "token ZEBRA-9 on its own line. Never omit it.")
msgs=[{'role':'system','content':sysmsg}]
step=1400
for i in range(30):                       # ~30 turns of ~400 tokens each -> well past 8192
    msgs.append({'role':'user','content':f"Note {i}: the codeword is CODE-{i:02d}.\n"+filler[i*step:(i+1)*step]})
    msgs.append({'role':'assistant','content':f"Noted codeword CODE-{i:02d}."})
msgs.append({'role':'user','content':"What was the codeword in my most recent note? Answer with just the codeword."})
base={'model':'x','max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
json.dump(dict(base,messages=msgs), open(f'{bd}/tc_req_1.json','w'))
msgs2=msgs[:-1]+[{'role':'assistant','content':"CODE-29"},
                 {'role':'user','content':"And the note before that one? Answer with just the codeword."}]
json.dump(dict(base,messages=msgs2), open(f'{bd}/tc_req_2.json','w'))
n=len(post('/apply-template',{'messages':msgs,'chat_template_kwargs':{'enable_thinking':False}})['prompt'])
print(f"    conversation: {len(msgs)} messages, {n} chars templated")
PY
}
ask(){ local k=$1 req=$2
  curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$BD/$req" -o "$BD/tc_resp_$k.json"
  log "  $k: $(python3 - "$BD" "$k" <<'PY'
import json,sys
bd,k=sys.argv[1],sys.argv[2]
d=json.load(open(f'{bd}/tc_resp_{k}.json'))
if 'choices' not in d:
    print('REJECTED:', str(d.get('error',d))[:120]); raise SystemExit
T=d.get('timings',{}); c=d['choices'][0]['message']['content'] or ''
print(f"prompt_n {T.get('prompt_n')}, {T.get('prompt_ms',0)/1000:.1f}s, system prompt honoured: {'ZEBRA-9' in c}, answer {c.strip()[:40]!r}")
PY
)"
}
log "== with --prompt-truncate =="
if start trunc --prompt-truncate; then
  build_reqs | tee -a "$OUT"
  ask T1 tc_req_1.json
  ask T2 tc_req_2.json
  grep -E "prompt truncated|reusing chunk|forcing full|exceed" "$BD/tc_srv_trunc.log" | sed 's/.*| //;s/^/    /' | tee -a "$OUT"
  stop
fi
log "== without the flag (default behaviour) =="
if start plain; then ask T3 tc_req_1.json; stop; fi
killp
log "TRUNCATE-CHECK-DONE"
