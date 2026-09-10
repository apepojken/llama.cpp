#!/bin/bash
# heretic_vision_compare.sh — can each quant read text out of pictures? Six rendered strings (words,
# digits, punctuation, an error line), same F16 projector, exact-match after whitespace/case folding.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
declare -A M=( [IQ3_M]=/home/jocke/LLM/models/heretic-IQ3_M-MTP/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-IQ3_M-MTP-00001-of-00006.gguf
               [Q4_K_M]=/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf )
OUT=$BD/heretic_vision_compare.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
for q in IQ3_M Q4_K_M; do
  killp; log "== $q =="
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "${M[$q]}" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 16384 --parallel 1 -b 2048 -ub 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on > "$BD/vc2_srv_$q.log" 2>&1 &
  PID=$!; ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; continue; }
  python3 - "$BD" "$PORT" <<'PY' | tee -a "$OUT"
import base64,json,sys,urllib.request,re
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj,t=600):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
norm=lambda s: re.sub(r'[\s`"\'*]+','',s).lower()
ok=0
for c in json.load(open(f'{bd}/vision_cases.json')):
    img='data:image/png;base64,'+base64.b64encode(open(f"{bd}/{c['file']}",'rb').read()).decode()
    d=post('/v1/chat/completions',{'model':'x','max_tokens':40,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},
        'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':img}},{'type':'text','text':'Transcribe the text in this picture exactly. Reply with the text only.'}]}]})
    a=(d['choices'][0]['message']['content'] or '').strip()
    hit=norm(a)==norm(c['text']); ok+=hit
    print(f"  {'OK ' if hit else 'BAD'} expected {c['text']!r:40s} got {a[:45]!r}")
print(f"  score {ok}/6")
PY
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
done
killp
log "VISION-COMPARE-DONE"
