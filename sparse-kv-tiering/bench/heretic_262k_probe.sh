#!/bin/bash
# heretic_262k_probe.sh — largest ubatch that fits at -c 262144, and what prefill it buys.
# 2048 is known to fail: the prefill scratch needs one 5.84 GB Vulkan buffer and the device's
# maxBufferSize is 4 GiB. Walk down until one loads, then measure prefill on a ~20k prompt.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_262k_probe.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
try_ub(){ local ub=$1; killp
  log "== -c 262144 -ub $ub =="
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 262144 --parallel 1 -b 2048 -ub "$ub" --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 > "$BD/pb_srv_$ub.log" 2>&1 &
  local PID=$! ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  if [ $ok != 1 ]; then
    log "  FAILED: $(grep -oE 'allocation of size [0-9]+|failed to allocate[^\n]*|ErrorOutOf[A-Za-z]*' "$BD/pb_srv_$ub.log" | head -2 | tr '\n' ' ')"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; return 1
  fi
  log "  loaded, MemAvailable $(avail) GiB"
  python3 - "$BD" "$PORT" <<'PY' | tee -a "$OUT"
import json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj,t=1800):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
txt=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
txt=txt[:txt.find('\n',70000)+1]
d=post('/completion',{'prompt':txt+"\n\nIn one sentence, what is this?",'n_predict':24,'temperature':0,'cache_prompt':True})
T=d['timings']
print(f"    prefill {T['prompt_n']} tokens at {T['prompt_per_second']:.0f} tok/s, decode {T['predicted_per_second']:.1f} t/s")
PY
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; return 0
}
for ub in ${UBS:-2048 1536 1024 768 512}; do
  try_ub "$ub" && log "  ok at ub $ub"
done
killp
log "262K-PROBE-DONE"
