#!/bin/bash
# heretic_shiftfix_oracle.sh — the standing regression check for a build that is about to be
# deployed: the 14k greedy oracle must be byte-identical to the reference produced before the
# K-shift/memory changes (vc_resp_C2_14k_gather.json, 2026-09-08). Same flags as that run.
# Nothing here shifts the cache, so this checks that the changes are inert on the normal path.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
REF=$BD/vc_resp_C2_14k_gather.json; OUT=$BD/heretic_shiftfix_oracle.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3
"$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
   -c 20480 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
   > "$BD/so_srv.log" 2>&1 &
PID=$!; ok=0
for i in $(seq 1 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
  kill -0 $PID 2>/dev/null || break; sleep 2
done
[ $ok = 1 ] || { log "LOAD FAILED"; exit 1; }
curl -s -m 1200 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$BD/heretic_qsa_req16k.json" -o "$BD/so_resp.json"
kill $PID 2>/dev/null; wait $PID 2>/dev/null
python3 - "$BD" "$REF" <<'PY' | tee -a "$OUT"
import json,sys
bd,ref=sys.argv[1],sys.argv[2]
def get(f):
    d=json.load(open(f));
    return (d.get('content') or (d.get('choices') or [{}])[0].get('message',{}).get('content') or ''), d.get('timings',{})
new,T=get(f'{bd}/so_resp.json'); old,_=get(ref)
same=new==old
i=next((k for k in range(min(len(new),len(old))) if new[k]!=old[k]), min(len(new),len(old)))
print(f"14k oracle vs reference: {'IDENTICAL' if same else f'DIVERGES at char {i}'} ({len(new)} vs {len(old)} chars); "
      f"prefill {T.get('prompt_per_second',0):.0f} tok/s, decode {T.get('predicted_per_second',0):.1f} t/s")
if not same:
    print("new:", repr(new[max(0,i-60):i+60])); print("old:", repr(old[max(0,i-60):i+60]))
PY
echo "SHIFTFIX-ORACLE-DONE" | tee -a "$OUT"
