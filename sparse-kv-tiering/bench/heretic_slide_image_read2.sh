#!/bin/bash
# heretic_slide_image_read2.sh — does the picture survive the shift? Same pair as the "kept" case but
# with a question only the picture can answer. A generates ONE token, so no answer of its own is
# left in the recurrent state before the slide (that trace, not the image, broke the first attempt).
# full prefill. The two answers must agree (and say PLUM42).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_slide_image_read2.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB"; exit 1; }
# requests ir2_req_A.json / ir2_req_B.json are prepared by the caller
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
run(){ local k=$1; shift; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 65536 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 -lv 4 > "$BD/ir2_srv_$k.log" 2>&1 &
  local PID=$! ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; return 1; }
  for req in "$@"; do
    curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$BD/$req" -o "$BD/ir2_resp_${k}_$(basename $req .json).json"
  done
  log "  $k: shifts $(grep -c 'shifting KV cache' "$BD/ir2_srv_$k.log"), forced re-prefill $(grep -c 'forcing full prompt' "$BD/ir2_srv_$k.log")"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
}
log "== slid: prefill A (image in the middle), then the picture question on the slid cache =="
run slid ir2_req_A.json ir2_req_B.json
log "== control: the same question, fresh server, full prefill =="
run ctl ir2_req_B.json
killp
python3 - "$BD" <<'PY' | tee -a "$OUT"
import json,sys
bd=sys.argv[1]
def g(f):
    d=json.load(open(f'{bd}/{f}'))
    return d['choices'][0]['message']['content'], d.get('timings',{})
s,ts=g('ir2_resp_slid_ir2_req_B.json'); c,tc=g('ir2_resp_ctl_ir2_req_B.json')
print(f"  slid   : prompt_n {ts.get('prompt_n')}, {ts.get('prompt_ms',0)/1000:.1f}s -> {s.strip()[:60]!r}")
print(f"  control: prompt_n {tc.get('prompt_n')}, {tc.get('prompt_ms',0)/1000:.1f}s -> {c.strip()[:60]!r}")
norm=lambda x: x.upper().replace(' ','').strip().strip('."*')
print(f"  picture read after the shift: {'PLUM42' in norm(s)}; control: {'PLUM42' in norm(c)}; same answer: {norm(s)==norm(c)}")
PY
log "IMAGE-READ2-DONE"
