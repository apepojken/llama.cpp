#!/bin/bash
# heretic_qsa_phase0.sh — DESIGN.md §9 Phase 0: is the indexer's block selection concentrated and
# stable at depth? Production-like server with LLAMA_QSA_TRACE set, no speculation (so every traced
# step is one real token), a 70k-token code prompt, then two generations at that depth:
#   G1: continue the code (generic decode)
#   G2: answer about the four needle facts (retrieval-heavy decode)
# The trace file is rewritten every 64 steps; the harness prints the final summary.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_qsa_phase0.out; TRACE=$BD/qsa_phase0_trace.txt; PORT=8097
: > "$OUT"; : > "$TRACE"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3
LLAMA_QSA_TRACE=$TRACE "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
   --flash-attn on --jinja -c 131072 --parallel 1 -b 2048 -ub 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
   --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 > "$BD/p0_srv.log" 2>&1 &
PID=$!; ok=0
for i in $(seq 1 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
  kill -0 $PID 2>/dev/null || break; sleep 2
done
[ $ok = 1 ] || { log "LOAD FAILED"; exit 1; }
log "loaded (CHARS=${CHARS:-215000}); trace enabled: $(grep -c 'QSA selection trace enabled' "$BD/p0_srv.log")"
CHARS=${CHARS:-215000} python3 - "$BD" "$PORT" <<'PY' | tee -a "$OUT"
import json,sys,time,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj,t=3600):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
import os
text=text[:text.find('\n',int(os.environ.get('CHARS','215000')))+1]
ins=[('3',"Fact 3: %s (marker 74)."),('80',"Fact 80: %s (marker 929)."),
     ('m120',"Fact 120: %s (marker 415)."),('157',"Fact 157: %s (marker 787).")]
body=''; prev=0
for i,(k,fmt) in enumerate(ins):
    at=text.find('\n', int(len(text)*(0.40+0.15*i)))+1     # facts at 40/55/70/85 %
    body+=text[prev:at]+fmt % needles[k]+'\n'; prev=at
body+=text[prev:]
t0=time.time()
d=post('/completion',{'prompt':body,'n_predict':500,'temperature':0,'cache_prompt':True,'ignore_eos':True})
T=d['timings']; print(f"  G1 continue: prefill {T['prompt_n']} tokens at {T['prompt_per_second']:.0f} tok/s, decode {T['predicted_n']} tokens at {T['predicted_per_second']:.1f} t/s ({time.time()-t0:.0f}s)")
q="\n\n### Task\nQuote Fact 3, Fact 80, Fact 157 and the fact carrying (marker 415) verbatim, then explain in detail how each relates to the code above.\n"
t0=time.time()
d=post('/completion',{'prompt':body+q,'n_predict':500,'temperature':0,'cache_prompt':True,'ignore_eos':True})
T=d['timings']; c=d.get('content') or ''
hit=sum(1 for f in ('3','80','157','m120') if needles[f] in c)
print(f"  G2 retrieve: prompt_n {T['prompt_n']}, decode {T['predicted_n']} tokens at {T['predicted_per_second']:.1f} t/s, recall {hit}/4 ({time.time()-t0:.0f}s)")
PY
sleep 2
kill $PID 2>/dev/null; wait $PID 2>/dev/null
log "== trace summary =="
cat "$TRACE" | tee -a "$OUT"
log "PHASE0-DONE"
