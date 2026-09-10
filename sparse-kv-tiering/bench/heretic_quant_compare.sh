#!/bin/bash
# heretic_quant_compare.sh — IQ3_M vs Q4_K_M (both QSA on, MTP draft, projector loaded), same binary
# and flags as production (-c 131072 -ub 1024). Per quant, in one server session:
#   load     : time, RAM free, GPU-held memory
#   14k      : MTP speculation on: prefill, decode, draft accept, needle recall, and how far the greedy
#              output agrees with the Q4 no-spec reference (vc_resp_C2_14k_gather.json)
#   56k      : same with the 56k needle prompt
#   picture  : the test image alone, and the picture inside a 32k prompt with four facts
# Then, with no server running: 64k perplexity on real code (ppl_corpus.txt) if llama-perplexity exists.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
declare -A M=( [IQ3_M]=/home/jocke/LLM/models/heretic-IQ3_M-MTP/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-IQ3_M-MTP-00001-of-00006.gguf
               [Q4_K_M]=/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf )
QUANTS=${QUANTS:-IQ3_M Q4_K_M}
OUT=$BD/heretic_quant_compare.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
gtt(){ python3 -c "print(round($(cat /sys/class/drm/card0/device/mem_info_gtt_used)/2**30,1))"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
for q in $QUANTS; do
  killp; log "== $q =="; t0=$SECONDS
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "${M[$q]}" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 131072 --parallel 1 -b 2048 -ub 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 > "$BD/qc_srv_$q.log" 2>&1 &
  PID=$!; ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -iE 'error' "$BD/qc_srv_$q.log" | head -2 | tr '\n' ' ' | cut -c1-160)"; continue; }
  log "  loaded in $((SECONDS-t0))s: RAM free $(avail) GiB, GPU-held $(gtt) GiB"
  python3 - "$BD" "$PORT" "$q" <<'PY' | tee -a "$OUT"
import base64,json,sys,time,urllib.request
bd,port,q=sys.argv[1],sys.argv[2],sys.argv[3]
def post(path,obj,t=3600):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
def recall(c): return sum(1 for f in ('3','80','157','m120') if n[f] in c)
def chat(req): 
    d=post('/v1/chat/completions',req); T=d['timings']; c=d['choices'][0]['message']['content'] or ''
    return c,T
ref=json.load(open(f'{bd}/vc_resp_C2_14k_gather.json'))['choices'][0]['message']['content']
for tag,f in (('14k','heretic_qsa_req16k.json'),('56k','heretic_qsa_req.json')):
    req=json.load(open(f'{bd}/{f}')); t0=time.time(); c,T=chat(req)
    acc=(T.get('draft_n_accepted',0)/T['draft_n']) if T.get('draft_n') else float('nan')
    extra=''
    if tag=='14k':
        i=next((k for k in range(min(len(c),len(ref))) if c[k]!=ref[k]), min(len(c),len(ref)))
        extra=f", agrees with the Q4 reference for {i} of {len(ref)} chars"
    print(f"  {tag}: prefill {T['prompt_n']} at {T['prompt_per_second']:.0f} tok/s, decode {T['predicted_n']} at {T['predicted_per_second']:.1f} t/s, draft accept {acc:.2f}, recall {recall(c)}/4{extra} ({time.time()-t0:.0f}s)")
    json.dump({'content':c,'timings':T},open(f'{bd}/qc_resp_{q}_{tag}.json','w'))
img='data:image/png;base64,'+base64.b64encode(open(f'{bd}/slide_test_image.png','rb').read()).decode()
c,T=chat({'model':'x','max_tokens':24,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},
    'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':img}},{'type':'text','text':'What text is written in this picture? Answer with just that text.'}]}]})
print(f"  picture alone: {c.strip()[:40]!r}")
req=json.load(open(f'{bd}/si_req_kept_A.json')); t0=time.time(); c,T=chat(req)
pic='PLUM42' in c.upper().replace(' ','')
print(f"  picture in a {T['prompt_n']}-token prompt: prefill {T['prompt_per_second']:.0f} tok/s, decode {T['predicted_per_second']:.1f} t/s, recall {recall(c)}/4, picture read: {pic} ({time.time()-t0:.0f}s)")
PY
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
done
killp
if [ -x "$NEW/llama-perplexity" ]; then
  for q in $QUANTS; do
    log "== perplexity 64k, $q =="
    "$NEW/llama-perplexity" -m "${M[$q]}" --device Vulkan0 -ngl 999 --flash-attn on -c 65536 -b 2048 -ub 1024 \
       --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on --chunks 1 -f "$BD/ppl_corpus.txt" > "$BD/qc_ppl_$q.log" 2>&1
    grep -E "Final estimate|\[1\]" "$BD/qc_ppl_$q.log" | tail -2 | sed 's/^/  /' | tee -a "$OUT"
  done
fi
log "QUANT-COMPARE-DONE"
