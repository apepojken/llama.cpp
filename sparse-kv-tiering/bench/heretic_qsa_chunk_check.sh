#!/bin/bash
# heretic_qsa_chunk_check.sh — validate the chunked indexer scoring (scratch bound) on llama.cpp-qsab:
#   K1: 14k no-spec greedy oracle vs the serving-binary output (R5) — same math per query column,
#       kernel tiling may differ, so identical or a late benign divergence
#   K2: PPL @64k (prefill exercises the chunked path: 2048-token ubatches -> 4 chunks) vs 1.1151/1.1154
#   K3: allocation probe: -c 262144 -ub 2048 must now create its context (health), then a 14k prefill
#       for the throughput number at that ubatch
# One server at a time on 8097. Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
REQ16=$BD/heretic_qsa_req16k.json; CORPUS=$BD/ppl_corpus.txt
OUT=$BD/heretic_qsa_chunk_check.out; RES=$BD/heretic_qsa_chunk_check_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
start(){ local k=$1 ctx=$2 ub=$3; shift 3; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c "$ctx" --parallel 1 -b 2048 -ub "$ub" --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     "$@" > "$BD/ck_srv_$k.log" 2>&1 &
  PID=$!; local t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'buffer of size [0-9]+|allocation of size [0-9]+|failed to allocate[^\n]*' "$BD/ck_srv_$k.log" | head -2 | tr '\n' ' ')"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s, MemAvailable $(avail) GiB"
}
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
log "== K1: 14k no-spec oracle =="
if start K1 20480 2048; then
  curl -s -m 1200 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$REQ16" -o "$BD/ck_resp_K1.json"
  log "  $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), tg: (.predicted_per_second*10|floor/10)}' "$BD/ck_resp_K1.json" 2>/dev/null)"
  grep -iE "GGML_ASSERT|abort|error" "$BD/ck_srv_K1.log" | grep -vE "0x|GET /|no error" | tail -2 | sed 's/^/  ERR: /' | tee -a "$OUT"; stop
fi
log "== K3: allocation probe -c 262144 -ub 2048 (spec6, prod flags) =="
if start K3 262144 2048 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75; then
  curl -s -m 1200 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$REQ16" -o "$BD/ck_resp_K3.json"
  log "  14k at -c 262144 -ub 2048: $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), tg: (.predicted_per_second*10|floor/10)}' "$BD/ck_resp_K3.json" 2>/dev/null); MemAvailable $(avail) GiB"
  grep -iE "GGML_ASSERT|abort|error" "$BD/ck_srv_K3.log" | grep -vE "0x|GET /|no error" | tail -2 | sed 's/^/  ERR: /' | tee -a "$OUT"; stop
fi
killp
log "== K2: PPL @64k =="; t0=$SECONDS
"$NEW/llama-perplexity" -m "$ON" -f "$CORPUS" -c 65536 -b 2048 -ub 2048 --chunks 1 --device Vulkan0 -ngl 999 --flash-attn on \
   --cache-type-k q8_0 --cache-type-v q8_0 --lazy-mode on > "$BD/ck_ppl.log" 2>&1
log "  done in $((SECONDS-t0))s: $(grep -E 'Final estimate' "$BD/ck_ppl.log" | tail -1 | cut -c1-90)"
python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json,sys,re
bd,res=sys.argv[1],sys.argv[2]; L=[]
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
def content(f):
    try: d=json.load(open(f)); return d['choices'][0]['message'].get('content') or '', d.get('timings',{})
    except Exception: return None, {}
def div(a,b):
    if a is None or b is None: return 'n/a'
    if a==b: return f'IDENTICAL ({len(a)} chars)'
    i=next((i for i,(p,q) in enumerate(zip(a,b)) if p!=q),min(len(a),len(b))); return f'diverge at char {i} of {len(a)}/{len(b)}'
k1,t1=content(f'{bd}/ck_resp_K1.json'); old,_=content(f'{bd}/dc_resp_R5_16k_nospec.json')
L.append(f"**K1 14k no-spec, chunked scoring vs serving binary:** {div(k1,old)}; recall {sum(1 for k in ('3','80','157','m120') if n[k] in k1) if k1 else '-'}/4; prefill {t1.get('prompt_per_second',0):.0f} tok/s")
k3,t3=content(f'{bd}/ck_resp_K3.json')
L.append(f"**K3 -c 262144 -ub 2048:** {'context created; ' if k3 is not None else 'FAILED; '}14k prefill {t3.get('prompt_per_second',0):.0f} tok/s, decode {t3.get('predicted_per_second',0):.1f} t/s, recall {sum(1 for k in ('3','80','157','m120') if n[k] in k3) if k3 else '-'}/4")
try:
    t=open(f'{bd}/ck_ppl.log',errors='replace').read(); m=re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', t)
    L.append(f"**K2 PPL @64k (chunked prefill):** {m.group(1)} ± {m.group(2)} (serving binary 1.1151, pooled-on before chunking 1.1154)")
except Exception as e: L.append(f"K2: {e}")
txt="\n".join(L); print(txt); open(res,'w').write("# Chunked indexer scoring (scratch bound) checks\n\n"+txt+"\n")
PY
log "CHUNK-CHECK-DONE"
