#!/bin/bash
# heretic_slide_check.sh — sliding window on the hybrid model: middle-range eviction + KV shift via the
# server's prompt-cache reuse (--cache-reuse). Real-code corpus, raw /completion (no chat template).
#   S0: allocation probe: -c 131072 -ub 2048 (prod flags + --cache-reuse 256)
#   S1: prefill A = corpus[0:C1]                                   (~60k tokens)
#   S2: send B = corpus[C0:C1] + question  -> the server must evict [0,C0) and SHIFT the kept part
#       (log: "reusing chunk ... shifting KV cache"), process only the question, answer from B's start
#   S3: send C = corpus[C0b:C1] + question -> a second slide on the shifted cache (pooled refill again)
#   S4: control: fresh server, B + question with a full prefill -> compare the answer with S2
# Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
CORPUS=$BD/ppl_corpus.txt; OUT=$BD/heretic_slide_check.out; RES=$BD/heretic_slide_check_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
start(){ local k=$1; shift; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c 131072 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 --cache-reuse 256 --cache-ram 0 -lv 4 "$@" > "$BD/sl_srv_$k.log" 2>&1 &
  PID=$!; local t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'allocation of size [0-9]+|failed to allocate[^\n]*' "$BD/sl_srv_$k.log" | head -2 | tr '\n' ' ')"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s, MemAvailable $(avail) GiB"
}
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
ask(){ local k=$1 req=$2; local t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d @"$req" -o "$BD/sl_resp_$k.json"
  log "  $k: $((SECONDS-t0))s; $(jq -c '{tokens_evaluated, tokens_cached, prompt_n: .timings.prompt_n, prompt_ms: (.timings.prompt_ms|floor), tg: (.timings.predicted_per_second*10|floor/10), acc: (if .timings.draft_n then ((.timings.draft_n_accepted/.timings.draft_n*100|floor)/100) else null end)}' "$BD/sl_resp_$k.json" 2>/dev/null)"
  log "  $k answer: $(jq -r '.content' "$BD/sl_resp_$k.json" 2>/dev/null | tr '\n' ' ' | cut -c1-120)"
}
# prompts are built TOKEN-exactly against the running server (the reuse loop needs the new prompt's
# first token to be a cache token followed by >= n_cache_reuse matches; a character cut re-tokenizes
# the boundary differently): A = corpus text; B = detokenize(tokens(A)[n/3:]); C = ...[2n/3:]
build_prompts(){ python3 - "$BD" "$PORT" <<'PY'
import json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]; t=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
A=t[:t.find('\n',215000)+1]
def post(path,obj):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req,timeout=600).read())
toks=post('/tokenize',{'content':A})['tokens']; n=len(toks); T0=n//3; T0b=(2*n)//3
B=post('/detokenize',{'tokens':toks[T0:]})['content']; C=post('/detokenize',{'tokens':toks[T0b:]})['content']
rb=post('/tokenize',{'content':B})['tokens']; print(f"round-trip B: first 8 tokens equal: {rb[:8]==toks[T0:T0+8]}, len {len(rb)} vs {n-T0}")
q="\n\n### Task: quote the very first line of the text above, verbatim, and nothing else.\n"
json.dump({'prompt':A,'n_predict':1,'temperature':0,'cache_prompt':True,'ignore_eos':True}, open(f'{bd}/sl_req_A.json','w'))
json.dump({'prompt':B+q,'n_predict':32,'temperature':0,'cache_prompt':True,'ignore_eos':True}, open(f'{bd}/sl_req_B.json','w'))
json.dump({'prompt':C+q,'n_predict':32,'temperature':0,'cache_prompt':True,'ignore_eos':True}, open(f'{bd}/sl_req_C.json','w'))
open(f'{bd}/sl_expect_B.txt','w').write(B.split('\n',1)[0]); open(f'{bd}/sl_expect_C.txt','w').write(C.split('\n',1)[0])
print(f"A {n} tokens; B from token {T0} ({n-T0} tokens); C from {T0b} ({n-T0b} tokens)")
PY
}
log "== S0/S1: server -c 131072 -ub 2048 --cache-reuse 256; prefill A =="
if start S1; then
  build_prompts | tee -a "$OUT"
  ask A "$BD/sl_req_A.json"
  log "== S2: slide to B (evict the first ~20k tokens, shift the rest) =="
  ask B "$BD/sl_req_B.json"
  log "  reuse log: $(grep -cE 'reusing chunk|shifting KV cache' "$BD/sl_srv_S1.log") shift lines; $(grep -oE 'after context reuse, new n_past = [0-9]+' "$BD/sl_srv_S1.log" | tail -1)"
  log "== S3: slide again to C =="
  ask C "$BD/sl_req_C.json"
  log "  reuse log: $(grep -oE 'after context reuse, new n_past = [0-9]+' "$BD/sl_srv_S1.log" | tail -1)"
  grep -iE "GGML_ASSERT|abort|error|diverged|exceed|nan" "$BD/sl_srv_S1.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
  stop
fi
log "== S4: control: fresh server, B with a full prefill =="
if start S4; then ask Bctl "$BD/sl_req_B.json"; stop; fi
killp
python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json,sys
bd,res=sys.argv[1],sys.argv[2]; L=[]
def r(k):
    try: return json.load(open(f'{bd}/sl_resp_{k}.json'))
    except Exception: return None
eB=open(f'{bd}/sl_expect_B.txt').read().strip(); eC=open(f'{bd}/sl_expect_C.txt').read().strip()
for k,exp in (('B',eB),('C',eC),('Bctl',eB)):
    d=r(k)
    if not d: L.append(f"- {k}: no result"); continue
    T=d.get('timings',{}); c=(d.get('content') or '').strip()
    L.append(f"- **{k}**: prompt tokens processed {T.get('prompt_n')} of {d.get('tokens_evaluated')} evaluated (cached {d.get('tokens_cached')}), prompt time {T.get('prompt_ms',0)/1000:.1f} s, decode {T.get('predicted_per_second',0):.1f} t/s, draft accept {(T.get('draft_n_accepted',0)/T['draft_n']) if T.get('draft_n') else float('nan'):.2f}; first line correct: {exp[:40] in c}; answer: {c[:80]!r}")
b,bc=r('B'),r('Bctl')
if b and bc: L.append(f"- slide vs control answers {'IDENTICAL' if (b.get('content') or '').strip()==(bc.get('content') or '').strip() else 'differ'} (legitimately may differ: the slid state keeps a trace of the dropped text)")
txt="\n".join(L); print(txt); open(res,'w').write("# Sliding window (middle eviction + KV shift) checks\n\n"+txt+"\n")
PY
log "SLIDE-CHECK-DONE"
