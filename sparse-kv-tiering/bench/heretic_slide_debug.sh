#!/bin/bash
# heretic_slide_debug.sh — isolate WHY a slid (evicted + shifted) cache answers worse than a full
# prefill (heretic_slide_recall_results.md: slid 0/4, control 4/4).
# Per variant, one server: prefill A (~33k tokens of code with 4 facts inserted at 70-94%), ask a
# question (full-prefill reference, in the same session), then slide to B = A minus its first third
# and ask again. Same needles, same model state -> A's recall IS the control for B's.
#   qsa       : production path (QSA on, pooled cache on, gather on)
#   nopool    : LLAMA_QSA_NO_POOLED_CACHE=1  -> pooled indexer keys recomputed every ubatch
#   nogather  : LLAMA_QSA_GATHER=0           -> masked attention instead of gathered
#   dense     : the QSA-off model (MTP/)     -> no indexer at all: isolates the core K-shift
#   f16       : q8_0 KV -> f16 KV            -> isolates the quantized K-shift (dequant/rope/requant)
# Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
M_QSA=${M_QSA:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
M_DENSE=${M_DENSE:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
OUT=$BD/heretic_slide_debug.out; RES=$BD/heretic_slide_debug_results.md; PORT=8097
VARIANTS=${VARIANTS:-qsa nopool nogather dense}
# MMPROJ=<path> loads a vision projector too: cache reuse must still engage for text-only prompts
MMPROJ=${MMPROJ:-}
MM=(); [ -n "$MMPROJ" ] && MM=(--mmproj "$MMPROJ")
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
ask(){ local k=$1 req=$2; local t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d @"$req" -o "$BD/sd_resp_$k.json"
  local line
  line=$(python3 - "$BD" "$k" <<'PY'
import json,sys
bd,k=sys.argv[1],sys.argv[2]
d=json.load(open(f'{bd}/sd_resp_{k}.json')); T=d.get('timings',{}); c=(d.get('content') or '')
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
hit=sum(1 for f in ('3','80','157','m120') if n[f] in c)
print(f"prompt_n {T.get('prompt_n')}, {T.get('prompt_ms',0)/1000:.1f}s, {T.get('predicted_per_second',0):.1f} t/s, "
      f"accept {(T.get('draft_n_accepted',0)/T['draft_n']) if T.get('draft_n') else float('nan'):.2f}, recall {hit}/4 | {c[:90]!r}")
PY
)
  log "    $k: $line"
}
build_prompts(){ python3 - "$BD" "$PORT" <<'PY'
import json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req,timeout=900).read())
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
text=text[:text.find('\n',100000)+1]
ins=[('3',"Fact 3: %s (marker 74)."),('80',"Fact 80: %s (marker 929)."),
     ('m120',"Fact 120: %s (marker 415)."),('157',"Fact 157: %s (marker 787).")]
body=''; prev=0
for i,(k,fmt) in enumerate(ins):
    at=text.find('\n', int(len(text)*(0.70+0.08*i)))+1
    body+=text[prev:at]+fmt % needles[k]+'\n'; prev=at
body+=text[prev:]
tmpl=post('/apply-template',{'messages':[{'role':'user','content':'@@BODY@@'}],
                             'chat_template_kwargs':{'enable_thinking':False}})['prompt']
head,tail=tmpl.split('@@BODY@@')
Q={'A':"\n\nNow answer in exactly this format and nothing else:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim.",
   'B':"\n\nAnswer in exactly this format and nothing else, using only the text above:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim."}
bt=post('/tokenize',{'content':body})['tokens']
bodies={'A':body,'B':post('/detokenize',{'tokens':bt[len(bt)//3:]})['content']}
for k,b in bodies.items():
    json.dump({'prompt':head+b+Q[k]+tail,'n_predict':120,'temperature':0,'cache_prompt':True},
              open(f'{bd}/sd_req_{k}.json','w'))
print(f"    A {len(post('/tokenize',{'content':head+bodies['A']+Q['A']+tail})['tokens'])} tokens, "
      f"B {len(post('/tokenize',{'content':head+bodies['B']+Q['B']+tail})['tokens'])} tokens")
PY
}
run_variant(){ local v=$1; killp
  local model=$M_QSA ktype=q8_0 envs=()
  case $v in
    nopool)   envs=(LLAMA_QSA_NO_POOLED_CACHE=1);;
    nogather) envs=(LLAMA_QSA_GATHER=0);;
    dense)    model=$M_DENSE;;
    f16)      ktype=f16;;
  esac
  log "== variant $v =="
  env "${envs[@]}" "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$model" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c 65536 --parallel 1 -b 2048 -ub 2048 --cache-type-k $ktype --cache-type-v $ktype --load-mode mmap --lazy-mode on \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 --cache-reuse 256 --cache-ram 0 -lv 4 "${MM[@]}" > "$BD/sd_srv_$v.log" 2>&1 &
  local PID=$! ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'failed to allocate[^\n]*|error[^\n]*' "$BD/sd_srv_$v.log" | head -2 | tr '\n' ' ')"; kill $PID 2>/dev/null; return 1; }
  [ "$v" = qsa ] && build_prompts | tee -a "$OUT"
  ask "${v}_A" "$BD/sd_req_A.json"
  ask "${v}_B" "$BD/sd_req_B.json"
  log "    shifts $(grep -c 'shifting KV cache' "$BD/sd_srv_$v.log"), forced re-prefill $(grep -c 'forcing full prompt' "$BD/sd_srv_$v.log")"
  grep -iE "GGML_ASSERT|abort|diverged|refilling the pooled|unpooled cells" "$BD/sd_srv_$v.log" | tail -2 | sed 's/^.*: //;s/^/    note: /' | tee -a "$OUT"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
}
for v in $VARIANTS; do run_variant "$v"; done
killp
{ echo "# Sliding window — where the shifted cache loses the answer"; echo
  echo "A = full prefill (reference), B = same text minus its first third, slid onto A's cache."; echo
  grep -E "variant|_A:|_B:|shifts|note:" "$OUT" | sed 's/^[0-9:]* *//'; } > "$RES"
log "SLIDE-DEBUG-DONE"
