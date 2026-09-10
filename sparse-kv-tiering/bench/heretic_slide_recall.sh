#!/bin/bash
# heretic_slide_recall.sh — QUALITY gate for the sliding window: does a SHIFTED cache still answer
# from the kept text? Needle prompt (160 "Fact N" lines, 56k tokens) instead of raw code, so the
# answer is checkable verbatim.
#   R1: prefill A = [chat head][160 facts][Q1]                      (full prefill, recall control at depth)
#   R2: send B = [chat head][facts from ~1/3 on][Q2] -> the server must evict the dropped third and
#       SHIFT the rest (prompt_n ~ Q2 only); the answer must still quote facts 80/157/m120 verbatim
#   R3: send C = [chat head][facts from ~2/3 on][Q3] -> a second slide on an already shifted cache
#   R4: control: fresh server, B with a full prefill -> same recall, so the shift is what we measure
# Judge: R2/R3 prompt_n ~ the question, no "forcing full prompt re-processing", recall equal to R4.
# Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
OUT=$BD/heretic_slide_recall.out; RES=$BD/heretic_slide_recall_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
start(){ local k=$1; shift; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c 131072 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 --cache-reuse 256 --cache-ram 0 -lv 4 "$@" > "$BD/sr_srv_$k.log" 2>&1 &
  PID=$!; local t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'failed to allocate[^\n]*|error[^\n]*' "$BD/sr_srv_$k.log" | head -2 | tr '\n' ' ')"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s, MemAvailable $(avail) GiB"
}
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
ask(){ local k=$1 req=$2; local t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d @"$req" -o "$BD/sr_resp_$k.json"
  log "  $k: $((SECONDS-t0))s; $(jq -c '{prompt_n: .timings.prompt_n, tokens_evaluated, prompt_s: (.timings.prompt_ms/100|floor/10), tg: (.timings.predicted_per_second*10|floor/10), acc: (if .timings.draft_n then ((.timings.draft_n_accepted/.timings.draft_n*100|floor)/100) else null end)}' "$BD/sr_resp_$k.json" 2>/dev/null)"
  log "  $k recall: $(python3 - "$BD" "$k" <<'PY'
import json,sys
bd,k=sys.argv[1],sys.argv[2]
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
c=(json.load(open(f'{bd}/sr_resp_{k}.json')).get('content') or '')
print(''.join(('%s:%s ' % (f, 'Y' if n[f] in c else 'n')) for f in ('3','80','157','m120')))
PY
)"
}
# prompts, token-exact against the running server: the reuse loop needs the first differing token of
# the new prompt to start a >= n_cache_reuse run that exists in the cache
build_prompts(){ python3 - "$BD" "$PORT" <<'PY'
import json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req,timeout=900).read())
# real code (no repetition) with the four needles inserted deep enough to survive both cuts. The
# needle prompt itself cannot be used here: it is one 160-fact block repeated, so dropping its first
# third yields a literal PREFIX of the cached prompt, which the server answers from a checkpoint
# instead of a slide (a valid path, but not the one under test).
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
text=text[:text.find('\n',215000)+1]
ins=[('3',"Fact 3: %s (marker 74)."),('80',"Fact 80: %s (marker 929)."),
     ('m120',"Fact 120: %s (marker 415)."),('157',"Fact 157: %s (marker 787).")]
body=''; prev=0
for i,(k,fmt) in enumerate(ins):                     # at 70 / 78 / 86 / 94 % of the text
    at=text.find('\n', int(len(text)*(0.70+0.08*i)))+1
    body+=text[prev:at]+fmt % needles[k]+'\n'; prev=at
body+=text[prev:]
# the templated wrapper, so the raw /completion prompts look exactly like a chat turn
tmpl=post('/apply-template',{'messages':[{'role':'user','content':'@@BODY@@'}],
                             'chat_template_kwargs':{'enable_thinking':False}})['prompt']
head,tail=tmpl.split('@@BODY@@')
# the three questions must differ: a prompt fully contained in the cache leaves no new token to
# evaluate, and the server then falls back to a full re-prefill (see HANDOFF 4g)
Q={'A':"\n\nNow answer in exactly this format and nothing else:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim.",
   'B':"\n\nAnswer in exactly this format and nothing else, using only the list above:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim.",
   'C':"\n\nReply now in exactly this format and nothing else, quoting from the list above:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim."}
# token-exact cuts: a character cut re-tokenizes the boundary differently and the reuse loop, which
# matches token ids, then finds nothing
bt=post('/tokenize',{'content':body})['tokens']
def cut(frac):
    return post('/detokenize',{'tokens':bt[int(len(bt)*frac):]})['content']
bodies={'A':body,'B':cut(1/3),'C':cut(2/3)}
toks={k:post('/tokenize',{'content':head+b+Q[k]+tail})['tokens'] for k,b in bodies.items()}
def run(a,b):    # longest common run of b inside a, as the reuse loop would find it
    best=0
    sa={}
    for i,t in enumerate(a): sa.setdefault(t,[]).append(i)
    for i in sa.get(b[8],[]):   # b[8] is past the chat head
        n=0
        while i+n<len(a) and 8+n<len(b) and a[i+n]==b[8+n]: n+=1
        best=max(best,n)
    return best
def pfx(a,b):
    n=0
    while n<len(a) and n<len(b) and a[n]==b[n]: n+=1
    return n
for k in ('B','C'):
    print(f"{k}: {len(toks[k])} tokens, longest run shared with A = {run(toks['A'],toks[k])} (need >= 256), "
          f"common prefix with A = {pfx(toks['A'],toks[k])} (must stay small: a prefix is answered from a "
          f"checkpoint, not a slide)")
for k,b in bodies.items():
    json.dump({'prompt':head+b+Q[k]+tail,'n_predict':200,'temperature':0,'cache_prompt':True},
              open(f'{bd}/sr_req_{k}.json','w'))
print(f"A {len(toks['A'])} tokens; B {len(toks['B'])}; C {len(toks['C'])}")
PY
}
log "== R1: server -c 131072 -ub 2048 --cache-reuse 256 --cache-ram 0; prefill A =="
if start R1; then
  build_prompts | tee -a "$OUT"
  ask A "$BD/sr_req_A.json"
  log "== R2: slide to B (drop the first third of the facts) =="
  ask B "$BD/sr_req_B.json"
  log "== R3: slide again to C =="
  ask C "$BD/sr_req_C.json"
  log "  shifts: $(grep -c 'shifting KV cache' "$BD/sr_srv_R1.log"), full re-processing forced: $(grep -c 'forcing full prompt re-processing' "$BD/sr_srv_R1.log") (want 2 and 0)"
  grep -E "recurrent state pos|reusing chunk" "$BD/sr_srv_R1.log" | tail -4 | sed 's/^.*| //;s/^/  /' | tee -a "$OUT"
  grep -iE "GGML_ASSERT|abort|error|nan" "$BD/sr_srv_R1.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
  stop
fi
log "== R4: control: fresh server, B with a full prefill =="
if start R4; then ask Bctl "$BD/sr_req_B.json"; stop; fi
killp
python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json,sys
bd,res=sys.argv[1],sys.argv[2]; L=[]
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
def r(k):
    try: return json.load(open(f'{bd}/sr_resp_{k}.json'))
    except Exception: return None
N=('3','80','157','m120')
for k,want in (('A',N),('B',N),('C',N),('Bctl',N)):
    d=r(k)
    if not d: L.append(f"- {k}: no result"); continue
    T=d.get('timings',{}); c=(d.get('content') or '')
    hit=[f for f in want if n[f] in c]
    L.append(f"- **{k}**: prompt_n {T.get('prompt_n')} of {d.get('tokens_evaluated')}, prompt {T.get('prompt_ms',0)/1000:.1f} s, "
             f"decode {T.get('predicted_per_second',0):.1f} t/s, accept {(T.get('draft_n_accepted',0)/T['draft_n']) if T.get('draft_n') else float('nan'):.2f}; "
             f"recall {len(hit)}/{len(want)} ({','.join(hit) or 'none'})")
b,bc=r('B'),r('Bctl')
if b and bc:
    L.append(f"- slid vs control answer: {'identical' if (b.get('content') or '')==(bc.get('content') or '') else 'differ (expected: the slid state keeps a trace of the dropped third)'}")
txt="\n".join(L); print(txt); open(res,'w').write("# Sliding window — recall on a shifted cache\n\n"+txt+"\n")
PY
log "SLIDE-RECALL-DONE"
