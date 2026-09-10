#!/bin/bash
# heretic_slide_image.sh — the sliding window with a picture in the conversation.
# Chat requests (images need the chat endpoint), one user message built from parts:
#   [text: code with 4 needle facts] [image: "PLUM 42"] [text: rest of the code] [question]
# Three placements, each: full prefill (reference answer) -> compaction-shaped follow-up that drops
# the first third of the text -> the server must reuse the rest.
#   kept    : the image sits in the KEPT part -> its chunk and cells must shift with the run
#   dropped : the image sits in the DROPPED part -> its cells are evicted, its map entry erased
#   newturn : no image in the cached text, one arrives with the follow-up -> reuse must still work
# Pass: follow-up prompt tokens ~ the question only, no "forcing full prompt re-processing",
# needle recall equal to the reference, and the image still readable where it survives.
# Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_slide_image.out; RES=$BD/heretic_slide_image_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
start(){ local k=$1; killp
  "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 65536 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 \
     --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 -lv 4 > "$BD/si_srv_$k.log" 2>&1 &
  PID=$!; local t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED: $(grep -oE 'failed to allocate[^\n]*|error[^\n]*' "$BD/si_srv_$k.log" | head -2 | tr '\n' ' ')"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s, MemAvailable $(avail) GiB"
}
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
ask(){ local k=$1 req=$2
  curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$req" -o "$BD/si_resp_$k.json"
  log "    $k: $(python3 - "$BD" "$k" <<'PY'
import json,sys
bd,k=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(f'{bd}/si_resp_{k}.json'))
except Exception as e:
    print('no response:', e); raise SystemExit
if 'choices' not in d:
    print('error:', str(d)[:160]); raise SystemExit
T=d.get('timings',{}); c=d['choices'][0]['message']['content'] or ''
n=json.load(open(f'{bd}/heretic_qsa_needles.json'))
hit=sum(1 for f in ('3','80','157','m120') if n[f] in c)
img='PLUM 42' in c.upper().replace('PLUM42','PLUM 42')
print(f"prompt_n {T.get('prompt_n')}, {T.get('prompt_ms',0)/1000:.1f}s, {T.get('predicted_per_second',0):.1f} t/s, "
      f"needles {hit}/4, image {'read' if img else 'no'} | {c[:80]!r}")
PY
)"
}
build_reqs(){ python3 - "$BD" "$PORT" <<'PY'
import base64,json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req,timeout=900).read())
img='data:image/png;base64,'+base64.b64encode(open(f'{bd}/slide_test_image.png','rb').read()).decode()
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
text=text[:text.find('\n',100000)+1]
ins=[('3',"Fact 3: %s (marker 74)."),('80',"Fact 80: %s (marker 929)."),
     ('m120',"Fact 120: %s (marker 415)."),('157',"Fact 157: %s (marker 787).")]
body=''; prev=0
for i,(k,fmt) in enumerate(ins):                      # needles at 70-94%, they survive the cut
    at=text.find('\n', int(len(text)*(0.70+0.08*i)))+1
    body+=text[prev:at]+fmt % needles[k]+'\n'; prev=at
body+=text[prev:]
bt=post('/tokenize',{'content':body})['tokens']
def piece(a,b=None):                                  # token-exact slices of the body
    return post('/detokenize',{'tokens':bt[a:b]})['content']
half=len(bt)//2
# the chat template TRIMS the message content, so a slice that starts with whitespace (code
# indentation) loses it and its first token stops matching the cache - and the reuse loop can only
# anchor on that first token. Pick the first cut whose slice already starts on a non-space, then
# verify the run the loop will find.
def templ_tokens(s):
    p=post('/apply-template',{'messages':[{'role':'user','content':s}],
                              'chat_template_kwargs':{'enable_thinking':False}})['prompt']
    return post('/tokenize',{'content':p})['tokens']
def shared_run(a,b):
    n=0
    while n<len(a) and n<len(b) and a[n]==b[n]: n+=1   # common prefix (the template head)
    pos={}
    for i,t in enumerate(a): pos.setdefault(t,[]).append(i)
    best=0
    for i in pos.get(b[n],[])[:4000]:
        k=0
        while i+k<len(a) and n+k<len(b) and a[i+k]==b[n+k]: k+=1
        best=max(best,k)
    return n,best
A_toks=templ_tokens(body)
third=len(bt)//3
for cand in range(len(bt)//3, len(bt)//3 + 400):
    if piece(cand,cand+1)[:1].strip() == '':
        continue
    n,run = shared_run(A_toks, templ_tokens(piece(cand)))
    if run >= 1000:
        third=cand
        print(f"    cut at token {cand}: common prefix {n}, matchable run {run}")
        break
else:
    print("    WARNING: no cut with a matchable run found; the test will not slide")
QA="\n\nAnswer in exactly this format and nothing else:\n1. Fact 80 verbatim.\n2. Fact 157 verbatim.\n3. The text written in the picture."
QB="\n\nReply in exactly this format and nothing else:\n1. Fact 80 verbatim.\n2. Fact 157 verbatim.\n3. The text written in the picture."
QT="\n\nReply in exactly this format and nothing else:\n1. Fact 80 verbatim.\n2. Fact 157 verbatim."
def req(parts, q):
    content=[{'type':'image_url','image_url':{'url':img}} if p=='IMG' else {'type':'text','text':p} for p in parts]
    if content[-1]['type']=='text':
        content[-1]['text']+=q
    else:
        content.append({'type':'text','text':q})
    return {'model':'x','messages':[{'role':'user','content':content}],'max_tokens':160,'temperature':0,
            'chat_template_kwargs':{'enable_thinking':False}}
# kept: image at the halfway mark, so the follow-up (drops the first third) keeps it
json.dump(req([piece(0,half),'IMG',piece(half)],QA), open(f'{bd}/si_req_kept_A.json','w'))
json.dump(req([piece(third,half),'IMG',piece(half)],QB), open(f'{bd}/si_req_kept_B.json','w'))
# dropped: image inside the first third -> evicted by the follow-up
json.dump(req([piece(0,third//2),'IMG',piece(third//2)],QA), open(f'{bd}/si_req_dropped_A.json','w'))
json.dump(req([piece(third)],QT),                          open(f'{bd}/si_req_dropped_B.json','w'))
# newturn: cached text has no image, the follow-up brings one
json.dump(req([piece(0)],QT),                              open(f'{bd}/si_req_newturn_A.json','w'))
json.dump(req([piece(third),'IMG'],QB),                    open(f'{bd}/si_req_newturn_B.json','w'))
print(f"    body {len(bt)} tokens; cut at {third}; image at {half}")
PY
}
for v in kept dropped newturn; do
  log "== $v =="
  start "$v" || continue
  [ "$v" = kept ] && build_reqs | tee -a "$OUT"
  ask "${v}_A" "$BD/si_req_${v}_A.json"
  ask "${v}_B" "$BD/si_req_${v}_B.json"
  log "    shifts $(grep -c 'shifting KV cache' "$BD/si_srv_$v.log"), forced re-prefill $(grep -c 'forcing full prompt' "$BD/si_srv_$v.log"), reuse refused $(grep -c 'cache reuse is not supported' "$BD/si_srv_$v.log")"
  grep -iE "GGML_ASSERT|abort|Chunk not found|exception" "$BD/si_srv_$v.log" | tail -2 | sed 's/^.*: //;s/^/    note: /' | tee -a "$OUT"
  stop
done
killp
{ echo "# Sliding window with an image in the conversation"; echo
  echo "A = full prefill (reference), B = the same minus its first third, slid onto A's cache."; echo
  grep -E "^[0-9:]+ (==|    (kept|dropped|newturn|shifts|note))" "$OUT" | sed 's/^[0-9:]* *//'; } > "$RES"
log "SLIDE-IMAGE-DONE"
