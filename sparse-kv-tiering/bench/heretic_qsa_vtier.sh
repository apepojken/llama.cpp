#!/bin/bash
# heretic_qsa_vtier.sh — does a drop-only hot tier hurt answers? 70k needle prompt in chat form (the
# raw-completion form makes the model stop after one token). Servers: none (control), then
# LLAMA_QSA_VTIER=<frac>[:prefill] — an LRU over blocks (union across layers; creation and every
# selection count as a use); blocks outside the tier cannot be selected. prefill=1 restricts the prompt
# processing too (a hard RAM cap), prefill=0 restricts decode only (a disk tier that fills during
# prefill). Two questions per server; judge by needle recall (4 facts at 40/55/70/85 % of the text).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_qsa_vtier.out; PORT=8097
FRACS=${FRACS:-none 0.25:0 0.10:0}
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
[ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
for spec in $FRACS; do
  killp
  frac=${spec%%:*}; pf=${spec#*:}; [ "$pf" = "$spec" ] && pf=1
  envs=(); [ "$frac" != none ] && envs=(LLAMA_QSA_VTIER=$frac LLAMA_QSA_VTIER_PREFILL=$pf)
  log "== hot tier: $frac (restricts prefill too: $pf) =="
  env "${envs[@]}" "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
     --flash-attn on --jinja -c 131072 --parallel 1 -b 2048 -ub 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
     --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 > "$BD/vt_srv_${frac}_p$pf.log" 2>&1 &
  PID=$!; ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; continue; }
  CHARS=${CHARS:-215000} python3 - "$BD" "$PORT" <<'PY' | tee -a "$OUT"
import json,os,sys,time,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj,t=3600):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
text=text[:text.find('\n',int(os.environ.get('CHARS','215000')))+1]
ins=[('3',"Fact 3: %s (marker 74)."),('80',"Fact 80: %s (marker 929)."),
     ('m120',"Fact 120: %s (marker 415)."),('157',"Fact 157: %s (marker 787).")]
body=''; prev=0
for i,(k,fmt) in enumerate(ins):
    at=text.find('\n', int(len(text)*(0.40+0.15*i)))+1
    body+=text[prev:at]+fmt % needles[k]+'\n'; prev=at
body+=text[prev:]
tmpl=post('/apply-template',{'messages':[{'role':'user','content':'@@BODY@@'}],'chat_template_kwargs':{'enable_thinking':False}})['prompt']
head,tail=tmpl.split('@@BODY@@')
def recall(c): return ''.join(f"{f}:{'Y' if needles[f] in c else 'n'} " for f in ('3','80','157','m120'))
qs=[("Q1","\n\nNow answer in exactly this format and nothing else:\n1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n4. The fact that carries (marker 415) verbatim."),
    ("Q2","\n\nReply in exactly this format and nothing else, quoting from the text above:\n1. The fact that carries (marker 415) verbatim.\n2. Fact 157 verbatim.\n3. Fact 80 verbatim.\n4. Fact 3 verbatim.")]
for tag,q in qs:
    t0=time.time()
    d=post('/completion',{'prompt':head+body+q+tail,'n_predict':160,'temperature':0,'cache_prompt':True})
    T=d['timings']; c=d.get('content') or ''
    print(f"  {tag}: prompt_n {T['prompt_n']} at {T['prompt_per_second']:.0f} tok/s, decode {T['predicted_n']} tokens at {T['predicted_per_second']:.1f} t/s, recall {recall(c)} ({time.time()-t0:.0f}s)")
PY
  grep -E "qsa_vtier: (filter|step [0-9]+:)" "$BD/vt_srv_${frac}_p$pf.log" | sed 's/^.*W *//;s/^/    /' | sort -u | head -4 | tee -a "$OUT"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
done
killp
log "VTIER-DONE"
