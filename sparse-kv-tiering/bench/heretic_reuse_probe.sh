#!/bin/bash
# heretic_reuse_probe.sh — why did cache reuse not engage for chat requests?
# One server (-lv 5, so the "trying to reuse chunks" DEBUG line is visible), same text, four pairs:
#   raw      : /completion, no media anywhere          (this shape is known to slide)
#   chat     : /v1/chat/completions, no media anywhere (the only difference is the endpoint)
#   chatimg  : /v1/chat/completions, image in the follow-up
# Each pair: A = full text, B = the same text minus its first third (token-exact) + a new question.
# Pass per pair: "reusing chunk" in the log and prompt_n ~ the question only.
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
MODEL=${MODEL:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
MMPROJ=${MMPROJ:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/mmproj/mmproj-Qwen3.8-Flash-Next-heretic-2-F16.gguf}
OUT=$BD/heretic_reuse_probe.out; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3
"$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$MODEL" --mmproj "$MMPROJ" --device Vulkan0 -ngl 999 \
   --flash-attn on --jinja -c 32768 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
   --load-mode mmap --lazy-mode on --cache-reuse 256 --cache-ram 0 --ctx-checkpoints 4 -lv 5 > "$BD/rp_srv.log" 2>&1 &
PID=$!; ok=0
for i in $(seq 1 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
  kill -0 $PID 2>/dev/null || break; sleep 2
done
[ $ok = 1 ] || { log "LOAD FAILED"; exit 1; }
log "loaded"
python3 - "$BD" "$PORT" <<'PY' | tee -a "$OUT"
import base64,json,sys,urllib.request
bd,port=sys.argv[1],sys.argv[2]
def post(path,obj,timeout=1800):
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=timeout).read())
text=open(f'{bd}/ppl_corpus.txt',errors='replace').read()
text=text[:text.find('\n',30000)+1]                      # ~8k tokens: the probe only needs the mechanics
bt=post('/tokenize',{'content':text})['tokens']
cut=len(bt)//3
tail_txt=post('/detokenize',{'tokens':bt[cut:]})['content']
img='data:image/png;base64,'+base64.b64encode(open(f'{bd}/slide_test_image.png','rb').read()).decode()
QA="\n\nIn one short sentence, what is this text about?"
QB="\n\nAnswer in one short sentence: what is the text above about?"
tmpl=post('/apply-template',{'messages':[{'role':'user','content':'@@B@@'}],'chat_template_kwargs':{'enable_thinking':False}})['prompt']
head,tail=tmpl.split('@@B@@')
def raw(body,q):  return post('/completion',{'prompt':head+body+q+tail,'n_predict':16,'temperature':0,'cache_prompt':True})
def chat(parts,q):
    content=[{'type':'image_url','image_url':{'url':img}} if p=='IMG' else {'type':'text','text':p} for p in parts]
    if content[-1]['type']=='text': content[-1]['text']+=q
    else: content.append({'type':'text','text':q})
    return post('/v1/chat/completions',{'model':'x','messages':[{'role':'user','content':content}],
                'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
def show(tag,d):
    T=d.get('timings',{})
    print(f"  {tag}: prompt_n {T.get('prompt_n')}, {T.get('prompt_ms',0)/1000:.1f}s")
print(f"text {len(bt)} tokens, cut at {cut}")
print("== raw /completion ==");             show('A',raw(text,QA));               show('B',raw(tail_txt,QB))
print("== chat, no media ==");              show('A',chat([text],QA));            show('B',chat([tail_txt],QB))
print("== chat, image in the follow-up =="); show('A',chat([text],QA));           show('B',chat([tail_txt,'IMG'],QB))
PY
log "reuse attempts: $(grep -c 'trying to reuse chunks' "$BD/rp_srv.log"), shifts: $(grep -c 'shifting KV cache' "$BD/rp_srv.log")"
grep -E "trying to reuse chunks|reusing chunk|cache reuse is not|forcing full" "$BD/rp_srv.log" | sed 's/.*| //' | tee -a "$OUT"
kill $PID 2>/dev/null; wait $PID 2>/dev/null
log "REUSE-PROBE-DONE"
