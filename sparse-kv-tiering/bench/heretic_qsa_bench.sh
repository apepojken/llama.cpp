#!/bin/bash
# heretic_qsa_bench.sh — does enabling QSA (block-sparse attention) on the Heretic Q4_K_M-MTP model
# help, and do the -new depth kernels (pooled-key cache + gathered attention) win once QSA is on?
# Background: both Heretic GGUFs shipped with compress_ratios = all 0 (QSA off -> dense attention on
# the 12 full-attention layers). MTP-qsa/ is the same weights with the metadata patched (see
# ~/LLM/sparse-kv-tiering/DESIGN.md §12). Every earlier Heretic bench ran with QSA off on both sides.
#
# Configs (all: temp 0, MTP spec on as in prod, q8_0 KV, -c 65536, one ~55k-token synthetic deep
# prompt, ~300 tokens decoded; verbatim-recall of 4 facts checked):
#   A  main/off : llama.cpp-main binary  + MTP      (QSA off)  = production today
#   B  main/on  : llama.cpp-main binary  + MTP-qsa  (QSA on; base path: masked attention, no pooled cache/gather)
#   C  her/on   : build-heretic binary   + MTP-qsa  (QSA on; -new kernels: pooled cache + gather at n_kv>=32768)
#   D  her/off  : build-heretic binary   + MTP      (QSA off) -> isolates the -new base-backend effect on Q4
# One server at a time on 8097. Requires 8096 stopped (>=82 GiB free). No systemd, no chat content.
# Usage: heretic_qsa_bench.sh [build]   ("build" = only build the request file and exit)
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
MAIN=/home/jocke/LLM/tools/llama.cpp-main/build/bin/llama-server
HER=/home/jocke/LLM/tools/llama.cpp-heretic/build-heretic/bin/llama-server
MD=/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M
OFF=$MD/MTP/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf
ON=$MD/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf
CTX=65536; PORT=8097; MINFREE=82
OUT=$BD/heretic_qsa_bench.out; RES=$BD/heretic_qsa_results.md; REQ=$BD/heretic_qsa_req.json
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }

# request: 12x deep prompt (~55k tok) + verbatim-recall instruction + short summary; temp 0, no thinking
build_req(){ python3 - "$BD" "$REQ" <<'PY'
import json,sys,re
bd,req=sys.argv[1],sys.argv[2]
b=open(f'{bd}/deep_prompt.txt').read()
facts={int(m.group(1)):m.group(2) for m in re.finditer(r'^Fact (\d+): (.*)$',b,re.M)}
body=lambda t: re.sub(r'\s*\(marker \d+\)\.?\s*$','',t).strip()
m120=re.search(r'\(marker (\d+)\)',facts[120]).group(1)
instr=("\n\nNow answer in exactly this format and nothing else:\n"
       "1. Fact 3 verbatim.\n2. Fact 80 verbatim.\n3. Fact 157 verbatim.\n"
       f"4. The fact that carries (marker {m120}) verbatim.\n"
       "5. A summary of the whole list in about 100 words.")
p="\n".join([b]*12)+instr
json.dump({'model':'x','messages':[{'role':'user','content':p}],'max_tokens':320,'temperature':0,
           'chat_template_kwargs':{'enable_thinking':False}},open(req,'w'))
json.dump({'3':body(facts[3]),'80':body(facts[80]),'157':body(facts[157]),'m120':body(facts[120]),'marker':m120},
          open(f'{bd}/heretic_qsa_needles.json','w'))
print(f"request built: {len(p)} chars (~{len(p)//4} tok), {len(facts)} facts/copy x12, marker for fact 120 = {m120}")
PY
}
if [ "${1:-}" = build ]; then build_req; exit $?; fi

: > "$OUT"
A=$(avail); log "MemAvailable: ${A} GiB"
[ "$A" -lt $MINFREE ] && { log "ABORT: need >=${MINFREE} GiB free (stop 8096)"; exit 1; }
for f in "$MAIN" "$HER"; do [ -x "$f" ] || { log "ABORT: missing binary $f"; exit 1; }; done
for f in "$OFF" "$ON"; do [ -f "$f" ] || { log "ABORT: missing model $f"; exit 1; }; done
build_req | tee -a "$OUT"

killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
# start <bin> <model> <log> [extra flags...]; waits for /health; sets PID; returns 1 on load failure
start(){ local bin=$1 model=$2 logf=$3; shift 3; killp
  "$bin" --host 127.0.0.1 --port $PORT -m "$model" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c $CTX --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap \
     --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75 "$@" > "$logf" 2>&1 &
  PID=$!; local t0=$SECONDS
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { log "  loaded in $((SECONDS-t0))s, MemAvailable $(avail) GiB"; return 0; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  log "  LOAD FAILED:"; grep -iE "assert|abort|not found|out of bounds|error|failed to" "$logf"|grep -v 0x|tail -5|sed 's/^/    /'|tee -a "$OUT"
  kill $PID 2>/dev/null; return 1; }
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
# run <key> <label> <bin> <model> [extra flags...]
run(){ local k=$1 label=$2 bin=$3 model=$4; shift 4
  log "== $k: $label =="
  local a=$(avail); [ "$a" -lt $MINFREE ] && { log "  SKIP: only ${a} GiB free"; return 1; }
  start "$bin" "$model" "$BD/qsa_srv_$k.log" "$@" || return 1
  grep -E "creating indexer KV cache|pooled indexer key cache|creating MTP draft context|n_rs_seq" "$BD/qsa_srv_$k.log" | sed -E 's/^[0-9.]+ I //' | cut -c1-110 | sed 's/^/  /' | tee -a "$OUT"
  local t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$REQ" -o "$BD/qsa_resp_$k.json"
  log "  request done in $((SECONDS-t0))s, MemAvailable $(avail) GiB"
  grep -iE "assert|abort|out of bounds|GGML_ASSERT|failed" "$BD/qsa_srv_$k.log" | grep -v 0x | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
  stop; return 0; }

run A "llama.cpp-main + MTP (QSA off) = prod"          "$MAIN" "$OFF" --lazy-mode on
run B "llama.cpp-main + MTP-qsa (QSA on)"              "$MAIN" "$ON"  --lazy-mode on
run C "build-heretic (-new kernels) + MTP-qsa (QSA on)" "$HER"  "$ON"
run D "build-heretic (-new kernels) + MTP (QSA off)"    "$HER"  "$OFF"
killp

python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json,sys,os
bd,res=sys.argv[1],sys.argv[2]
needles=json.load(open(f'{bd}/heretic_qsa_needles.json'))
names={'A':'main / QSA off (prod)','B':'main / QSA on','C':'build-heretic / QSA on','D':'build-heretic / QSA off'}
R={}
for k in 'ABCD':
    f=f'{bd}/qsa_resp_{k}.json'
    if not os.path.exists(f) or os.path.getsize(f)==0: R[k]=None; continue
    d=json.load(open(f))
    if 'choices' not in d: R[k]={'err':str(d)[:200]}; continue
    c=d['choices'][0]['message'].get('content') or ''; T=d.get('timings',{})
    a=T.get('draft_n_accepted',0); n=T.get('draft_n',0)
    rec=sum(1 for key in ('3','80','157','m120') if needles[key] in c)
    R[k]={'pn':T.get('prompt_n',0),'pps':T.get('prompt_per_second',0),'ps':T.get('prompt_ms',0)/1000,
          'gn':T.get('predicted_n',0),'gps':T.get('predicted_per_second',0),'acc':(a/n if n else 0),
          'recall':rec,'content':c}
    open(f'{bd}/qsa_out_{k}.txt','w').write(c)
def fmt(k):
    r=R[k]
    if r is None: return f"| {k} | {names[k]} | (no result) | | | |"
    if 'err' in r: return f"| {k} | {names[k]} | ERROR {r['err']} | | | |"
    return (f"| {k} | {names[k]} | {r['pps']:.1f} tok/s ({r['ps']:.0f}s, {r['pn']} tok) | "
            f"{r['gps']:.1f} t/s ({r['gn']} tok) | {r['acc']:.3f} | {r['recall']}/4 |")
lines=["| cfg | binary / model | prefill @~55k | decode @~55k | MTP accept | recall |","|---|---|---|---|---|---|"]+[fmt(k) for k in 'ABCD']
def div(x,y):
    a,b=R.get(x),R.get(y)
    if not a or not b or 'err' in a or 'err' in b: return f"{x} vs {y}: n/a"
    ca,cb=a['content'],b['content']
    if ca==cb: return f"{x} vs {y}: IDENTICAL ({len(ca)} chars)"
    i=next((i for i,(p,q) in enumerate(zip(ca,cb)) if p!=q),min(len(ca),len(cb)))
    return f"{x} vs {y}: diverge at char {i} of {len(ca)}/{len(cb)}"
lines+=["","greedy oracle: "+div('A','B'),"               "+div('B','C'),"               "+div('A','D'),"               "+div('C','D')]
txt="\n".join(lines); print(txt)
open(res,'w').write("# Heretic Q4_K_M: QSA on/off x binary, ~55k synthetic deep prompt (temp 0, MTP on, q8_0 KV, -c 65536)\n\n"+txt+"\n")
PY
log "QSA-BENCH-DONE"
