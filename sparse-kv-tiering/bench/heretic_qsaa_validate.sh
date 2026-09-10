#!/bin/bash
# heretic_qsaa_validate.sh — validate change (A), the pooled indexer-key cache, as a same-binary A/B:
# the llama.cpp-qsab binary with the pooled path ON (default) vs OFF (LLAMA_QSA_NO_POOLED_CACHE=1,
# i.e. change (B) only). Same model (MTP-qsa, QSA on), same requests, q8_0 KV, temp 0.
#   A2  14k no-spec  ON vs OFF : greedy oracle — the pooled math is the same ops on fewer rows, so
#                                the outputs are expected IDENTICAL (deterministic at 14k)
#   A1  56k no-spec  ON vs OFF : perf logger on both -> per-op buckets; recall; speed
#   A3  56k spec6    ON        : recall; speed (prod config); oracle vs OFF/old benign
#   A5  save/restore ON        : after A3's request, save the slot, restore it, decode again ->
#                                exercises state_read -> watermark reset -> full refill
#   A4  PPL @64k     ON vs OFF : must match within noise (prefill dirty path, 512 blocks/ubatch)
# One server at a time on 8097. Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
REQ56=$BD/heretic_qsa_req.json; REQ16=$BD/heretic_qsa_req16k.json; CORPUS=$BD/ppl_corpus.txt
OUT=$BD/heretic_qsaa_validate.out; RES=$BD/heretic_qsaa_validate_results.md; PORT=8097
SAVEDIR=$BD/slot-save-qsaa; mkdir -p "$SAVEDIR"
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
for f in "$REQ56" "$REQ16" "$CORPUS" "$NEW/llama-server" "$NEW/llama-perplexity"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 1; }; done
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
PID=
# start <key> <ctx> <pooled 0|1> <perf 0|1> [spec flags...]
start(){ local k=$1 ctx=$2 pooled=$3 perf=$4; shift 4; killp
  local envs=(); [ "$perf" = 1 ] && envs+=(GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1); [ "$pooled" = 0 ] && envs+=(LLAMA_QSA_NO_POOLED_CACHE=1)
  env "${envs[@]}" "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c "$ctx" --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     --slot-save-path "$SAVEDIR" "$@" > "$BD/va_srv_$k.log" 2>&1 &
  PID=$!; local t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; grep -iE "error|assert|abort" "$BD/va_srv_$k.log" | grep -v 0x | tail -4 | sed 's/^/    /' | tee -a "$OUT"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s; pooled cache: $(grep -c 'pooled indexer key cache' "$BD/va_srv_$k.log") (log line only visible at higher verbosity)"
}
ask(){ local k=$1 req=$2; local t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$req" -o "$BD/va_resp_$k.json"
  log "  request $k done in $((SECONDS-t0))s; $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), predicted_n, tg: (.predicted_per_second*10|floor/10)}' "$BD/va_resp_$k.json" 2>/dev/null)"
}
errs(){ grep -iE "GGML_ASSERT|abort|error|nan|diverged" "$BD/va_srv_$1.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"; }
stop(){ kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3; }
SPEC6="--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75"
# A2: 14k no-spec, ON vs OFF
log "== A2_14k_on: pooled ON =="; start A2_14k_on 20480 1 0 && { ask A2_14k_on "$REQ16"; errs A2_14k_on; stop; }
log "== A2_14k_off: pooled OFF =="; start A2_14k_off 20480 0 0 && { ask A2_14k_off "$REQ16"; errs A2_14k_off; stop; }
# A1: 56k no-spec with the perf logger, ON vs OFF
log "== A1_56k_on: pooled ON, perf =="; start A1_56k_on 65536 1 1 && { ask A1_56k_on "$REQ56"; errs A1_56k_on; stop; }
log "== A1_56k_off: pooled OFF, perf =="; start A1_56k_off 65536 0 1 && { ask A1_56k_off "$REQ56"; errs A1_56k_off; stop; }
# A3 + A5: 56k spec6 ON, then save / restore / decode again
log "== A3_56k_on_spec6: pooled ON, spec6 =="
if start A3_56k_on_spec6 65536 1 0 $SPEC6; then
  ask A3_56k_on_spec6 "$REQ56"; errs A3_56k_on_spec6
  log "== A5: slot save -> restore -> decode again (state_read refill path) =="
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=save" -H 'Content-Type: application/json' -d '{"filename":"qsaa.bin"}' | jq -c '{n_saved, id_slot}' | sed 's/^/  save: /' | tee -a "$OUT"
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=erase" >/dev/null
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" -H 'Content-Type: application/json' -d '{"filename":"qsaa.bin"}' | jq -c '{n_restored, id_slot}' | sed 's/^/  restore: /' | tee -a "$OUT"
  ask A5_after_restore "$REQ56"; errs A3_56k_on_spec6
  stop
fi
killp
# A4: perplexity at 64k, ON vs OFF
ppl(){ local k=$1 pooled=$2; log "== A4 PPL pooled=$pooled =="; local t0=$SECONDS
  local envs=(); [ "$pooled" = 0 ] && envs+=(LLAMA_QSA_NO_POOLED_CACHE=1)
  env "${envs[@]}" "$NEW/llama-perplexity" -m "$ON" -f "$CORPUS" -c 65536 -b 2048 -ub 2048 --chunks 1 --device Vulkan0 -ngl 999 --flash-attn on \
     --cache-type-k q8_0 --cache-type-v q8_0 --lazy-mode on > "$BD/va_ppl_$k.log" 2>&1
  log "  done in $((SECONDS-t0))s: $(grep -E 'Final estimate' "$BD/va_ppl_$k.log" | tail -1 | cut -c1-100)"
  grep -iE "GGML_ASSERT|abort|error" "$BD/va_ppl_$k.log" | grep -v 0x | tail -2 | sed 's/^/  ERR: /' | tee -a "$OUT"; }
ppl on 1
ppl off 0
python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json, re, os, collections, sys
bd, res = sys.argv[1], sys.argv[2]
n = json.load(open(f'{bd}/heretic_qsa_needles.json'))
def content(f):
    try: d = json.load(open(f)); return d['choices'][0]['message'].get('content') or '', d.get('timings', {})
    except Exception: return None, {}
def recall(c): return sum(1 for k in ('3','80','157','m120') if n[k] in c) if c else '-'
def div(a, b):
    if a is None or b is None: return 'n/a'
    if a == b: return f'IDENTICAL ({len(a)} chars)'
    i = next((i for i,(p,q) in enumerate(zip(a,b)) if p != q), min(len(a), len(b)))
    return f'diverge at char {i} of {len(a)}/{len(b)}'
def ms(T): return 1000/T['predicted_per_second'] if T.get('predicted_per_second') else float('nan')
def parse(path):
    blocks=[]; cur=None
    for line in open(path, errors='replace'):
        if line.startswith('Vulkan Timings:'): cur={'ops':{}, 'total':0.0}; continue
        if cur is None: continue
        m=re.match(r'^(.*?): (\d+) x ([\d.]+) us = ([\d.]+) us', line)
        if m: cur['ops'][m.group(1)]=(int(m.group(2)), float(m.group(4))); continue
        m=re.match(r'^Total time: ([\d.]+) us', line)
        if m: cur['total']=float(m.group(1)); blocks.append(cur); cur=None
    return blocks
def buckets(path):
    dec=[b for b in parse(path) if b['total']<1e6 and not any(' n=2048' in o for o in b['ops'])][5:]
    nb=max(1,len(dec)); agg=collections.defaultdict(float); tot=sum(b['total'] for b in dec)/nb/1000
    fams=('FLASH_ATTN_EXT','RMS_NORM_MUL RMS_NORM(128,','TOPK_QSA','TOP_K','GET_ROWS','MULTI_ADD','ROPE','SCALE','CONT','CPY','SET_ROWS')
    for b in dec:
        for name,(c,t) in b['ops'].items():
            key = next((f for f in fams if name.startswith(f)), 'ADD' if name=='ADD' else None)
            if key: agg[key]+=t/nb/1000
    return agg, tot, len(dec)
L=[]
c_on,t_on=content(f'{bd}/va_resp_A2_14k_on.json'); c_off,t_off=content(f'{bd}/va_resp_A2_14k_off.json'); old14,_=content(f'{bd}/dc_resp_R5_16k_nospec.json')
L.append(f"**A2 14k no-spec:** ON vs OFF: {div(c_on,c_off)}; OFF (=B fixed) vs old serving binary: {div(c_off,old14)}; ON vs old: {div(c_on,old14)}; recall ON {recall(c_on)}/4 OFF {recall(c_off)}/4; decode ON {ms(t_on):.1f} ms/tok, OFF {ms(t_off):.1f} ms/tok (old, no logger: 47.6)")
c1,t1=content(f'{bd}/va_resp_A1_56k_on.json'); c0,t0=content(f'{bd}/va_resp_A1_56k_off.json')
L.append(f"**A1 56k no-spec (perf logger on both):** ON vs OFF: {div(c1,c0)}; recall ON {recall(c1)}/4 OFF {recall(c0)}/4; wall ON {ms(t1):.1f} ms/tok vs OFF {ms(t0):.1f}; prefill ON {t1.get('prompt_per_second',0):.0f} vs OFF {t0.get('prompt_per_second',0):.0f} tok/s")
c3,t3=content(f'{bd}/va_resp_A3_56k_on_spec6.json'); oR1,_=content(f'{bd}/dc_resp_R1_56k_spec6.json'); a=t3.get('draft_n_accepted',0); m=t3.get('draft_n',0)
L.append(f"**A3 56k spec6 ON (prod config):** recall {recall(c3)}/4; decode {t3.get('predicted_per_second',0):.1f} t/s ({ms(t3):.1f} ms/tok; old serving binary 27.0); prefill {t3.get('prompt_per_second',0):.0f} tok/s (old 274); accept {(a/m if m else 0):.3f}; vs old: {div(c3,oR1)}")
c5,t5=content(f'{bd}/va_resp_A5_after_restore.json')
L.append(f"**A5 after slot save/erase/restore (refill path):** recall {recall(c5)}/4; decode {t5.get('predicted_per_second',0):.1f} t/s; prompt cache hit: prompt_n {t5.get('prompt_n',0)} (processed {t5.get('prompt_n',0) - t5.get('cache_n',0) if 'cache_n' in t5 else '?'}); vs A3 output: {div(c5,c3)}")
try:
    on_b,on_t,n1=buckets(f'{bd}/va_srv_A1_56k_on.log'); off_b,off_t,n0=buckets(f'{bd}/va_srv_A1_56k_off.log')
    L.append(""); L.append(f"**Per-op GPU time per decode step at 56k (ms), pooled OFF vs ON ({n0}/{n1} graphs):**")
    L.append("| op family | OFF | ON | Δ |"); L.append("|---|---|---|---|")
    for k in sorted(set(off_b)|set(on_b), key=lambda x: -(off_b[x]-on_b[x])):
        L.append(f"| {k} | {off_b[k]:.2f} | {on_b[k]:.2f} | {on_b[k]-off_b[k]:+.2f} |")
    L.append(f"| **GPU total** | {off_t:.1f} | {on_t:.1f} | {on_t-off_t:+.1f} |")
except Exception as e: L.append(f"(perf parse failed: {e})")
def ppl(k):
    try:
        t=open(f'{bd}/va_ppl_{k}.log', errors='replace').read(); mm=re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', t)
        return (float(mm.group(1)), float(mm.group(2))) if mm else None
    except Exception: return None
p1,p0=ppl('on'),ppl('off')
L.append(""); L.append(f"**A4 perplexity @64k:** pooled ON {p1} vs OFF {p0}" + (f" → Δ {p1[0]-p0[0]:+.4f} ({(p1[0]-p0[0])/p0[0]*100:+.2f} %)" if p1 and p0 else "") + f"; old serving binary was (1.1151, 0.00429)" + (f" → OFF vs old {(p0[0]-1.1151)/1.1151*100:+.2f} %, ON vs old {(p1[0]-1.1151)/1.1151*100:+.2f} %" if p1 and p0 else ""))
txt="\n".join(L); print(txt)
open(res,'w').write("# Validation of (A) pooled indexer-key cache — same binary, pooled ON vs OFF (kill switch)\n\n"+txt+"\n")
PY
log "QSAA-VALIDATE-DONE"
