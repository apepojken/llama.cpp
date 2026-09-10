#!/bin/bash
# heretic_qsab_validate.sh — validate change (B), block-granular QSA selection (llama.cpp-qsab,
# branch qwen4exp-qsa-block) against the serving binary (llama.cpp-main). Same model (MTP-qsa,
# QSA on), same requests as the earlier benches, q8_0 KV, temp 0.
#   V2  14k no-spec : smoke test + greedy oracle vs old (the old binary is deterministic at 14k)
#   V1  56k no-spec : Vulkan perf logger -> per-op buckets vs old P56; recall; speed
#   V3  56k spec6   : recall; speed; oracle vs old R1 (old is NOT deterministic at 56k: benign divergence)
#   V4  PPL @64k    : old vs new llama-perplexity on a real-code corpus (must match within noise)
# One server at a time on 8097. Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
OLD=${LLAMA_BIN_OLD:-/home/jocke/LLM/tools/llama.cpp-main/build/bin}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
REQ56=$BD/heretic_qsa_req.json; REQ16=$BD/heretic_qsa_req16k.json; CORPUS=$BD/ppl_corpus.txt
OUT=$BD/heretic_qsab_validate.out; RES=$BD/heretic_qsab_validate_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
for f in "$REQ56" "$REQ16" "$CORPUS" "$NEW/llama-server" "$NEW/llama-perplexity" "$OLD/llama-perplexity"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 1; }; done
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
# run <key> <ctx> <req> <perf 0|1> [spec flags...]
run(){ local k=$1 ctx=$2 req=$3 perf=$4; shift 4; killp
  log "== $k: NEW binary, -c $ctx ${*:-(no speculation)} perf=$perf =="
  local envs=(); [ "$perf" = 1 ] && envs=(GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1)
  env "${envs[@]}" "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c "$ctx" --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     "$@" > "$BD/vb_srv_$k.log" 2>&1 &
  local PID=$! t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; grep -iE "error|assert|abort" "$BD/vb_srv_$k.log" | grep -v 0x | tail -4 | sed 's/^/    /' | tee -a "$OUT"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s"
  t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$req" -o "$BD/vb_resp_$k.json"
  log "  request done in $((SECONDS-t0))s; $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), predicted_n, tg: (.predicted_per_second*10|floor/10)}' "$BD/vb_resp_$k.json" 2>/dev/null)"
  grep -iE "GGML_ASSERT|abort|error|nan" "$BD/vb_srv_$k.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
}
SPEC6="--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75"
run V2_14k_nospec 20480 "$REQ16" 0
run V1_56k_nospec 65536 "$REQ56" 1
run V3_56k_spec6  65536 "$REQ56" 0 $SPEC6
killp
# V4: perplexity at 64k on real code, old binary then new
ppl(){ local k=$1 bin=$2; log "== V4 PPL $k binary =="; local t0=$SECONDS
  "$bin/llama-perplexity" -m "$ON" -f "$CORPUS" -c 65536 -b 2048 -ub 2048 --chunks 1 --device Vulkan0 -ngl 999 --flash-attn on \
     --cache-type-k q8_0 --cache-type-v q8_0 --lazy-mode on > "$BD/vb_ppl_$k.log" 2>&1
  log "  done in $((SECONDS-t0))s: $(grep -E 'Final estimate' "$BD/vb_ppl_$k.log" | tail -1 | cut -c1-100)"
  grep -iE "GGML_ASSERT|abort|error" "$BD/vb_ppl_$k.log" | grep -v 0x | tail -2 | sed 's/^/  ERR: /' | tee -a "$OUT"; }
ppl old "$OLD"
ppl new "$NEW"
python3 - "$BD" "$RES" <<'PY' | tee -a "$OUT"
import json, re, os, collections, sys
bd, res = sys.argv[1], sys.argv[2]
n = json.load(open(f'{bd}/heretic_qsa_needles.json'))
def content(f):
    try: d = json.load(open(f)); return d['choices'][0]['message'].get('content') or '', d.get('timings', {})
    except Exception: return None, {}
def recall(c): return sum(1 for k in ('3','80','157','m120') if n[k] in c)
def div(a, b):
    if a is None or b is None: return 'n/a'
    if a == b: return f'IDENTICAL ({len(a)} chars)'
    i = next((i for i,(p,q) in enumerate(zip(a,b)) if p != q), min(len(a), len(b)))
    return f'diverge at char {i} of {len(a)}/{len(b)}'
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
lines=[]
# V2 oracle at 14k
c2,t2 = content(f'{bd}/vb_resp_V2_14k_nospec.json'); o14,_ = content(f'{bd}/prof_resp_P14.json'); o14b,_ = content(f'{bd}/dc_resp_R5_16k_nospec.json')
lines.append(f"**V2 14k no-spec (deterministic baseline):** new vs old P14: {div(c2,o14)}; new vs old R5: {div(c2,o14b)}; recall {recall(c2) if c2 else '-'}/4; decode {t2.get('predicted_per_second',0):.1f} t/s (old 21.0)")
# V1 56k no-spec
c1,t1 = content(f'{bd}/vb_resp_V1_56k_nospec.json'); o56,_ = content(f'{bd}/prof_resp_P56.json')
lines.append(f"**V1 56k no-spec:** new vs old P56: {div(c1,o56)} (old vs itself diverged at 810); recall {recall(c1) if c1 else '-'}/4; prefill {t1.get('prompt_per_second',0):.0f} tok/s (old 302); decode {t1.get('predicted_per_second',0):.1f} t/s (old 16.4, perf logger on both)")
# V3 56k spec6
c3,t3 = content(f'{bd}/vb_resp_V3_56k_spec6.json'); oR1,_ = content(f'{bd}/dc_resp_R1_56k_spec6.json')
a=t3.get('draft_n_accepted',0); m=t3.get('draft_n',0)
lines.append(f"**V3 56k spec6 (prod config):** new vs old R1: {div(c3,oR1)} (old vs itself diverged at 520); recall {recall(c3) if c3 else '-'}/4; prefill {t3.get('prompt_per_second',0):.0f} tok/s (old 274); decode {t3.get('predicted_per_second',0):.1f} t/s (old 27.0); accept {(a/m if m else 0):.3f}")
# perf buckets
try:
    new_b, new_tot, nn = buckets(f'{bd}/vb_srv_V1_56k_nospec.log'); old_b, old_tot, no = buckets(f'{bd}/prof_srv_P56.log')
    lines.append(""); lines.append(f"**Per-op GPU time per decode step at 56k (ms), old vs new ({no}/{nn} graphs):**")
    lines.append("| op family | old | new | Δ |"); lines.append("|---|---|---|---|")
    for k in sorted(set(old_b)|set(new_b), key=lambda x: -(old_b[x]-new_b[x])):
        lines.append(f"| {k} | {old_b[k]:.2f} | {new_b[k]:.2f} | {new_b[k]-old_b[k]:+.2f} |")
    lines.append(f"| **GPU total** | {old_tot:.1f} | {new_tot:.1f} | {new_tot-old_tot:+.1f} |")
except Exception as e: lines.append(f"(perf parse failed: {e})")
# PPL
def ppl(k):
    try:
        t=open(f'{bd}/vb_ppl_{k}.log', errors='replace').read(); m=re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', t)
        return (float(m.group(1)), float(m.group(2))) if m else None
    except Exception: return None
po, pn = ppl('old'), ppl('new')
lines.append(""); lines.append(f"**V4 perplexity @64k (real code, 1 chunk):** old {po} vs new {pn}" + (f" → Δ {pn[0]-po[0]:+.4f} ({(pn[0]-po[0])/po[0]*100:+.2f} %)" if po and pn else ""))
txt="\n".join(lines); print(txt)
open(res,'w').write("# Validation of (B) block-granular QSA selection — llama.cpp-qsab vs serving binary\n\n"+txt+"\n")
PY
log "QSAB-VALIDATE-DONE"
