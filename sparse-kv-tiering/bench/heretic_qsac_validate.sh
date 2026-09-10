#!/bin/bash
# heretic_qsac_validate.sh — validate change (C), the gathered QSA attention, as a same-binary A/B
# on the llama.cpp-qsab binary (with (B)+(A) already in): LLAMA_QSA_GATHER=0 (masked path) vs the
# gather (default engages at n_kv >= 32768; "1" forces it at any depth). Same model (MTP-qsa),
# same requests, q8_0 KV, temp 0. The gather never runs in prefill (n_tokens > 16), so the
# perplexity check does not apply; the per-op profile and the oracle/recall runs do.
#   C2  14k no-spec  forced ON vs OFF : greedy oracle (same visible set, different kernel: expect
#                                        identical or a late benign divergence); speed
#   C1  56k no-spec  ON vs OFF, perf  : FLASH_ATTN bucket; wall; recall
#   C3  56k spec6    ON vs OFF        : prod config; recall; speed; oracle
# One server at a time on 8097. Requires 8096 stopped (>=82 GiB free).
set -u
BD=${BD:-$(cd "$(dirname "$0")" && pwd)}
NEW=${LLAMA_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
ON=${ON:-/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M/MTP-qsa/Qwen3.8-Flash-Next-heretic-2-Q4_K_M-MTP-00001-of-00006.gguf}
REQ56=$BD/heretic_qsa_req.json; REQ16=$BD/heretic_qsa_req16k.json
OUT=$BD/heretic_qsac_validate.out; RES=$BD/heretic_qsac_validate_results.md; PORT=8097
: > "$OUT"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }
avail(){ awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo; }
A=$(avail); log "MemAvailable: ${A} GiB"; [ "$A" -lt 82 ] && { log "ABORT: need >=82 GiB (stop 8096)"; exit 1; }
for f in "$REQ56" "$REQ16" "$NEW/llama-server"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 1; }; done
killp(){ for p in $(ss -tlnp 2>/dev/null|grep ":$PORT"|grep -oP 'pid=\K[0-9]+'); do kill $p 2>/dev/null; done; sleep 3; }
# run <key> <ctx> <req> <gather env value: 0|1|default> <perf 0|1> [spec flags...]
run(){ local k=$1 ctx=$2 req=$3 gather=$4 perf=$5; shift 5; killp
  local envs=(); [ "$perf" = 1 ] && envs+=(GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1); [ "$gather" != default ] && envs+=(LLAMA_QSA_GATHER=$gather)
  log "== $k: -c $ctx gather=$gather perf=$perf ${*:-(no speculation)} =="
  env "${envs[@]}" "$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
     -c "$ctx" --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
     "$@" > "$BD/vc_srv_$k.log" 2>&1 &
  local PID=$! t0=$SECONDS ok=0
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  [ $ok = 1 ] || { log "  LOAD FAILED"; grep -iE "error|assert|abort" "$BD/vc_srv_$k.log" | grep -v 0x | tail -4 | sed 's/^/    /' | tee -a "$OUT"; kill $PID 2>/dev/null; return 1; }
  log "  loaded in $((SECONDS-t0))s"
  t0=$SECONDS
  curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$req" -o "$BD/vc_resp_$k.json"
  log "  request done in $((SECONDS-t0))s; $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), predicted_n, tg: (.predicted_per_second*10|floor/10)}' "$BD/vc_resp_$k.json" 2>/dev/null)"
  grep -iE "GGML_ASSERT|abort|error|nan" "$BD/vc_srv_$k.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
}
SPEC6="--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75"
run C2_14k_gather 20480 "$REQ16" 1 0
run C2_14k_masked 20480 "$REQ16" 0 0
run C1_56k_gather 65536 "$REQ56" default 1
run C1_56k_masked 65536 "$REQ56" 0 1
run C3_56k_masked_spec6 65536 "$REQ56" 0 0 $SPEC6
# C3 (gather, prod config) + C5: save -> erase -> restore -> EXTEND the conversation (the production
# restart case: the first decode after a restore refills the whole pooled cache in one step, and the
# gather runs on restored cells). Repeating the same prompt would instead trim the generated tail, a
# rollback the recurrent cache refuses, and force a full re-prefill.
SAVEDIR=$BD/slot-save-qsac; mkdir -p "$SAVEDIR"; killp
log "== C3_56k_gather_spec6: gather, spec6 (then C5 restore+extend) =="
"$NEW/llama-server" --host 127.0.0.1 --port $PORT -m "$ON" --device Vulkan0 -ngl 999 --flash-attn on --jinja \
   -c 65536 --parallel 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode on \
   --slot-save-path "$SAVEDIR" $SPEC6 > "$BD/vc_srv_C3_56k_gather_spec6.log" 2>&1 &
PID=$!; ok=0
for i in $(seq 1 600); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 2; done
if [ $ok = 1 ]; then
  t0=$SECONDS; curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$REQ56" -o "$BD/vc_resp_C3_56k_gather_spec6.json"
  log "  C3 request done in $((SECONDS-t0))s; $(jq -c '.timings | {prompt_n, pp: (.prompt_per_second|floor), predicted_n, tg: (.predicted_per_second*10|floor/10)}' "$BD/vc_resp_C3_56k_gather_spec6.json" 2>/dev/null)"
  log "== C5: save -> erase -> restore -> extend =="
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=save" -H 'Content-Type: application/json' -d '{"filename":"qsac.bin"}' | jq -c '{n_saved}' | sed 's/^/  save: /' | tee -a "$OUT"
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=erase" >/dev/null
  curl -s -m 300 -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" -H 'Content-Type: application/json' -d '{"filename":"qsac.bin"}' | jq -c '{n_restored}' | sed 's/^/  restore: /' | tee -a "$OUT"
  python3 - "$REQ56" "$BD/vc_resp_C3_56k_gather_spec6.json" "$BD/vc_req_extend.json" <<'PY'
import json,sys
req=json.load(open(sys.argv[1])); resp=json.load(open(sys.argv[2]))
req['messages']=req['messages']+[{'role':'assistant','content':resp['choices'][0]['message']['content']},
    {'role':'user','content':'Now quote Fact 42 verbatim, then Fact 99 verbatim. Nothing else.'}]
req['max_tokens']=120
json.dump(req,open(sys.argv[3],'w'))
PY
  t0=$SECONDS; curl -s -m 2400 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$BD/vc_req_extend.json" -o "$BD/vc_resp_C5_extend.json"
  log "  C5 extend done in $((SECONDS-t0))s; $(jq -c '.timings | {prompt_n, cache_n, pp: (.prompt_per_second|floor), predicted_n, tg: (.predicted_per_second*10|floor/10)}' "$BD/vc_resp_C5_extend.json" 2>/dev/null)"
  grep -iE "GGML_ASSERT|abort|error|nan|diverged|exceed" "$BD/vc_srv_C3_56k_gather_spec6.log" | grep -vE "0x|GET /|no error" | tail -3 | sed 's/^/  ERR: /' | tee -a "$OUT"
else
  log "  LOAD FAILED (C3 gather spec6)"; grep -iE "error|assert|abort" "$BD/vc_srv_C3_56k_gather_spec6.log" | grep -v 0x | tail -4 | sed 's/^/    /' | tee -a "$OUT"
fi
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
killp
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
def fam(path):
    dec=[b for b in parse(path) if b['total']<1e6][5:]
    nb=max(1,len(dec)); agg=collections.defaultdict(float); tot=sum(b['total'] for b in dec)/nb/1000
    fams=('FLASH_ATTN_EXT','GET_ROWS','CPY','CONCAT','SET_ROWS','TOP_K','MULTI_ADD','ROPE','SCALE','CONT','RMS_NORM_MUL RMS_NORM(128,')
    for b in dec:
        for name,(c,t) in b['ops'].items():
            key = next((f for f in fams if name.startswith(f)), 'ADD' if name=='ADD' else None)
            if key: agg[key]+=t/nb/1000
    return agg, tot, len(dec)
L=[]
g,tg=content(f'{bd}/vc_resp_C2_14k_gather.json'); m_,tm=content(f'{bd}/vc_resp_C2_14k_masked.json')
L.append(f"**C2 14k no-spec, gather forced vs masked:** {div(g,m_)}; recall gather {recall(g)}/4 masked {recall(m_)}/4; wall gather {ms(tg):.1f} ms/tok vs masked {ms(tm):.1f}")
g1,t1=content(f'{bd}/vc_resp_C1_56k_gather.json'); m1,t0=content(f'{bd}/vc_resp_C1_56k_masked.json')
L.append(f"**C1 56k no-spec (perf logger on both):** gather vs masked output: {div(g1,m1)}; recall gather {recall(g1)}/4 masked {recall(m1)}/4; wall gather {ms(t1):.1f} vs masked {ms(t0):.1f} ms/tok; prefill {t1.get('prompt_per_second',0):.0f} vs {t0.get('prompt_per_second',0):.0f} tok/s")
g3,t3=content(f'{bd}/vc_resp_C3_56k_gather_spec6.json'); m3,t3m=content(f'{bd}/vc_resp_C3_56k_masked_spec6.json'); a=t3.get('draft_n_accepted',0); mm=t3.get('draft_n',0)
L.append(f"**C3 56k spec6 (prod config):** gather {t3.get('predicted_per_second',0):.1f} t/s vs masked {t3m.get('predicted_per_second',0):.1f} t/s; recall gather {recall(g3)}/4 masked {recall(m3)}/4; accept {(a/mm if mm else 0):.3f}; output: {div(g3,m3)}")
c5,t5=content(f'{bd}/vc_resp_C5_extend.json')
f42=[l for l in open(f'{bd}/deep_prompt.txt') if l.startswith('Fact 42:')][0].split(': ',1)[1].split(' (marker')[0]
f99=[l for l in open(f'{bd}/deep_prompt.txt') if l.startswith('Fact 99:')][0].split(': ',1)[1].split(' (marker')[0]
L.append(f"**C5 restore + extend (refill path, gather on restored cells):** prompt_n {t5.get('prompt_n',0)}, cache hit {t5.get('cache_n','?')} tokens, prefill {t5.get('prompt_per_second',0):.0f} tok/s, decode {t5.get('predicted_per_second',0):.1f} t/s; Fact 42 quoted: {bool(c5 and f42 in c5)}, Fact 99 quoted: {bool(c5 and f99 in c5)}")
try:
    ga,gt,ng=fam(f'{bd}/vc_srv_C1_56k_gather.log'); ma,mt,nm=fam(f'{bd}/vc_srv_C1_56k_masked.log')
    L.append(""); L.append(f"**Per-op GPU time per decode step at 56k (ms), masked vs gather ({nm}/{ng} graphs):**")
    L.append("| op family | masked | gather | Δ |"); L.append("|---|---|---|---|")
    for k in sorted(set(ma)|set(ga), key=lambda x: -(ma[x]-ga[x])):
        L.append(f"| {k} | {ma[k]:.2f} | {ga[k]:.2f} | {ga[k]-ma[k]:+.2f} |")
    L.append(f"| **GPU total** | {mt:.1f} | {gt:.1f} | {gt-mt:+.1f} |")
    fa=[k for k in parse(f'{bd}/vc_srv_C1_56k_gather.log')[-1]['ops'] if k.startswith('FLASH_ATTN_EXT')]
    L.append(f"gather-run flash-attention shapes seen in the last graph: {[k[:80] for k in fa][:3]}")
except Exception as e: L.append(f"(perf parse failed: {e})")
txt="\n".join(L); print(txt)
open(res,'w').write("# Validation of (C) gathered QSA attention — same binary, LLAMA_QSA_GATHER on vs off\n\n"+txt+"\n")
PY
log "QSAC-VALIDATE-DONE"
