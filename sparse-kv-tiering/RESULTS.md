# Results — numbers for the write-up (all measured on the author's box, 2026-09-08 … 09-10)

Box: Strix Halo (Ryzen AI MAX+ 395, gfx1151, 123 GiB unified RAM, GTT cap 120 GiB), llama.cpp Vulkan/RADV.
Model: Qwen3.8-Flash-Next Heretic (qwen4exp, 125B-A6B, Q4_K_M, MTP draft), q8_0 KV cache, 12 block-sparse
attention layers + 36 recurrent (GDN) layers. Every number below comes from a harness in `~/LLM/bench/`;
the file that produced it is named in the right-hand column.

## 1. Enabling the model's own sparse attention (QSA) — 56k-token prompt, four needle facts, no speculation
| binary / mode | prefill | decode | recall | source |
|---|---|---|---|---|
| QSA off (as shipped: `compress_ratios` all zero in the GGUF) | 295 tok/s | 19.4 t/s | 4/4 | `heretic_qsa_bench.out` (DESIGN §12a) |
| QSA on (12-byte metadata patch, same weights) | 283 tok/s | **25.9 t/s (+34 %)** | 4/4 | same |
| QSA on + block-level selection + pooled index + gathered attention, MTP speculation | 274 tok/s | **28.8 t/s** warm | 4/4 | DESIGN §12b |
| same at 14k depth with speculation | 349 tok/s | 30.1 t/s | 4/4 | DESIGN §12a |

Target-only GPU time per decode step at 56k: 59.1 → 42.5 ms after the three depth changes. Shallow-context
ceiling for this quant ≈ 40 t/s (weight bytes).

## 2. The sliding window instead of compaction (`--cache-reuse 256 --cache-ram 0`)
Client drops its oldest third and resends the rest; the server evicts and shifts instead of re-processing.
| scenario | tokens processed | prompt time | quality | source |
|---|---|---|---|---|
| 47,075-token prompt, code with needles, slid | **71** | **1.7 s** | recall 4/4, answer byte-identical to control | `heretic_slide_recall_results.md` |
| same prompt, fresh full prefill (control) | 47,075 | 218.4 s | recall 4/4 | same |
| second slide on the already-shifted cache (23,576 tokens) | 72 | 1.2 s | recall 4/4 | same |
| 46,944-token code prompt, slid / control | 22 / 46,944 | 1.3 s / 174.0 s | — | `heretic_slide_check_results.md` |
| 21,875-token prompt with a **picture** inside the kept part, slid / control | 64 / 21,875 | 2.1 s / 103.6 s | picture read correctly both ways ("PLUM 42") | `heretic_slide_image_read2.out` |
| picture in the dropped part / arriving with the new turn | 39 / 151 | 0.9 s / 2.2 s | needles equal to reference | `heretic_slide_image_results.md` |

## 3. Server-side prompt truncation (`--prompt-truncate`) — no client changes at all
| scenario | result | source |
|---|---|---|
| 14,964-token conversation into an 8,192 window, flag on | served: system prompt kept, newest turn answered correctly (7,798 tokens processed) | `heretic_truncate_check.out` |
| one more turn on top | **40 tokens, 0.8 s** (trim + shift) | same |
| same request, flag off | rejected: "exceeds the available context size" (unchanged default) | same |

## 4. Prefill vs ubatch vs window (the scratch bound)
| window | ubatch | loads | RAM free after load | prefill | source |
|---|---|---|---|---|---|
| 131k | 512 / 1024 / 2048 | yes | — | 255 / 300 / 370 tok/s (14k depth) | DESIGN §12b |
| 262k | 2048 | **no**: one 6.98 GB Vulkan allocation, device max 4 GiB | — | — | `heretic_262k_probe.out` |
| 262k | 1536 / 1024 / 512 | yes | 3 / 9 / 15 GiB | 124 / 162 / **208** tok/s (22.6k depth) | same |
At 262k a bigger ubatch is slower: the compute scratch (∝ window × ubatch) evicts the model's own pages.

## 5. Memory
| item | 131k window | 262k window |
|---|---|---|
| attention + indexer + pooled + draft caches | 2.63 GiB | 5.26 GiB |
| recurrent state (fixed) | 0.77 GiB | 0.77 GiB |
| one context checkpoint (hybrid model: recurrent state + K/V) | ~365 MiB | ~619 MiB |
| checkpoints at the default 32 / at `--ctx-checkpoints 4` | 11.4 / 1.4 GiB | 19.3 / 2.4 GiB |
GPU-held (GTT) total with the model: 97.1 GiB at 131k, 99.6 GiB at 262k.

## 6. Bug found on the way (affects upstream llama.cpp forks with a rotated KV cache)
With a quantized KV cache this tree stores K in a Hadamard-rotated basis across the whole head. The K-shift
graph handed the rotation a view of only the rotary dims (64 of 256 on this model), so every shifted cell's
K was destroyed: the model kept generating but could only see tokens decoded after the shift (needle recall
0/4, draft accept 1.00 while it copied its own prompt back). Any model with `n_rot < n_embd_head_k` and a
quantized cache under `--ctx-shift` or `--cache-reuse` is affected. Fix: `[TAG_KV_SHIFT_HADAMARD]`.

## 7. Phase 0 of the disk tier — is the block selection concentrated and stable? (70,501-token prompt, 1,006 decoded tokens)
| hot set = blocks used in the last W steps | hot blocks per layer (of 17,625) | miss rate per step |
|---|---|---|
| W = 1 | 513 (3 %) | 32–50 % |
| W = 16 | ~2,000 (11 %) | 10–14 % |
| W = 64 | ~3,500–4,200 (20–24 %) | **4–6 %** |
| W = 256 | ~5,500–7,400 (31–42 %) | 2.4–3.2 % |
30–45 % of blocks were never selected; the top 10 % of blocks take 64–81 % of selections; one decode step
touches 2,286 blocks across the 12 layers (114 MiB). A quarter-size hot tier would fetch ~1.3 MiB per token.
Source: `qsa_phase0_70k_analysis.txt`, `qsa_phase0_analyze.py`.

## 8. Can the cache simply be capped at 25 % or 10 % with a smart retention policy? No (measured)
Blocks outside a virtual hot tier were hidden from the selection (no disk behind it). 70k prompt, four facts:

| hot tier | keep by | recall |
|---|---|---|
| none | — | 4/4 |
| 25 % / 10 % | most recently used | 1/4 / 0/4 |
| 25 % / 10 % | most often selected during prefill | 1/4 / 1/4 |

Neither recency nor frequency predicts what a later question needs; a dropped block cannot be asked for. Only a
tier that keeps the index resident and fetches on demand preserves retrieval. Source: `heretic_qsa_vtier*.out`.

## 9. IQ3_M vs Q4_K_M, same binary and flags, back to back (2026-09-10)
| | IQ3_M | Q4_K_M | source |
|---|---|---|---|
| GPU-held after load / RAM free | 76.0 / 39.3 GiB | 97.1 / 18.3 GiB | `heretic_quant_compare.out` |
| load time | 51 s | 80 s | same |
| 56k: prefill / decode / draft accept | 248 tok/s / 34.1 t/s / 0.94 | 220 / 29.3 / 0.87 | same |
| 14k: prefill / decode | 317 / 37.0 | 303 / 19.3 (cold n-gram pages) | same |
| needle recall 14k, 56k | 4/4, 4/4 | 4/4, 4/4 | same |
| perplexity 64k code | 1.1179 ± 0.0043 | 1.1151 ± 0.0043 | `qc_ppl_*.log` |
| six rendered strings transcribed | 6/6 | 6/6 | `heretic_vision_compare.out` |
Deployed IQ3_M on 8096 at 14:05.

## Code tags (branch `qwen4exp-qsa-block`, worktree `~/LLM/tools/llama.cpp-qsab`)
`[TAG_HYBRID_MID_RM]` `[TAG_HYBRID_TAIL_SHIFT]` `[TAG_KV_DROP_SHIFT]` `[TAG_KV_SHIFT_HADAMARD]`
`[TAG_REUSE_MTMD]` `[TAG_PROMPT_TRUNCATE]` `[TAG_QSA_TRACE]` `[TAG_QSA_VTIER]` `[TAG_QSA_POOLED_CACHE]` `[TAG_QSA_GATHER]`
