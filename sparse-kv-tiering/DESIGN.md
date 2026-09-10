# Sparse-KV Tiering — Design (v0)

**Status:** see [HANDOFF.md](HANDOFF.md) for the live state (2026-09-10). Built, validated and DEPLOYED on
the serving tree: block-level selection, pooled-key cache, gathered attention, chunked scoring, hybrid
middle-range eviction (§12b), and the sliding window — eviction + KV shift via `--cache-reuse`, images
included, plus server-side prompt truncation so a client needs no changes (§12c). That delivers the
"no compaction stall" half of §0 without the disk: dropped context is gone from attention rather than
retrievable, apart from the recurrent trace. The cold tier (§7) is still not built and **Phase 0 has not
been run** — it remains the go/no-go measurement for everything in §7–§9.
**Working name:** sparse-kv-tiering (rename later).
**Author's box:** Strix Halo (Ryzen AI MAX+ 395, gfx1151, 128 GB unified RAM), llama.cpp Vulkan/RADV.
**Verification stance:** every architectural claim below was checked against the actual code or GGUF
metadata on 2026-09-08 (see §13 for the exact commands). If you are building this and something in
the code disagrees with this doc, **trust the code, re-run §13, and fix the doc.**

---

## 0. TL;DR

Long-context agents pay a large, recurring cost when the context fills: the agent framework
summarises old turns to text and re-prefills (llama.cpp: minutes per "compaction"). This design
removes the concept of compaction for a specific class of models by making the **server** manage
context length with a **tiered KV cache**:

- **hot** KV in RAM (only what the model is currently attending to),
- **cold** KV on SSD (everything else, retrievable),
- **garbage-collected** KV (dropped after being cold and unused for a long time),
- **plus** a resident, tiny **index** (the block summary keys) so the model can still *decide*
  which cold blocks it wants,
- **plus**, on hybrid models, a fixed-size **linear-attention state** that carries a lossy summary
  of *everything* regardless of tiering — the safety floor.

The retriever is not bolted on: it is the model's **own** block-sparse attention indexer, which
already scores every block and selects a bounded top-k each step. Context becomes effectively
unbounded at roughly constant resident memory, with no summarisation and no re-prefill.

---

## 1. The problem this solves

Agent frameworks (here: `pi`, but the pattern is universal) "compact" when context nears the
window: they serialise the old conversation to text, ask the model to summarise it, then rebuild the
context as `summary + recent turns`. On llama.cpp with a 125B-A6B model at ~100k context this costs:

| step | measured on the author's box | why |
|---|---|---|
| re-prefill ~78k tokens | ~5 min @ ~250 tok/s | the serialised blob shares no KV prefix, so nothing is reused |
| write a ~5k-token summary | ~3 min @ ~25 tok/s | it is a decode |
| next turn re-prefills the new context | ~1.5 min | new prefix again |
| **total** | **~10 min, blocking** | |

Two facts make it structurally hard to fix in the framework: the summarisation request cannot
reuse the live KV (different prefix), and quality-preserving summaries are long. The fix belongs in
the inference server, where the KV lives. (The *semantic* half — who writes the summary and how memory
is curated — is a separate, complementary idea: see [CURATOR.md](CURATOR.md).)

---

## 2. Core idea (model-generic)

Some architectures contain a **block-sparse attention** mechanism with three properties:

1. **Block summaries.** Tokens are grouped into fixed-size *blocks* (positions `[b·r, (b+1)·r)`),
   and each complete block has one small **pooled summary key** (mean of its members' indexer keys,
   normalised, roped).
2. **A bounded selection.** Each query scores *every* block against the pooled keys, then attends to
   only the **top-k blocks** (`k` is a model hyper-parameter, e.g. 2048), plus an always-visible
   incomplete tail.
3. **Gather-based attention** (in this fork): for decode, attention gathers exactly the selected
   cells' K/V rows and attends over those `n_sel` rows instead of masking all `n_kv`.

Property 1 gives a **cheap resident index** (the pooled keys are ~1/8 the size of the KV they
summarise, see §5). Property 2 bounds the **working set** (only `k` blocks are touched per step).
Property 3 means the attention already reads **by cell index**, so redirecting a cell's K/V to a
different tier is a change at one place, not throughout attention.

Therefore:

- keep **all pooled keys resident** (the index),
- keep only **recently-selected blocks' full K/V resident** (hot tier),
- move everything else's full K/V to **SSD** (cold tier), page it back on selection,
- **GC** blocks that stay cold and unselected past a budget (drop the K/V *and* its index entry),
- and, on hybrid models, leave the **linear-attention (recurrent) state untouched**: it is a fixed-size
  running summary of the entire history and needs no management (it self-decays via gating — it
  cannot be garbage-collected per item, and does not need to be).

What this is *not*: it is not a bolt-on importance estimator (H2O/SnapKV/Quest/InfiniGen estimate
what a dense transformer might attend to). Here the model **emits its exact selection natively**, so
the retrieval signal is free and exact.

---

## 3. Applicability — which models / techniques this works on

### Required model properties
| requirement | why | qwen4exp | dense transformer (Llama, Mistral, Qwen-dense) |
|---|---|---|---|
| block-sparse attention with a **pooled-key indexer** that selects a bounded **top-k** of blocks | the free, exact retrieval signal + bounded working set | ✅ QSA ("DeepSeek lightning indexer" style) | ❌ no native selection → need Quest/InfiniGen-style estimators instead |
| the sparse layers' KV is **addressable by cell** | so a cell's storage can be tiered | ✅ | ✅ |
| (for the *floor*) a **recurrent / linear-attention** component | lossy summary of dropped content | ✅ GDN on 36 of 48 layers | ❌ (tiering still works; no floor) |

### Known instances
- **Qwen3.8-Flash-Next (`qwen4exp`)**, 125B-A6B: the concrete instance below. QSA on 12 layers,
  GDN on 36. This doc's numbers are for it.
- **DeepSeek-V4-family sparse attention (DSA / lightning indexer):** the same indexer design; the
  fork's code literally names the hparams `dsv4_compress_ratios` and cites the "DeepSeek lightning
  indexer" (`qwen4exp.cpp:941`). Not verified on a DeepSeek GGUF — treat as "should transfer, verify".

### ⚠️ Deployment gotcha discovered during verification — READ THIS
The sparse path is **gated per layer** on a GGUF metadata array:

```
src/models/qwen4exp.cpp:1267
const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;
```

On the author's box (verified 2026-09-08, §13 cmd A):
- **prod** `unsloth/Qwen3.8-Flash-Next UD-Q3_K_XL`: `compress_ratios = 4` on layers
  `3,7,11,15,19,23,27,31,35,39,43,47` → **QSA ON**.
- **both Heretic GGUFs** (`spiritfather/…heretic-2 Q4_K_M-MTP` and `…heretic-2 IQ3_M-MTP`):
  `compress_ratios = all 49 zeros` → **QSA OFF**. Those models run **dense** attention on the 12
  full-attention layers. The indexer *weights are present* in the files (52 `blk.N.indexer.*`
  tensors, verified) — only the enabling metadata was dropped by that conversion.

Consequences:
1. This design does **nothing** on a model whose sparse path is disabled. **Pre-flight (§12) must
   confirm `compress_ratios > 0` on the target GGUF**, or patch it.
2. Enabling QSA on the Heretic files is a **metadata patch** (set `4` on those 12 layers; indexer
   tensors already exist; shard 1, which holds all metadata and no tensors, is 10.9 MB). Verified
   recipe in §12. The serving binary (`-main`) **does** carry the base QSA path (`build_qsa_top_k` at
   `qwen4exp.cpp:788`, gate at `:1035`, `llama-memory-hybrid-idx.*` present), so no rebuild is needed
   to turn it on — only the pooled-key cache and the gather kernel are `-new`-only.
3. Every earlier Heretic benchmark on this box (the `-new` vs `-main` "port" comparison, Q4 vs IQ3)
   ran with QSA **off on both sides**, so none of them measured sparse attention or the depth kernels.
   Those conclusions hold for dense attention only. Measured with QSA on in §12a (2026-09-08).

---

## 4. The concrete instance — verified architecture facts (qwen4exp)

Source: GGUF metadata of the Q4_K_M-MTP Heretic file and prod Q3_K_XL (§13 cmd A/B), and the loader.

| fact | value | source |
|---|---|---|
| trunk layers / total blocks | 48 trunk + 1 NextN(MTP) = `block_count 49` | GGUF |
| hidden size | `embedding_length 2560` | GGUF |
| full-attention layers | `(i+1) % full_attention_interval == 0`, interval 4 → **12 layers: 3,7,…,47**; remaining **36 are GDN** (recurrent) | `qwen4exp.cpp:72-80` |
| attention heads | 24 query heads, **2 KV heads** (GQA), head dim **256** (`key_length = value_length = 256`) | GGUF |
| indexer | **4 heads × 128 dims** (`indexer.head_count 4`, `indexer.key_length 128`), **`indexer.top_k = 2048`** | GGUF |
| block size `r` | `compress_ratios[il] = 4` on QSA layers (prod) → **4 tokens per block** | GGUF |
| selection width | `width = min(n_kv, indexer_top_k + r − 1)` = **2051 cells**; gather pads to `n_sel = GGML_PAD(width,256)` = **2304** | `qwen4exp.cpp:985-990,1044` |
| gather activation | decode only (`n_tokens ≤ 16`), `n_kv ≥ 32768` (env `LLAMA_QSA_GATHER`), requires flash-attn | `qwen4exp.cpp:1016-1052` |
| GDN / SSM | `d_conv 4`, `d_inner 6144`, `d_state 128`, `n_group 16`, `dt_rank 48` | GGUF |
| PLE n-gram table | `ple.ngram_size 3`; 50.7 GiB at Q8_0, already served from SSD via `--lazy-mode` (proof SSD-backed sparse tensors work on this box) | GGUF + model card |

---

## 5. Memory model — what grows, what is fixed, what is tiny

All sizes per **single sequence**, derived from §4. KV type assumed **q8_0** (the production
units use `--cache-type-k q8_0 --cache-type-v q8_0`; q8_0 = 34 bytes per 32 values ≈ 1.0625 B/value).

### 5.1 Per-token / per-block on the 12 QSA layers (this is what grows and what gets tiered)
- K row per layer per token: `head_count_kv(2) × 256 = 512` values → 16 q8_0 blocks × 34 B = **544 B**; V same.
- **K+V per layer per token ≈ 1,088 B**; × 12 layers ≈ **12.75 KiB / token**.
- Per 4-token block: ~4.25 KiB/layer, **~51 KiB across the 12 layers** (the unit of tiering).
- Indexer cache (`mem_idx`, one 128-dim key per token per QSA layer): built with the **same
  `type_k`/`type_v` as the attention cache** (`llama-memory-hybrid-idx.cpp:57-60`, MQA with one head
  of `indexer_head_size`), so at q8_0 = 136 B/layer/token ≈ **1.6 KiB/token** for K (check whether an
  unused V is also allocated: `get_v_storage`). Once a block is complete and pooled, its per-token keys
  are only needed to *re*-pool — also a candidate for tiering.

### 5.2 The index — resident, small
- Pooled key: **one f32 row of 128 per complete block per QSA layer** (`llama-memory-hybrid-idx.cpp:68-110`:
  `GGML_TYPE_F32 [idx_dim, pooled_rows]`, `pooled_rows = kv_size/ratio + 2`, allocated in the idx cache's
  buffer type) = 512 B/block/layer → **6 KiB per block across 12 layers = 1.5 KiB/token**. At `-c 114688`
  that is 28,674 rows × 12 layers ≈ 176 MB. It is sized by the **cache size**, not by an index budget (§8.4).
- So the index is **~12 % of the KV it summarises** (1.5 vs 12.75 KiB/token); storing it f16 halves it.

### 5.3 The floor — fixed size, never grows
Per recurrent (GDN) layer, from `llama-hparams.cpp:183-230`:
- conv state `n_embd_r = (d_conv−1)·(d_inner + 2·n_group·d_state) + ple_conv_state` = `3·(6144 + 4096)` = **30,720** floats (+ small PLE term)
- ssm state `n_embd_s = d_state · d_inner = 128 · 6144` = **786,432** floats
- at f32: ≈ 3.0 MiB + 0.12 MiB per layer → **≈ 112 MiB for all 36 GDN layers**, × `(1 + n_rs_seq)` for
  the speculative-rollback ring ("tensors are widened to (1 + n_rs_seq) groups", `llama-memory-recurrent.h:73`;
  `n_rs_seq = params.speculative.need_n_rs_seq()`, `common/common.cpp:1730`, derived from
  `--spec-draft-n-max`; the draft context gets 0; logged at context creation as `n_rs_seq = N`).
  With a draft of 6 this stays well under 1 GB. **Independent of context length.**

### 5.4 Worked example — 1,000,000 tokens, one sequence
| component | size | tier |
|---|---|---|
| QSA K/V (q8_0) | ~12.75 GB | mostly **cold (SSD)**; hot subset bounded by selection |
| indexer per-token keys | ~1.6 GB (q8_0) | tierable after pooling |
| pooled-key index | ~1.5 GB f32 (~0.75 GB f16) | **resident** |
| GDN state | ~112 MiB | **resident, fixed** |
| **per-step working set** (gathered rows) | `n_sel 2304 × 1,088 B × 12` ≈ **29 MiB/step** | must be resident *this step* |

Today, without tiering, that context needs ~16 GB resident just for the sparse layers; with tiering the
resident set is the index + floor + a hot cache of a few hundred MB.

---

## 6. Verified code hooks (branch `qwen4exp-spec-mtp`, worktree `~/LLM/tools/llama.cpp-new`)

> The pooled-key cache and the gathered attention exist **only on this branch**. The `-main` worktree
> (`qwen4exp-main`, the binary the Heretic unit runs) has **neither** (verified: 0 hits for
> `get_pooled_k` / `build_attn_qsa_gather`). Build on `qwen4exp-spec-mtp`.

### 6.1 The three caches (`src/llama-memory-hybrid-idx.h`)
`llama_memory_hybrid_idx : llama_memory_hybrid` owns:
- `mem_attn` — `llama_kv_cache`, the attention **K/V of the 12 QSA layers** (what we tier),
- `mem_recr` — `llama_memory_recurrent`, the **GDN state** (`r_l`/`s_l` per layer; the floor — do not touch),
- `mem_idx` — `llama_kv_cache`, **one indexer key per token**, "same size, padding, stream count and
  slots as the attention cache, so **cell j is the same token in both**" (header comment, lines 13-16).
- pooled-key cache: `pooled_k` (`map<il, ggml_tensor*>`), `pooled_rows` (rows per stream **incl. a
  trailing dustbin row**), per-sequence **watermark** `pooled_w` via `pooled_valid(seq)`; clamp helpers
  `pooled_rm` / `pooled_reset`. Rows below the watermark are valid; rows at/after may be stale garbage
  and are masked by the bias.

### 6.2 The indexer + pooled cache write path — `src/models/qwen4exp.cpp:789-1011 build_qsa_top_k`
- writes the raw indexer key for new tokens into `mem_idx` (`mctx_idx->cpy_k`, line 862),
- **dirty path** (lines 870-907): gathers the members of newly-complete blocks (`dirty_cells`),
  mean-pools (`Σ/r`), RMS-norms with `index_k_norm`, ropes with `dirty_pos`, and `ggml_set_rows` them
  into the pooled store at `dirty_rows`; then reads the **whole store** as `pooled [idx_dim, n_blocks]`,
- scores `q · pooled` per head → `relu` → sum over the 4 heads → per-block score, expanded to
  per-cell via `cell_blk` and offset by `bias` (lines 946-982),
- `ggml_top_k(expanded, width)` (radix on Vulkan), padded to `n_sel`; returns `top_k [n_sel, n_tps, 1, ns]`
  of **cell indices** (line 997-1010).

### 6.3 The fetch hook — `qwen4exp.cpp:1062-1145 build_attn_qsa_gather`
```
k_rows = view of K cache as [d_k*hkv, n_kv, ns]     // "a cell is one contiguous row of d*hkv values"
k_sel  = ggml_cast(ggml_get_rows(k_rows, idx), F16)  // idx = top_k cell indices, flattened   (line 1097)
v_sel  = ggml_cast(ggml_get_rows(v_rows, idx), F16)  //                                      (line 1098)
...flash_attn_ext over the n_sel gathered rows, mask gathered per selected cell (1113-1127)
```
**This is the single place attention reads K/V by cell.** Tiering = making the rows named by `idx`
resident before this node runs (§8.3). The gathered rows are dequantised to F16 right here, so the cold
tier may store rows in any format (the original q8_0 cache bytes are the obvious choice). The dense masked path (`build_attn_qsa`, line 1149) is used
for prefill and below the gather threshold and reads the whole cache; tiering targets the decode/gather
path first.

### 6.4 Host-side block map and visibility — `src/llama-memory-hybrid-idx.cpp:744-905 set_input_qsa`
- block of a cell = `pos / r` (line 817): **blocks cut the position line, not the cell array**;
- a block is scorable only if all `r` cells are present (`filled[b] == r`), else its cells get `-inf`;
- bias (lines 897-903): `-inf` if the cell is **empty**, belongs to another seq, or is in the future;
  `1e9` for the incomplete tail (always visible); `0` for complete scored cells.
  → **an evicted (freed) cell is automatically invisible and unscored** — no extra masking needed;
- pooled-cache bookkeeping (847-885): `n_complete` = index of the *last* complete block + 1;
  dirty range = `[watermark, n_complete)`; watermark advances **before** compute.
  → this assumes valid rows form a **contiguous prefix**. Mid-range eviction creates holes below the
  watermark; today's code survives them (a re-pooled hole reads empty cells → garbage row → masked by
  `-inf`) but wastes work. **§8.2 replaces the watermark with a per-block validity bitmap.**

### 6.5 ⚠️ `seq_rm` cannot be used for eviction — verified
`llama_memory_hybrid_idx::seq_rm` (`hybrid-idx.cpp:248`) calls **`get_mem_recr()->seq_rm` first and
aborts if it refuses**. `llama_memory_recurrent::seq_rm` (`llama-memory-recurrent.cpp:150-200`) only
permits a **tail rollback of ≤ `n_rs_seq` tokens** ("models like Mamba can't have a state partially
erased"); any other partial range returns `false`, so nothing is removed from any cache.
**Eviction must bypass the recurrent cache**: operate on `mem_attn` + `mem_idx` + the pooled bitmap
directly, leaving `mem_recr` untouched. This is exactly the desired semantics (the GDN state must
*not* be rewound), but it means a **new method**, not `seq_rm`.

### 6.6 A related precedent already in the server
llama-server keeps a host-RAM **prompt cache** of whole-KV snapshots and evicts them under pressure
(observed log: "making room for prompt cache entry, removing oldest entry (size = 5109.695 MiB)").
This design is the block-granular, on-SSD analogue driven by the model's selection instead of by
prompt identity.

---

## 7. Design

### 7.1 Tiers and the index
```
                   ┌────────────── resident ──────────────┐
  pooled-key index │ all blocks, ~1.5 KiB/token (§5.2)    │  scored every step
  GDN state        │ fixed ~112 MiB (§5.3)                │  never touched, self-decays
  HOT K/V          │ LRU of recently-selected blocks      │  attention reads from here
                   └──────────────────────────────────────┘
  COLD K/V         SSD store, keyed (seq, block_id) → 51 KiB (12 layers) [+ indexer keys]
  GC'D             dropped: K/V deleted from SSD AND its pooled row invalidated
```
Invariants:
- **I1** Every block in the index is either hot or cold (never GC'd). GC removes the index entry too.
- **I2** A selected block is hot before `build_attn_qsa_gather` runs for that step (or is deferred, §8.3).
- **I3** `mem_recr` is never modified by tiering. Only `seq_rm`-class framework operations touch it.
- **I4** Positions are never reused: evicting cells frees slots but new tokens get new, larger positions.

### 7.2 Eviction policy (what becomes cold)
Score each complete block by the indexer's own decisions: maintain a per-block **selection counter**
and **last-selected step**, updated from the `top_k` of each decode step (cell → block via `pos / r`).
Keep hot: attention-sink blocks (first few), the recent window (last `W` blocks), and the top-`H` by
selection frequency; evict the rest to cold when the hot budget is exceeded. `W`, `H`, budgets are knobs.

### 7.3 GC policy (what is dropped for good)
Cold blocks not selected within the last `G` steps *and* beyond the disk budget are deleted (K/V from
SSD) and their pooled row marked invalid (bitmap). This is the only lossy tier. The GDN state still
carries a faded trace of them (§2). GC is ordinary cache management; nothing special.

### 7.4 What is deliberately out of scope for v0
- Multi-stream / multi-sequence (`n_stream > 1`): the pooled cache is **single-stream only**
  (`set_input_qsa` line 848 asserts it). v0 = one sequence.
- Prefill over cold blocks: v0 tiers the **decode/gather** path; prefill (`build_attn_qsa`, dense
  masked) keeps today's behaviour and requires the touched range resident.
- Unbounded *positions*: see §8.4 — the current block map assumes `pos < n_kv`.

---

## 8. Mechanics

### 8.1 Decode step (target behaviour)
1. Host: build `cell_blk`/`bias`/dirty tables as today (`set_input_qsa`), using the **bitmap** for
   validity instead of the watermark.
2. GPU: `build_qsa_top_k` scores all blocks from the resident pooled store → `top_k` cell indices.
3. **Materialise**: ensure every selected block is hot (§8.3), then `build_attn_qsa_gather` gathers.
4. Host, after the step: update selection counters from `top_k`; if hot budget exceeded → evict (§7.2);
   periodically GC (§7.3).

### 8.2 Eviction = bypass-`seq_rm` removal + bitmap
New method on `llama_memory_hybrid_idx` (name TBD, e.g. `tier_evict(seq, block_id)`):
- copy the block's `r` cells' K/V rows from **all 12 QSA layers** (and its indexer keys) to the cold
  store, keyed `(seq, block_id, layer)`;
- free those cells in `mem_attn` **and** `mem_idx` **directly** (the underlying `llama_kv_cache` cell
  ops — not the hybrid `seq_rm`, §6.5);
- mark the block's pooled row **valid-but-cold** in the bitmap (still scored: that is how it gets
  re-selected). The pooled row itself stays resident — it *is* the index.
- **Do not touch `mem_recr`.**
The freed cells are automatically `-inf` in the bias (§6.4), so correctness holds immediately.

### 8.3 Fetch — the key design choice
The selection is computed **on the GPU inside the same graph** as the gather, so a cold block cannot
be fetched "in between" without a host sync. Three options, in order of preference for v0:
- **(c) One-step-late materialisation (v0 default):** gather this step from resident rows only
  (cold selections are masked `-inf` for this step); after the step, read back `top_k`, page the newly
  selected cold blocks into hot (allocate cells, `pos_set` to their *original* positions, write rows,
  flip bitmap to hot). They are attendable from the next step. Cheap and simple; correct whenever
  selection is temporally stable — which **Phase 0 measures** (§9).
- **(b) Prefetch:** predict next step's selection from this step's (same stability assumption) and
  page asynchronously so (c)'s one-step lag mostly disappears. InfiniGen's approach.
- **(a) Exact two-pass:** run the indexer scoring as a small first graph, sync `top_k` to host, fetch,
  then run attention. Exact but adds a GPU↔host sync per token; fallback if (c)/(b) prove lossy.

### 8.4 Beyond `-c`: positions vs cells
Today `n_blocks = (n_kv + r − 1)/r` and `blk_of = pos/r` with `if (b >= n_blocks) continue;`
(`set_input_qsa` 765, 817-821): **the block map assumes `pos < n_kv` (the cache size)**. Blocks with
`pos ≥ n_kv` would be silently unscored. True unbounded context therefore needs the block map keyed by a
**per-sequence logical block id** (a small host-side table `block_id → {pos, tier, cell slots, pooled row}`)
rather than `pos/r`, and the pooled store sized by the *index* budget, not by `n_kv`. This is the
largest structural change and belongs to Phase 3; Phases 0-2 work within `-c`.

---

## 9. Phased plan with acceptance criteria

| phase | work | proves / delivers | accept when |
|---|---|---|---|
| **0 — measure the premise** | add logging to `build_qsa_top_k`/host: per-block selection frequency and **step-to-step overlap** of the selected set, at deep context (≥ 64k). *No cache surgery.* | whether selection is **concentrated** (evictable middle) and **stable** (cheap fetch, option (c) viable) | you can state: "% of blocks never selected", "median step-to-step overlap of the 2048-set" |
| **1 — bitmap + counters** | replace watermark with per-block validity bitmap; add selection counters | correct handling of holes; the data for policies | greedy oracle identical to today with no eviction |
| **2 — evict-to-cold + one-step-late fetch (within `-c`)** | `tier_evict`, SSD store, hot LRU, option (c) fetch | the whole loop working, bounded RAM | needle-at-depth ≥ target recall; cold fetches/step within budget |
| **3 — unbounded positions** | logical block ids (§8.4), disable framework compaction | context never fills | a 1M-token session runs at ~constant RAM |
| **4 — prefetch / polish** | option (b), GC tuning, multi-stream if wanted | speed | per-step latency ≈ today's |

**Phase 0 is cheap and decides everything.** If selection is diffuse or churny, stop: fall back to
lossy eviction only, or abandon.

---

## 10. Validation

- **Correctness floor:** greedy oracle — temp-0 output with tiering enabled vs disabled must match
  byte-for-byte (or a single benign late flip, the fork's established noise floor) *while nothing has
  been evicted*; any divergence before eviction is a bug, not a quality trade-off.
- **Quality vs aggressiveness:** needle-at-depth — plant facts early, run past the eviction budget, ask
  for them; plot recall vs hot budget and vs GC horizon. This curve **is** the research result.
- **Perf:** cold blocks fetched per decode step (target: single digits, amortised), per-step latency vs
  today, and RSS/GTT vs context length (must be ~flat past the hot budget).
- **Selection stability (Phase 0):** Jaccard overlap of consecutive `top_k` block sets; fraction of
  blocks with selection count 0 after N steps.

---

## 11. Risks and open questions

1. **Selection churn** — if the indexer picks different distant blocks every token, option (c) is
   lossy and SSD-bound. Phase 0 answers this before anything is built.
2. **Unified-memory GTT** — on this box the GPU maps system RAM (GTT, ~120 GB ceiling, pinned,
   non-swappable). Hot K/V must stay well under the ceiling; a hard-hang, not an OOM, is the failure
   mode. Keep the hot budget conservative and consider an amdgpu GTT cap.
3. **Quality of the floor** — GDN state is lossy and decays; evicted specifics are only recoverable via
   re-selection from cold (option c) — GC'd content is gone. The needle test bounds this.
4. **Prefill path** — dense masked attention over cold ranges is undefined in v0; a long prompt that
   references GC'd/cold history must either re-materialise the range or accept masking.
5. **Multi-stream** — pooled cache is single-stream; v0 is one sequence.
6. **Speculative decoding (MTP)** — the recurrent rollback ring and the pooled watermark already
   cooperate with spec rollbacks (`seq_rm` clamps); tiering must keep the bitmap consistent under a
   rollback (a rolled-back block returns to "incomplete"). Test with `--spec-type draft-mtp` on.
7. **The MTP head's own indexer** — block 48 also carries indexer tensors (verified); the draft context
   runs dense (the fork comments say so). Tiering must not assume the draft context has an index.

---

## 12. Pre-flight for the author's deployment (do these first)

1. **Confirm which GGUF you are on and whether QSA is enabled** (§13 cmd A). The Heretic files are
   **OFF**. Enabling is a 12-int patch of `qwen4exp.attention.compress_ratios` (array of INT32, 49
   elements) in **shard 1 only** (10.9 MB, all metadata, no tensors). Verified recipe (tested
   2026-09-08 on a scratch copy: exactly 12 bytes differ; re-read shows the 12 QSA layers):
   ```bash
   # never patch a file a live server is mmapping: sibling dir, hardlink shards 2-6, copy + patch shard 1
   D=/home/jocke/LLM/models/Qwen3.8-flash-next-heretic-Q4K_M-MTP/Q4_K_M; mkdir -p $D/MTP-qsa
   for n in 2 3 4 5 6; do ln -f $D/MTP/*-0000$n-of-00006.gguf $D/MTP-qsa/; done
   cp $D/MTP/*-00001-of-00006.gguf $D/MTP-qsa/
   cd ~/LLM/tools/llama.cpp-main/gguf-py && PYTHONPATH=. ~/.pyenv/versions/comfy-rocm/bin/python - $D/MTP-qsa/*-00001-of-00006.gguf <<'EOF'
   import sys, types; sys.modules.setdefault("yaml", types.ModuleType("yaml"))
   from gguf import GGUFReader; import numpy as np
   r = GGUFReader(sys.argv[1], 'r+'); f = r.fields["qwen4exp.attention.compress_ratios"]
   assert f.types[1].name == "INT32" and len(f.data) == 49
   for i in [3,7,11,15,19,23,27,31,35,39,43,47]: f.parts[f.data[i]][0] = np.int32(4)   # trunk only; block 48 (MTP) stays 0
   r.data.flush(); print([int(f.parts[i][0]) for i in f.data])
   EOF
   ```
   **This dir already exists** (`…/Q4_K_M/MTP-qsa/`, created 2026-09-08; the 8096 unit serves it since
   18:36 the same day; see its README.txt). A slot snapshot taken with QSA **off** has an unwritten
   indexer cache and must not be restored into a QSA-on server (the old autosave was set aside as
   `autosave.bin.dense-qsa-off-2026-09-08`). Then **benchmark** on the same binary with `-m …/MTP-qsa/…-00001-of-00006.gguf`:
   greedy oracle vs the dense run (expect only a benign-flip-level difference: dense ⊇ sparse, and sparse
   is how the model was trained) and deep decode/prefill at 64k+. Expectations differ by binary (§12a).
   Worth doing **regardless** of tiering.
2. **Build on `qwen4exp-spec-mtp`** (`~/LLM/tools/llama.cpp-new`). Note the earlier finding: that
   tree's older Vulkan base is slower for **IQ3/Q4** quants than the rebased `-main` at *shallow*
   depth (measured 2.7× decode). For prod's Q3_K_XL it is the fast path. Tiering is orthogonal to this;
   pick the model/tree pair accordingly (prod Q3_K_XL on `-new` is the least-friction dev target).
3. **One heavy GPU job at a time** while developing (the box hard-hung at ~8 GB free).

### 12a. What enabling QSA buys — by binary (expectation; measure it)
Decode at 100k on the 12 dense layers reads the whole K/V every token: `12 × 100k × 1,088 B ≈ 1.3 GB/token`,
against roughly 3.4 GB of active expert weights per token at Q4 — about +40 % memory traffic per token,
growing linearly with context. With QSA the layer needs only the selected rows (~29 MB/step, §5.4).
- **`-main` (the serving binary today):** base QSA path only. It recomputes the pooled keys every step
  (cheap, 128-dim) and runs attention **masked over all `n_kv`** (`build_attn_qsa`, no gather), so the
  traffic saving is realised only if the Vulkan flash-attention kernel skips fully-masked tiles. Expect a
  modest decode gain of unknown size, plus the trained attention pattern restored. Prefill is `O(n²)` on
  those layers either way unless masked tiles are skipped.
- **`-new` (`qwen4exp-spec-mtp`) or the `llama.cpp-heretic` worktree (`-new` kernels + the MTP alias
  loader, `build-heretic/`):** pooled-key cache + **gathered attention** at `n_kv ≥ 32768`, so the
  per-token K/V read collapses to the selected rows. This is where the 1.3 GB/token goes away. Caveat:
  that tree's older Vulkan base measured 2.7× slower for **IQ3_M decode at shallow depth**; Q4_K_M was
  never measured there, and no run so far had QSA on. The decisive experiment is Q4_K_M on
  `build-heretic` vs `-main`, both on `MTP-qsa`, decode at ≥ 64k (the existing
  `~/LLM/bench/heretic_q4_vs_iq3.sh` harness with the model path swapped).
- **GDN is not a switch:** 36 of 48 layers are linear attention by architecture and always run that
  way. Only the 12 full-attention layers have the sparse/dense choice, and it is decided by this metadata.

**Measured 2026-09-08** (`~/LLM/bench/heretic_qsa_bench.sh`; Heretic Q4_K_M-MTP, one ~56.6k-token
synthetic prompt, temp 0, MTP draft on, q8_0 KV, `-c 65536`, one server at a time, ComfyUI idle):

| cfg | binary / model | prefill @56.6k | decode @56.6k | MTP accept | verbatim recall |
|---|---|---|---|---|---|
| A | `-main` / QSA off (production today) | 295 tok/s | 19.4 t/s | 0.906 | 4/4 |
| B | `-main` / QSA on (`MTP-qsa`) | 283 tok/s | **25.9 t/s (+34 %)** | 0.865 | 4/4 |
| C | `build-heretic` (fork kernels) / QSA on | 228 tok/s | 20.6 t/s | 0.847 | 4/4 |
| D | `build-heretic` (fork kernels) / QSA off | 252 tok/s | 19.7 t/s | 0.841 | 4/4 |

Greedy oracle: all four quote the four requested facts verbatim; outputs diverge only inside the
free-form summary (A vs B at char 479, B vs C at 520). (An earlier reference of "39.9 t/s at 16k" for
this binary was wrong: that number came from a ~100-token prompt with `-c 16384`, QSA off. The real
shallow-vs-deep curve is in the draft-cost table below.)
Conclusions: (1) the metadata patch works and QSA-on is a clear net win on the serving binary at
depth, quality intact; (2) the fork's depth kernels do **not** win on Q4_K_M — QSA-on gains only
+5 % there (C vs D) and the tree prefills 15 % slower (D vs A), while its decode is on par for this
quant (D ≈ A; the 2.7× penalty seen earlier was IQ3_M-specific); (3) the gather path **does engage** — verified by a one-time
stderr announcement added to `qsa_gather_n_sel` in the `llama.cpp-heretic` worktree (uncommitted):
`qsa: gather engaged (n_kv = 56832, n_tokens = 4, width = 2051, n_sel = 2304, min_kv = 32768)` —
but buys only **+3 % decode** on this quant (rerun: default 22.0 t/s vs `LLAMA_QSA_GATHER=0`
21.3 t/s; prefill 232 vs 235 tok/s, unaffected as expected; recall 4/4 both). The dense masked path
is already cheap here, so for this design the gather is the **hook**, not a speedup; (4) the
remaining depth loss with QSA on is therefore not the target's attention-KV reads.

**Draft-cost experiment (2026-09-08, `~/LLM/bench/heretic_draft_cost.sh`; serving binary + `MTP-qsa`,
q8_0 KV, temp 0; the MTP draft block attends densely by design — `qwen4exp.cpp:522-525` "v1
simplification", plain `build_attn`, and the draft context's memory is a plain KV cache with no indexer
cache, so a `compress_ratios[48]` metadata patch is inert):**

| run | prefill | decode | MTP accept | mean accepted/step | recall |
|---|---|---|---|---|---|
| 14k, no spec | 390 tok/s | 21.0 t/s | – | – | 4/4 |
| 14k, MTP n_max 6 | 349 tok/s | 30.1 t/s | 0.882 | 1.75 | 4/4 |
| 56k, no spec | 302 tok/s | 16.4 t/s | – | – | 4/4 |
| 56k, MTP n_max 3 | 279 tok/s | 24.8 t/s | 0.921 | 1.54 | 4/4 |
| 56k, MTP n_max 6 (prod) | 274 tok/s | **27.0 t/s** | 0.886 | 2.26 | 4/4 |

Reading: (a) **the draft's dense attention is not the depth cost** — with speculation the loss from 14k to
56k is −10 %, without it −22 %, and shortening the draft makes things worse (wasted draft steps are
cheap: ~62 MB of KV per drafted token ≈ 0.3 ms). Hypothesis rejected; do not wire QSA into the draft
for speed. (b) The target alone still pays **+13 ms per step** at 56k vs 14k (47.6 → 61.0 ms) although
its attention reads are bounded, so the cost is **per-step work that scales with `n_kv`** in the base
QSA path of the serving tree: the masked flash-attention still spanning all `n_kv` (no gather), the
per-step re-pooling of all block keys (no pooled cache), the top-k over `n_kv` per-cell scores (a
dedicated `GGML_OP_TOP_K` Vulkan op in both trees — its scaling with `n_kv` is unmeasured), and the
host-side mask/bias build and upload. Which
dominates needs a per-op profile (e.g. the Vulkan perf logger on a short decode at 56k), not guessing.
(c) Speculation is worth more at depth (+65 % at 56k vs +43 % at 14k) because it amortises that
per-step overhead over ~3.3 tokens. (d) Prefill loses 23 % from 14k to 56k: prefill attention is
dense-masked over `n_kv` on both trees, and the MTP draft context prefills its layer too (−9 %).

**Per-op GPU profile (2026-09-08, `~/LLM/bench/heretic_qsa_profile.sh`; serving binary + `MTP-qsa`,
target-only decode, Vulkan perf logger `GGML_VK_PERF_LOGGER=1`, ~210 decode graphs averaged per depth;
full tables in `heretic_qsa_profile_results.md`):**

| | 14k | 56k | Δ |
|---|---|---|---|
| GPU op time per step | 44.9 ms | 59.1 ms | +14.2 ms |
| wall per token | 51.9 ms | 65.6 ms | +13.7 ms |
| host + sync (wall − GPU) | 7.0 ms | 6.5 ms | ≈ 0 |

The depth cost is entirely GPU-side and it is **QSA bookkeeping that scales with `n_kv`**, in three
buckets (ms per step, all 12 QSA layers; the ~40 ms of expert/dense matmuls, LM head and norms are flat):

| bucket | ops | 14k | 56k | Δ | known fix |
|---|---|---|---|---|---|
| **(B) selection over cells**: expand block scores to every cell, add the per-cell bias, top-k over `n_kv` | MULTI_ADD 3.51, ADD 3.15, TOPK_QSA GET_ROWS 1.90, TOP_K 0.66, SCALE 0.75 | 2.3 | 10.0 | **+7.7** | **block-granular selection** — score, bias and top-k over `n_blocks` (¼ the elements, no per-cell expansion), expand only the ~512 selected blocks to cells. New work, self-contained in `build_qsa_top_k` + `set_input_qsa`. |
| **(A) re-pooling every block key every step** (no pooled cache) | GET_ROWS 3.75, RMS_NORM(128,n_blocks) 1.76, ROPE 1.18 | 2.9 | 6.7 | **+3.8** | pooled-key cache (exists in the fork tree: only dirty blocks are pooled) |
| **(C) masked flash-attention over all `n_kv`** (no gather) | FLASH_ATTN_EXT 3.14 | 1.3 | 3.1 | **+1.9** | gathered attention (exists in the fork tree) |

Implications: (1) extrapolated linearly the bookkeeping (~20 ms at 56k) is ~90 ms at 256k → ~7 t/s
target-only, so **on the serving tree the wall for 256k+ is this bookkeeping, not attention volume**;
(2) every bucket has a known fix; with all three the `n_kv`-dependent work shrinks to the block-score
matmul + top-k over blocks + the gather ≈ 1–2 ms at 56k and ~5 ms at 256k, i.e. **decode at 256k ≈ decode
at 14k**; (3) order of work on the serving tree: **(B) first** (largest, new, no port), then port (A) and (C)
from the fork; (4) the fork already has (A)+(C) but its older Vulkan base costs more on the flat 40 ms
than they save, which is why config C measured below B; (5) bucket (B) is the same per-block change §8.4
already requires for the tiering design, so it is on the path regardless; the eventual wall past ~1M
tokens is the block-score scan itself (see the "stage 3" discussion), which needs a hierarchical index.
Side note: the LM head (`MUL_MAT_VEC q8_0 m=248320`) reads ~0.6 GB per token ≈ 3 ms/step at every depth.

### 12b. Implementation on the serving tree (2026-09-08 evening)

Worktree `~/LLM/tools/llama.cpp-qsab`, branch `qwen4exp-qsa-block` (from `qwen4exp-main`), own build dir.
Order per §12a: (B) → (A) → (C). Validation per change, scripts in `~/LLM/bench/heretic_qsa{b,a,c}_validate.sh`:
greedy oracle at 14k (the serving binary is deterministic there; at 56k it is **not**: two identical runs
diverge at char ~500–800), verbatim recall at 56k, per-op Vulkan profile, and 64k perplexity on real code
(`ppl_corpus.txt` = ggml.c + ggml-vulkan.cpp + llama-context.cpp, one chunk; old binary 1.1151 ± 0.0043).

**(B) block-granular selection.** `build_qsa_top_k`: top-k over the block scores (`k_blk = ⌈2051/4⌉ = 513`),
winners expanded to cells through `blk_cells` (I32 `get_rows`, a device op on Vulkan), the `cell_blk` input
dropped in this mode; `build_attn_qsa`: the selected rows take their original mask values via `get_rows`
instead of a zero write plus an `n_kv`-wide add. Host: the spare block's member row holds the real tail
cells, padded by repeats (harmless on the masked path: rows are unmasked, not gathered).
**Bug caught by the perplexity check:** the old per-block bias gave every block from the query's tail
onward the "always visible" value and relied on the per-cell mask being added *before* the top-k. With
the top-k over blocks that mask comes after, so in prefill ubatches up to 512 future blocks won selection
slots for the early queries (PPL +0.63 %, 14k oracle diverged at char 424). Fix: blocks that start past the
query get −inf in the per-block bias, the spare block only while its cells are visible. After the fix the
14k oracle is **byte-identical** to the serving binary and PPL is 1.1154 (within the error).

**(A) pooled-key cache.** Ported from the fork, adapted to this tree's compact block ids: rows indexed by
block id (= position block for one sequence without holes, verified per ubatch with a refill on mismatch),
watermark per sequence in blocks, dirty range = watermark → last complete block, capacity from
`qsa_pooled_n_dirty_max` at graph build (1 in steady decode, every block after a state load), single
stream and single sequence only (a second sequence in the stream returns capacity 0 → masked path),
kill switch `LLAMA_QSA_NO_POOLED_CACHE=1`. Output identical to the cache-off arm and to the serving
binary at 14k, and identical to the serving binary's speculative run at 56k; PPL 1.1154 (on) / 1.1149
(off) vs 1.1151 (old), all within the error; a slot save/erase/restore of 56,855 tokens round-trips; the refill after a restore was exercised with a raw-text
prompt (`heretic_qsa_refill_check.sh`: prefill, save, erase, restore, extend with a prefix hit of 56,567
tokens → the extension's first ubatch refilled all ~14k blocks in one graph and the model completed
"Fact 42:" with the correct text). Note: re-sending the *same* chat prompt, or a chat history whose
assistant turn re-renders differently (the template inserts an empty think block), trims the generated
tail — a rollback larger than `n_rs_seq` that the recurrent cache refuses — so the server clears and
re-prefills; pi's own turns hit the cache (journal: 294 new tokens after a 108k restore).

| 56k, target-only, GPU ms per decode step | |
|---|---|
| serving binary (QSA on) | 59.1 |
| + (B) | 56.1 |
| + (A) | **43.7** (same-session cache-off arm: 57.9) |
| old binary at 14k, for reference | 44.9 |

Removed per step: MULTI_ADD 3.5→0.4, ADD 3.1→0.4, GET_ROWS 3.8→0.9, RMS_NORM(128,n_blocks) 1.3→0,
ROPE 1.2→0.1, TOPK_QSA GET_ROWS 1.9→0. Still `n_kv`-dependent: flash-attention over all cells 2.4 ms
(→ (C)) and the block scoring + top-k ≈ 0.5 ms.

**Wall-clock caveat for this batch:** it ran with the n-gram table's pages evicted (a dozen 80 GB model
reloads plus the corpus), 40–200 major page faults per second during decode → +13 ms per token on
*every* arm (the table is `--lazy-mode` SSD-backed, and the disk is dm-crypt). Only same-session pairs are
comparable (cache off 78.0 → on 63.5 ms/token). In production the pages warm within minutes and stay.
Warm number, speculation on, 56k: **28.8 t/s** (old 27.0). The 26 % target-step saving becomes +7 %
with speculation because a step is one 7-query verify pass plus six draft passes; the verify pass's
context-dependent cost is the flash-attention over all cells for 7 queries, which is what (C) replaces.
Expect low 30s at 56k after (C); the shallow-context ceiling for this quant is ~40 t/s (weight bytes).

**(C) gathered attention.** Ported from the fork and adapted to the block-level selection: on the gather
path the spare block is kept out of the top-k (a repeated row would be attended twice) and each query's
visible tail cells arrive as their own rows with a per-slot visibility mask from set_input; block rows
take their original mask values via `get_rows`; pads to the 256-row granularity are masked. Env
`LLAMA_QSA_GATHER` (0 = off, N = per-query n_kv threshold, default 32768, 1 = force). Verified: at 14k
with the gather forced the output is **byte-identical** to the masked path and to the serving binary;
at 56k the profile shows the flash-attention shape switch (56,832 rows → 2,304) and GPU time 44.1 →
42.5 ms/step target-only (the FA saving of 2.75 ms minus ~1 ms of row gather, F16 casts and mask concat).
**Measured limit:** the gather is per query, so a 7-query speculative verify batch gathers and casts
seven selections and *loses* at 56k (19.7 vs 21.9 t/s). The threshold therefore scales with the batch,
`n_kv ≥ 32768 × n_tokens`: single-token decode gathers from 32k, a 7-token verify batch only from
~230k. Confirmed: the speculative 56k run is back at the masked speed (27.3 t/s with the perf logger
on; only the 1-query steps gathered, 21 of ~380 graphs). Follow-up if the depth curve at 200k+ needs it: gather the *union* of the batch's selections
once and mask per query (needs a GPU unique/dedup), or a fused quant→F16 gather kernel.

**Deployed 2026-09-08 22:13** on the 8096 unit: the `qsab` binary, `-c 262144` (the model's native
window), `-ub 512`. At `-ub 2048` the context creation failed: the *prefill compute scratch* is
19.2 GB at 256k (it scales with `n_ctx × n_ubatch`: the `[n_kv, n_ubatch]` masks and the
`[n_blocks, 4, n_ubatch]` indexer scores) and the allocator's 5 GiB chunk exceeded the Vulkan device
buffer limit; at `-ub 512` it is ~5 GB and headroom is better than before (19 GiB available). The ubatch trades
prefill throughput for headroom: 14k prefill 255 tok/s at 512, 300 at 1024 (12–13 GiB left, too tight
next to ComfyUI's 16 GB reserve), ~370 at 2048 (does not allocate at 256k). Deployed with 512.
**Scratch bound, step 1 (2026-09-08 22:44):** the indexer scoring is now done in chunks of queries
inside the graph (`LLAMA_QSA_SCORE_CHUNK`, default 512): the `[n_blocks × 4 × n_ubatch]` score
tensors shrink to one chunk's worth. Validated: 14k output byte-identical to the serving binary,
PPL 1.1152, prefill 365 tok/s at ubatch 2048 (14k). Not sufficient alone: at 256k with 2048 the
context creation still fails on a 6.5 GiB request (the device's `maxBufferSize` is 4 GiB), which
by its size looked like the chunks' score tensors packed together. **Probes (23:00):** chunk 256 and
128 fail on the same 5.4 GiB request (so it is not the scores), and with the backend allowed 4 GiB
blocks (`GGML_VK_SUBALLOCATION_BLOCK_SIZE`) it fails differently — *out of device memory* at a 23 GB
compute buffer. Conclusion: at 256k × 2048 the prefill scratch of the remaining graph (the two
`[n_kv, n_ubatch]` F16 masks, the flash-attention temporaries, the bias input, the draft context's
own mask) is ~20 GB, which does not fit next to the 80 GB model on the 120 GB GTT whatever the
chunking. The scores were the part this tree owned; the rest is core mask/FA scratch. Working
configurations at 256k: ubatch 512 (scratch ~5 GB, 19 GiB headroom, 255 tok/s prefill at 14k) or
1024 (~10 GB, 12–13 GiB headroom, 300 tok/s). Deployed with 512; 1024 is the option when ComfyUI is
not rendering. A ubatch of 2048 needs a smaller context (~114k, the previous setting).
Resident state at 256k is only ~5 GB (K/V 3.4, indexer 0.4, pooled 0.4, draft 0.3, recurrent 0.1),
which is why memory is not the motivation for tiering below the model's window; and note the design's
prefill still uses the masked path (§7.4), so the scratch bound needs its own treatment (a small
ubatch, or block-sparse prefill) regardless of tiering. pi's `contextWindow` raised to 262144 →
autocompaction at ~245,760.

**Draft sparse path (asked, deferred):** possible — a hybrid-indexed memory without recurrent layers for
the MTP context, the three QSA calls in `graph_mtp`, a metadata patch for block 48. Crossover ≈ 150–200k:
the dense draft costs ~1.1 KB × n_kv per drafted token (0.3 ms @56k, 1.4 ms @256k) vs ~0.4 ms fixed
for the sparse machinery, ×6 drafts per step → a wash at 56k, ~5 % of a speculative step at 256k.
Acceptance-rate effect unknown.

### 12c. The sliding window: eviction + KV shift on a hybrid model (2026-09-09)

The cheap half of the tiering design, shipped on its own: instead of re-processing the conversation every
time the client drops old turns, the server evicts the dropped range and **shifts** the rest. Measured on
the serving model, 70k-token prompt of real code with four needle facts inserted at 70–94 %, the client
resending the same text minus its first third:

| | prompt tokens processed | prompt time | recall |
|---|---|---|---|
| slid (evict 23.5k, shift 47k) | 71 | 1.7 s | 4/4, byte-identical to the control |
| control, full prefill | 47,075 | 218.4 s | 4/4 |

Server flags: `--cache-reuse N` (find the kept tail in the cache and shift it) and `--cache-ram 0` (the
host prompt cache otherwise clears the slot on a zero-prefix request, leaving nothing to reuse).

What a hybrid (attention + recurrent) model needs beyond stock llama.cpp — all generic, all in the memory
layer, none of it model-specific:
1. **Middle/head eviction must not be refused.** A recurrent cache rejects any partial `seq_rm` that is
   not a bounded tail rollback. But dropping a *prefix or middle* range needs no rewind: the state is a
   running summary through the end of the sequence and stays valid, keeping a faded trace of the dropped
   text. Evict from the attention and indexer caches only (`[TAG_HYBRID_MID_RM]`).
2. **The recurrent state's position must follow the shift.** Its position is just "the last position of
   the sequence", and the stock `seq_add` moves it only if it lies inside the shifted range — which a
   prompt-cache shift never covers, because it stops at the last matched token. Left behind, the sequence
   reports a `pos_min` far past the shifted cells, and the server (which takes a hybrid's `pos_min` as the
   max over both memories) concludes the cache cannot serve the resume point and re-processes everything.
   Move it to the end of the renumbered chunk (`[TAG_HYBRID_TAIL_SHIFT]`).
3. **Caches whose keys are stored before RoPE must not be rotated by the shift graph** (here the QSA
   indexer): update their positions, drop the pending K-shift (`[TAG_KV_DROP_SHIFT]`), and invalidate any
   position-derived cache built on top (the pooled block keys).
4. **M-RoPE**: `get_can_shift()` refuses when `n_pos_per_embd() > 1`. For text every position component
   equals the token position, so a uniform shift is exact (the shift graph already rotates the whole
   vector with NEOX ordering, which is identical to M-RoPE/IMRoPE at equal positions — verified in
   `ggml_mrope_cache_init` and `rotate_pairs`). Cells also carry an M-RoPE extent that must shift with
   the position. Only 2-D image tokens would be positioned wrongly by a shift.

**A KV-shift bug this uncovered, relevant to any llama.cpp fork that rotates the KV cache**: with a
quantized cache this tree stores K in a Walsh-Hadamard-rotated basis (better quantization), so the shift
must dequantize, undo the rotation, rope, redo it, requantize. The shift graph passed a view of only the
rotary dims. On a model with partial rotary (`n_rot` 64 of a 256-wide head) the rotation's reshape then
glued four heads' slices into one vector: every shifted cell's K was destroyed, silently — the model kept
generating, but could only see the tokens decoded after the shift (recall 0/4, draft accept 1.00 while it
copied its own prompt back). Fix: view the whole head and rope inside it at the nope offset
(`[TAG_KV_SHIFT_HADAMARD]`). Affects any model with `n_rot < n_embd_head_k` and a quantized KV cache
under `--ctx-shift` or `--cache-reuse`.

**What the client must do**: drop old turns and resend the rest verbatim. The reuse loop advances in the
new prompt only on a match, so text inserted before the kept tail (a compaction summary, an edited system
prompt) means nothing is found and the whole prompt is re-processed. A client that inserts a *constant*
note pays that once and slides from then on. The server also rejects a prompt larger than the slot
(it does not truncate), so the client's window must stay ≤ `-c`.

**Cost note**: eviction frees cells at the head of the cache but does not compact it, so the cell window
the graph covers stays at the high-water mark. The slide saves the re-prefill, not the per-token
attention width — QSA already bounds that.

### 12d. Phase 0 result (2026-09-10): the selection is concentrated enough, and stable at a 20–25 % hot tier

Measured with `LLAMA_QSA_TRACE` (an eval callback on `indexer_top_blk-<il>`, `[TAG_QSA_TRACE]`) on the
serving build, no speculation, a 70,501-token code prompt (17,625 blocks) and 1,006 decoded tokens (500
continuing the code, 500 answering about four needle facts, recall 4/4). Analyzer:
`bench/qsa_phase0_analyze.py` on `bench/qsa_phase0_trace_70k.ids`. Per layer, 513 blocks are selected per
step; a "hot set at W" is every block selected in the previous W steps.

| hot set W (steps) | hot blocks per layer | share of the 17,625 | miss rate per step |
|---|---|---|---|
| 1 (previous step only) | 513 | 3 % | 32–50 % |
| 4 | ~1,000 | 6 % | 19–27 % |
| 16 | ~2,000 | 11 % | 10–14 % |
| 64 | ~3,500–4,200 | 20–24 % | 4–6 % |
| 256 | ~5,500–7,400 | 31–42 % | 2.4–3.2 % |

Blocks never selected in 1,006 steps: 30–45 % per layer (union of everything ever selected: 56–71 %).
The most-selected 10 % of blocks take 64–81 % of all selections. Cross-layer, one decode step touches
2,286 blocks on average (max 3,106), i.e. 114 MiB of K/V.

**Reading.** Not diffuse: a third of the context is dead weight over a thousand tokens and a hot tier of a
quarter of the blocks serves 95 % of selections. Not fully stable either: the 5 % that miss at W = 64 are
about 26 blocks per layer per step, ~1.3 MiB of K/V to fetch per token (26 MB/s at 20 t/s — trivial
bandwidth, but ~300 small random reads on the encrypted volume per token, on the critical path unless
fetched one step late, which loses exactly the blocks the model just started wanting).

**Payoff, honestly.** At the current 131k window the attention K/V is 1.6 GiB, so a 25 % hot tier saves
about 1.2 GiB on a 123 GiB box — not worth the complexity by itself. The tier pays only with Phase 3
(positions beyond `-c` at constant resident memory): at 1M tokens it would hold ~3 GiB hot plus a 1.5 GiB
index instead of 12.75 GiB, and the prefill-scratch bound (§12b) would still need its own fix. The
sliding window plus server-side truncation (§12c) already deliver "no compaction stall" without any of
this. Recommendation: build phases 1–3 only if context beyond the window is the goal; the numbers say it
would work, at a quality cost to be measured with needle recall.

### 12e. Drop-only tier measured (2026-09-10): retention policy cannot substitute for retrieval

A virtual hot tier (`LLAMA_QSA_VTIER=<frac>`, `[TAG_QSA_VTIER]`: the trace feeds every selection into
an LRU over blocks, union across the 12 layers, capacity frac × n_blocks; blocks outside it get −inf in the
per-block bias so they cannot be selected — a drop-only tier with no disk behind it). Same 70,570-token
prompt, four facts at 40/55/70/85 % of the text, two chat-form questions each (`bench/heretic_qsa_vtier.sh`,
`heretic_qsa_vtier_freq.sh`).

| hot tier | retention policy | applied to prefill too | recall Q1 | recall Q2 |
|---|---|---|---|---|
| none | — | — | 4/4 | 4/4 |
| 25 % | most recently used | no | 1/4 (the fact at 85 %) | 1/4 |
| 10 % | most recently used | no | 0/4 | 0/4 |
| 25 % | most recently used | yes | 0/4 | 0/4 |
| 25 % | most often selected | no | 1/4 (the fact at 40 %) | 1/4 |
| 10 % | most often selected | no | 1/4 (the fact at 40 %) | 1/4 |

Reading: recency keeps a sliding window of the newest blocks; frequency keeps whatever prefill queries
happened to like; neither predicts what a *later* question will need, and once a block is dropped the model
has no way to ask for it. A drop-only cache at a quarter of the context loses three of four planted facts.
So the only tier that preserves retrieval is the one with an index that stays resident and a fetch path
(§7): the decision has to be deferred until the model asks. Phase 0 (§12d) gives that design's cost — ~5 %
of a step's blocks fetched one step late at a 25 % hot set — and 12c already delivers "no compaction stall"
without any tier. Verdict: no drop-only tier; the disk tier only if context beyond the window is the goal.

---

## 13. Verification checklist — re-derive every fact

Run from the box; adapt paths. `PYN` needs numpy **and PyYAML** (`gguf/metadata.py` imports `yaml` at
module level even for reading): `~/.pyenv/versions/comfy-rocm/bin/python` has both (verified). With a
numpy-only python the snippets below still work because they stub `yaml` first (it is only used for
model-card parsing).

**A. Is QSA enabled? (`compress_ratios`) + core hparams**
```bash
PYN=/home/jocke/.pyenv/versions/comfy-rocm/bin/python
cd /home/jocke/LLM/tools/llama.cpp-main/gguf-py && PYTHONPATH=. $PYN - <<'EOF'
import sys, types; sys.modules.setdefault("yaml", types.ModuleType("yaml"))
from gguf import GGUFReader; import numpy as np
r=GGUFReader("/path/to/model-00001-of-N.gguf")
def get(k):
    f=r.fields[k]; return [int(np.asarray(f.parts[i]).ravel()[0]) for i in f.data] if len(f.data)>1 else np.asarray(f.parts[f.data[0]]).ravel()[0]
cr=get("qwen4exp.attention.compress_ratios"); print("QSA layers:", [i for i,v in enumerate(cr) if v], "ratio", sorted(set(cr)))
for k in ["qwen4exp.block_count","qwen4exp.full_attention_interval","qwen4exp.attention.head_count","qwen4exp.attention.head_count_kv",
          "qwen4exp.attention.key_length","qwen4exp.attention.indexer.head_count","qwen4exp.attention.indexer.key_length",
          "qwen4exp.attention.indexer.top_k","qwen4exp.ssm.inner_size","qwen4exp.ssm.state_size","qwen4exp.ssm.group_count","qwen4exp.ssm.conv_kernel"]:
    print(k,"=",get(k))
EOF
```
**B. Are the indexer tensors present? (needed to enable QSA)**
```bash
# list blk.N.indexer.* across all shards; expect q_proj [2560,512] BF16, k_proj [2560,128] BF16, q_norm/k_norm [128] F32 on the 12 layers (+ block 48)
PYTHONPATH=. $PYN -c "
import glob,sys,types;sys.modules.setdefault('yaml',types.ModuleType('yaml'));from gguf import GGUFReader
for f in sorted(glob.glob('/path/to/model-*.gguf')):
  for t in GGUFReader(f).tensors:
    if 'indexer' in t.name: print(f[-18:], t.name, list(t.shape), t.tensor_type.name)"
```
**C. Which tree has the prerequisites**
```bash
for t in llama.cpp-new llama.cpp-main; do d=~/LLM/tools/$t; echo "$t $(git -C $d branch --show-current) pooled=$(grep -c get_pooled_k $d/src/models/qwen4exp.cpp) gather=$(grep -c build_attn_qsa_gather $d/src/models/qwen4exp.cpp)"; done
```
**D. The hooks** (line numbers drift — grep, don't trust the numbers above)
```bash
NEW=~/LLM/tools/llama.cpp-new
grep -nE "const bool qsa =|build_qsa_top_k\(|build_attn_qsa_gather\(|ggml_get_rows\(ctx0, k_rows|indexer_top_k \+ r - 1|GGML_PAD\(width, 256\)" $NEW/src/models/qwen4exp.cpp
grep -nE "get_pooled_k|pooled_valid|pooled_rows|set_input_qsa|seq_rm" $NEW/src/llama-memory-hybrid-idx.h
grep -n -A6 "bool llama_memory_hybrid_idx::seq_rm" $NEW/src/llama-memory-hybrid-idx.cpp
sed -n '/bool llama_memory_recurrent::seq_rm/,/^}/p' $NEW/src/llama-memory-recurrent.cpp | grep -nE "n_rs_seq|return false|partially"
grep -n -A12 "uint32_t llama_hparams::n_embd_s" $NEW/src/llama-hparams.cpp
```
**E. Resolved on 2026-09-08 (were open):** `mem_idx` uses the attention cache's `type_k/type_v`
(`hybrid-idx.cpp:57-60`); `n_rs_seq = params.speculative.need_n_rs_seq()` (`common.cpp:1730`), 0 for the
draft context; `pooled_rows = kv_size/ratio + 2` (`hybrid-idx.cpp:83`).
**Still open before Phase 1:** whether `mem_idx` allocates an unused V; whether `-main`'s Vulkan
flash-attention skips fully-masked tiles (decides how much QSA-on helps on that binary, §12a); the exact
line numbers on the day you build (grep with cmd D — never trust the numbers in this doc).

---

## 14. Glossary
- **QSA** — the model's block-sparse attention (query scores block summaries, attends top-k).
- **indexer** — the small projection (`index_q_proj`/`index_k_proj`, 4×128) that produces scoring keys/queries.
- **pooled key** — the mean of a block's `r` indexer keys, normed + roped; one row per block; the index.
- **cell** — one slot in the attention/indexer KV cache (one token); addressed by index `j`, carries a position.
- **block** — `r` consecutive *positions*; `block = pos / r`.
- **GDN** — Gated DeltaNet linear attention; its fixed-size recurrent state is the floor.
- **hot / cold / GC'd** — the three K/V tiers (RAM / SSD / dropped).
- **watermark** — today's "valid pooled rows form a contiguous prefix" bookkeeping; replaced by a bitmap.
