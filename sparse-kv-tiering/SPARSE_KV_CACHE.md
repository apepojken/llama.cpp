# No more compaction stalls: a sliding KV cache for a 125B hybrid MoE on one box (llama.cpp fork)

**TL;DR**

- When a coding agent fills its context, it "compacts": summarises the old turns and re-prefills the rest. On a 125B model at 100k+ tokens that is a 10 to 15 minute stall, every time.
- This fork makes the **server** slide the context instead: it evicts the dropped turns and *shifts* the kept ones in place. Same 47k-token conversation, client drops its oldest third: **218 s of re-prefill becomes 1.7 s**, and the answer is byte-identical.
- It works with pictures in the conversation, it needs **no client changes** (an optional flag trims over-long prompts at message boundaries), and enabling the model's own block-sparse attention on the way gave **+34 % decode** at 56k.
- Found and fixed a silent KV-cache corruption bug in the context-shift path that affects any model with partial rotary embeddings and a quantized cache.
- Everything is cherry-pickable, one commit per feature. Links at the bottom.

---

## The problem

Agent frameworks (I use `pi`, the pattern is universal) keep resending the whole conversation. When it no longer fits the context window they compact: serialise the old turns to text, ask the model for a summary, rebuild the prompt as `summary + recent turns`. Two things make that expensive on a local model: the summarisation request cannot reuse the live KV cache (different prefix), and the rebuilt prompt shares no prefix with the cache either, so the whole thing is prefilled again. On my box that is roughly 250 tokens per second of prefill, so a 78k-token history is about five minutes, plus a several-minute summary decode, plus the next turn re-prefilling the new prefix. Ten minutes of nothing, several times a day.

llama.cpp has had the right primitive for years: `--cache-reuse`. When a new prompt matches a run of tokens that is already in the cache but at a different offset, the server removes the cells in between and **shifts** the matching run to its new position by re-applying rotary embeddings (the "K-shift"). Only the genuinely new tokens are processed. It just did not work for this model, for three separate reasons, and fixing those is most of this work.

## The model, and why it matters here

Qwen3.8-Flash-Next (`qwen4exp` in llama.cpp), 125B parameters, about 6B active per token (MoE), with a multi-token-prediction draft block. Its 48 layers are a hybrid:

- **12 block-sparse attention layers.** Tokens are grouped in blocks of 4; a small *indexer* scores every block with a pooled key and each query attends only to the top-k blocks (k = 2048 tokens), plus the incomplete tail. This is the DeepSeek-style "lightning indexer" design. The GGUF conversions I had shipped with this **switched off** in the metadata (`compress_ratios` all zero), running dense attention instead. A 12-byte metadata patch turns it on; the indexer weights are already in the file.
- **36 recurrent (Gated DeltaNet) layers.** These keep a single fixed-size state, about 112 MiB, no matter how long the conversation is. They cannot be rewound: once the state has seen a token, it has seen it.

Two consequences that are easy to get wrong:

- Sparse attention does **not** shrink the KV cache. Every token's keys and values are still stored, because the indexer may pick any block later. What shrinks is the *reading*: each token attends to about 2 % of a 131k window. The index that decides is about a tenth the size of the cache it summarises.
- Only the 12 attention layers have a cache at all, which is why a 131k window costs 2.6 GiB rather than four times that.

## What was built

All of it is server-side, on the branch linked below. In cherry-pick order:

**1. K-shift fix for a rotated, quantized cache (a real bug).** With a quantized KV cache this tree stores keys in a Walsh-Hadamard-rotated basis spanning the whole head. The shift graph handed the rotation a view of only the rotary dims, 64 of 256 on this model. The rotation's reshape glued four heads into one vector, roped that, and wrote it back: every shifted cell's keys were destroyed, silently. The model kept generating but could only see tokens decoded after the shift (needle recall 0/4, draft acceptance 1.00 while it copied its own prompt back). Any model with `n_rot < n_embd_head_k` and a quantized cache under `--ctx-shift` or `--cache-reuse` is affected.

**2. K-shift on M-RoPE models.** The cache refused to shift multi-axis rotary models at all. For text every position component equals the token position, and for image patches the extent is an absolute offset from the image's start, so all components move by the same delta and a single rotation per cell is exact.

**3. The sparse attention depth work.** Selection at block granularity, a cache of the pooled block keys so only new blocks are re-pooled each step, a gather path that reads exactly the selected rows on decode instead of masking the whole window, and chunked scoring to bound prefill scratch. Target-only step time at 56k: 59 ms to 42 ms.

**4. Sliding-window support in the hybrid memory.** Three things a recurrent cache needed. Evicting a head or middle range was refused because the recurrent state cannot rewind; but dropping a range that does not touch the end needs no rewind, the state is a running summary through the end and stays valid, so the eviction now applies to the attention and indexer caches only. The recurrent state's *position* has to follow a reuse shift, which stops one token short of it; left behind, the server concluded the cache could not serve the resume point and re-prefilled everything. And the indexer's keys are stored before rotary embedding, so the shift must update their positions without rotating them.

**5. Cache reuse with pictures.** Upstream disables `--cache-reuse` entirely when a projector is loaded. Now a media chunk is matched as a whole against the same chunk, token indices are mapped to positions correctly when images are present, and a moved run carries its media entries. A picture inside the kept part of the window survives the shift and is still read correctly.

**6. `--prompt-truncate`.** The server rejects a prompt larger than the window; it does not trim. With this flag it keeps the system prompt (everything before the first user message, or `--keep N` tokens) plus the newest messages and drops whole messages in between, never splitting an image and never dropping the newest user message. Paired with `--cache-reuse` the kept tail is shifted, not re-processed. Any client that just keeps sending its whole conversation gets a sliding window with no changes at all.

## Numbers

Box: Strix Halo (Ryzen AI MAX+ 395, 123 GiB unified memory), Vulkan/RADV, q8_0 KV cache. Every number comes from a harness in the repo, named in `RESULTS.md`.

Enabling the sparse attention (56k-token prompt, four planted facts, no speculation):

| | prefill | decode | recall |
|---|---|---|---|
| dense, as the GGUF shipped | 295 tok/s | 19.4 t/s | 4/4 |
| sparse attention on | 283 tok/s | **25.9 t/s** | 4/4 |
| plus the depth work, with MTP speculation | 274 tok/s | **28.8 t/s** | 4/4 |

The sliding window (`--cache-reuse 256 --cache-ram 0`):

| scenario | tokens processed | prompt time | quality |
|---|---|---|---|
| 47,075-token prompt, client drops its oldest third | **71** | **1.7 s** | recall 4/4, byte-identical to control |
| same prompt, fresh full prefill | 47,075 | 218.4 s | recall 4/4 |
| second slide on the already-shifted cache | 72 | 1.2 s | recall 4/4 |
| 21,875-token prompt with a picture in the kept part | 64 | 2.1 s | picture read correctly, same as control at 103.6 s |

Server-side truncation (`--prompt-truncate`):

| scenario | result |
|---|---|
| 15k-token conversation into an 8k window | served, system prompt kept, newest turn answered correctly |
| one more turn on top | **40 tokens, 0.8 s** |
| same request without the flag | rejected, unchanged default |

Decode speed after a slide was the same as after a full prefill (39.8 to 43.3 t/s against 38.1 to 43.0).

## How to run it

    llama-server -m model-00001-of-N.gguf --mmproj mmproj.gguf \
      -c 131072 -ub 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
      --cache-reuse 256 --cache-ram 0 --prompt-truncate --ctx-checkpoints 4

- `--cache-ram 0` matters: the host prompt cache otherwise clears the slot on a zero-prefix request and there is nothing left to reuse.
- `--ctx-checkpoints 4`: on a hybrid model each context checkpoint holds a copy of the recurrent state plus per-token data, about 365 MiB at a 131k window and 619 MiB at 262k. The default keeps 32, which is 11 to 19 GiB of host RAM waiting to happen.
- The sparse attention needs `compress_ratios > 0` in the GGUF. If your conversion has zeros, the patch recipe is in `DESIGN.md`, section 12. Shard 1 of these files is metadata only, so it is a 10 MB copy.

**What a client has to do:** drop old turns and resend the rest verbatim. Reuse can only anchor on the first token that differs, so text *inserted* before the kept turns (a changing summary) means nothing matches. A constant note pays one re-prefill and slides from then on; no note at all slides immediately. With `--prompt-truncate` the client can also do nothing and let the server trim.

## Caveats, honestly

- The recurrent layers remember what was dropped. Their state is not rewound when the window slides, so a fading trace of the dropped text survives. In every test the slid answers matched a fresh prefill exactly, but it is a difference in kind from a pure transformer.
- Eviction frees cells at the front of the cache but does not compact it, so the cell window the graph covers stays at its high-water mark. The slide saves the re-prefill, not the per-token attention width, which the sparse attention bounds anyway.
- Two things about my box that may not apply to yours: the model file and its 50 GB n-gram table live on an encrypted NVMe (dm-crypt), and anything that has to come back from disk is expensive, so at a 262k window a *larger* ubatch is slower (124 tok/s at 1536, 208 at 512) because the compute scratch evicts the model's pages. And the Vulkan device caps a single allocation at 4 GiB, which is what stops ubatch 2048 at 262k regardless of free memory.
- The chat template trims message content. A harness that slices text mid-line loses its leading whitespace and the reuse loop finds nothing; real clients resend whole messages and are unaffected. It cost me an afternoon.

## Can the cache be made smaller instead? Measured, and no

The obvious next question: if each token reads 2 % of the cache, why keep all of it? I traced the indexer's block selection at 70k depth over a thousand decoded tokens. A "hot set" of blocks used in the last 64 steps, about a quarter of the context, serves 95 % of what the next step asks for, and a third of the blocks were never selected. That is the case for a *disk* tier with a resident index: evict, but fetch on demand.

A *drop-only* cap does not work. Hiding everything outside a 25 % hot tier from the selection kept 1 of 4 planted facts, with either recency or selection-frequency deciding what stays; 10 % kept 0 or 1. Neither policy predicts what a later question will need, and once a block is dropped the model cannot ask for it. The decision has to be deferred until the model wants the block, which is exactly what an index plus a fetch path does, and that is the part not built.

## Bonus: the IQ3_M quant on the same harnesses

Same binary, same flags, back to back:

| | IQ3_M | Q4_K_M |
|---|---|---|
| GPU-held memory after load | 76.0 GiB | 97.1 GiB |
| 56k: prefill / decode / draft accept | 248 tok/s / 34.1 t/s / 0.94 | 220 / 29.3 / 0.87 |
| needle recall, 14k and 56k | 4/4, 4/4 | 4/4, 4/4 |
| perplexity, 64k of real code | 1.1179 ± 0.0043 | 1.1151 ± 0.0043 |
| text read out of six rendered pictures | 6/6 | 6/6 |

21 GiB back and faster decode for a perplexity difference inside the error bars. I switched.

## Where the work is

Branch `qwen4exp-qsa-block` on my fork: `<your fork URL here>`. The commits, in cherry-pick order; each one's message says what it does and what it applies to:

| commit | what |
|---|---|
| `230efe5c5` | kv-cache: fix K-shift on a Hadamard-rotated quantized cache with partial rotary |
| `fafebdde8` | kv-cache: allow K-shift on M-RoPE sequences |
| `cdf82ef32` | qwen4exp: block-level QSA selection, pooled indexer-key cache, gathered attention, chunked scoring |
| `d692e17d0` | hybrid memory: sliding window support |
| `42533a9b6` | server: cache reuse with media, and with a projector loaded |
| `2b08e2880` | server: `--prompt-truncate` |
| `b54dcc996` | server: `LLAMA_QSA_TRACE`, per-step trace of the indexer's block selection |
| `17ddf2fd0` | measurement: `LLAMA_QSA_VTIER`, a virtual hot tier over the block selection |

Documents in the repo: `README.md` (the table above with model applicability), `sparse-kv-tiering/RESULTS.md` (every number with the harness that produced it), `sparse-kv-tiering/DESIGN.md` (the design, the verification commands, and the measurements including the selection trace and the drop-only experiment).
