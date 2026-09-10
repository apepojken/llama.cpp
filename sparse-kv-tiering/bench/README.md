# Bench harnesses

These produced every number in `../RESULTS.md`. They were written for the author's box and then made
overridable; each script sets its paths at the top and every one can be overridden from the environment:

| variable | meaning | default |
|---|---|---|
| `LLAMA_BIN` | directory with `llama-server` / `llama-perplexity` | `<repo>/build/bin` |
| `BD` | this directory (prompts, corpus, outputs) | the script's own directory |
| `MODEL`, `ON`, `M_QSA` | the sparse-attention-enabled GGUF (shard 1) | the author's path |
| `OFF`, `M_DENSE` | the same weights with sparse attention off | the author's path |
| `MMPROJ` | the vision projector | the author's path |
| `PYTHON` | a python with numpy (analyzer) and PIL (test images) | `python3` |

What they need: a spare port (8097), the GPU to themselves (they refuse to start with less than 82 GiB
free, because the model is ~80 GB), and patience — a 70k-token prefill is several minutes. Judge by
`prompt_n` (tokens actually processed), `reusing chunk ... shifting KV cache` lines in the server log,
draft acceptance, and needle recall; the "quote the first line" style answers on raw code are unreliable
and the scripts say so where it matters.

| script | measures |
|---|---|
| `heretic_qsa_bench.sh` | enabling the block-sparse attention: prefill, decode, recall, sparse on/off |
| `heretic_qsab_validate.sh`, `heretic_qsaa_validate.sh`, `heretic_qsac_validate.sh`, `heretic_qsa_chunk_check.sh` | the depth changes (block selection, pooled key cache, gathered attention, chunked scoring): greedy oracle at 14k, recall at 56k, per-op profile, 64k perplexity |
| `heretic_slide_check.sh` | the slide mechanics on raw code: eviction + shift, `prompt_n` ≈ the question |
| `heretic_slide_recall.sh` | the quality gate: needle recall on a slid cache vs a fresh prefill, twice |
| `heretic_slide_debug.sh` | isolation variants (pooled cache off, gather off, sparse-off model, f16 cache) — this is what found the K-shift bug |
| `heretic_slide_image.sh`, `heretic_slide_image_read2.sh` | the slide with a picture in the kept part, the dropped part, or the new turn; whether the picture is still readable |
| `heretic_reuse_probe.sh` | raw `/completion` vs chat endpoint reuse (the chat template trims content) |
| `heretic_truncate_check.sh` | `--prompt-truncate`: served vs rejected, follow-up cost |
| `heretic_262k_probe.sh` | largest ubatch that fits at 262k and its prefill speed |
| `heretic_qsa_phase0.sh` + `qsa_phase0_analyze.py` | selection trace at depth: hot-set sizes and miss rates |
| `heretic_qsa_vtier.sh`, `heretic_qsa_vtier_freq.sh` | the drop-only virtual tier at 25 % / 10 % with recency or frequency retention |
| `heretic_quant_compare.sh`, `heretic_vision_compare.sh` | IQ3_M vs Q4_K_M: memory, speed, recall, perplexity, picture transcription |
| `heretic_shiftfix_oracle.sh` | the 14k greedy oracle against the stored reference, run before every deployment |

Data: `ppl_corpus.txt` is ggml source concatenated (the filler and the perplexity corpus);
`heretic_qsa_req*.json` and `deep_prompt.txt` are synthetic fact lists; `heretic_qsa_needles.json` the
four facts checked for recall; the PNGs are rendered strings. `results/` holds the outputs the numbers
were read from.
