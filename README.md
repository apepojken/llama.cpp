# This fork: sparse-attention depth work, a real sliding window, and server-side context trimming

Seven commits on top of upstream, ordered so each can be cherry-picked on its own or in the order listed.
Together they turn "the context is full" from a minutes-long re-prefill into a shift that costs seconds,
make the model read about two percent of its KV cache per token instead of all of it, and let any client
keep sending its whole conversation without ever trimming it. Measurements, design notes and the harnesses
behind every number are in `sparse-kv-tiering/` (`RESULTS.md`, `DESIGN.md`).

| commit | what it does | works with | needs |
|---|---|---|---|
| `230efe5c5` kv-cache: fix K-shift on a Hadamard-rotated quantized cache with partial rotary | **Bug fix.** With a quantized KV cache this tree stores K Hadamard-rotated across the whole head, but the K-shift graph viewed only the rotary dims, so any context shift or cache reuse silently destroyed every shifted cell's keys. | any model with `n_rot < n_embd_head_k` and a quantized cache (Qwen3-Next / qwen4exp: 64 of 256) | nothing |
| `fafebdde8` kv-cache: allow K-shift on M-RoPE sequences | The cache refused to shift M-RoPE models at all. Every position component moves by the same delta, so shift the extent with the position. `LLAMA_KV_NO_MROPE_SHIFT=1` restores the refusal. | any M-RoPE model (Qwen2.5-VL / Qwen3-VL family, qwen4exp) | nothing |
| `cdf82ef32` qwen4exp: block-level QSA selection, pooled indexer-key cache, gathered attention, chunked scoring | The depth work: select at block granularity, cache the pooled block keys, gather only the selected rows on decode, score in chunks. Decode at 56k: 19.4 → 25.9 t/s from enabling QSA, 28.8 t/s with this and speculation. | qwen4exp (Qwen3.8-Flash-Next) with `compress_ratios > 0` in the GGUF — the Heretic conversions ship with zeros; a 12-byte metadata patch enables it (DESIGN §12) | nothing |
| `d692e17d0` hybrid memory: sliding window support | A recurrent cache cannot rewind, so evicting a head or middle range was refused; evict it from the attention and indexer caches only and keep the state. The recurrent position follows a reuse shift; the indexer cache (keys stored before RoPE) skips the rope shift. | hybrid attention + recurrent models with the indexed memory (qwen4exp); the idea applies to any `llama_memory_hybrid` | the three above |
| `42533a9b6` server: cache reuse with media, and with a projector loaded | `--cache-reuse` was disabled whenever a projector was loaded. Media chunks are now matched whole, positions are mapped correctly when images are present, and a moved run carries its media entries. | any model; matters for multimodal serving | nothing (server only) |
| `2b08e2880` server: `--prompt-truncate` | A prompt larger than the window is trimmed at message boundaries (system prompt kept, newest message never dropped) instead of rejected; with `--cache-reuse` the kept tail is shifted, not re-processed. Off by default. | any model, any chat client | the reuse commit |
| `b54dcc996` server: `LLAMA_QSA_TRACE` | Measurement only: per-step trace of the indexer's block selection, inert without the variable. | qwen4exp | the qwen4exp commit |

**Flags that make the sliding window work end to end:** `--cache-reuse 256 --cache-ram 0 --prompt-truncate`
(`--cache-ram 0` because the host prompt cache otherwise clears the slot on a zero-prefix request). Keep the
hybrid model's context checkpoints small: `--ctx-checkpoints 4` (each one is 365 MiB at a 131k window here,
and the default keeps 32).

## Measured (Strix Halo, 123 GiB unified RAM, Vulkan/RADV, Qwen3.8-Flash-Next Heretic 125B-A6B Q4_K_M, q8_0 KV)

| | before | after |
|---|---|---|
| decode at 56k depth, no speculation | 19.4 t/s (QSA off, as shipped) | 25.9 t/s (QSA on) |
| decode at 56k with MTP speculation | 27.0 t/s | 28.8 t/s |
| client drops its oldest third of a 47k-token conversation | 218.4 s (full re-prefill) | **1.7 s** (71 tokens processed), answer byte-identical |
| same, with a picture inside the kept part (21.9k tokens) | 103.6 s | **2.1 s**, picture still read correctly |
| conversation of 15k tokens into an 8k window | rejected | served; the follow-up turn costs 40 tokens, 0.8 s |
| prefill at a 131k window, ubatch 512 / 1024 / 2048 | | 255 / 300 / 370 tok/s |
| memory: caches + fixed recurrent state, 131k / 262k window | | 3.4 / 6.0 GiB |

Two notes on this box that may not apply to yours. The model file and its n-gram table live on an encrypted
NVMe (dm-crypt), and anything that has to come back from disk is expensive: cold n-gram pages add about
13 ms per token, and at a 262k window a larger ubatch is *slower* (124 tok/s at 1536, 208 at 512) because the
compute scratch evicts the model's own pages. And the Vulkan device caps a single allocation at 4 GiB, which
is what stops ubatch 2048 at 262k (a 6.98 GB scratch buffer), independent of free RAM.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
