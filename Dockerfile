# Qwen3.8-Flash-Next (qwen4exp) on Strix Halo — Vulkan build with speculative
# decoding fixes, native MTP support, and fused mat-vec epilogs (this branch).
#
# Build:
#   docker build -t qwen4exp-llama .
#
# Run (mount a directory containing the model shards and the MTP sidecar):
#   docker run --device /dev/dri -p 8095:8095 \
#     -v /path/to/models:/models qwen4exp-llama
#
# The default command expects these files in the mounted /models directory:
#   Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf  (+ shards 2 and 3)
#     from https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
#   mtp-Qwen-Qwen3.8-Flash-Next-Q8_0.gguf
#     from https://huggingface.co/jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF
# Pass your own llama-server arguments after the image name to override them.
#
# Requires a host GPU with a Vulkan driver exposed through /dev/dri. The RADV
# userspace driver ships inside the image (Strix Halo / gfx1151 needs a recent
# Mesa, hence the Ubuntu base below).

ARG UBUNTU_VERSION=26.04
ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

WORKDIR /app/tools/ui
COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci
COPY tools/ui/ ./
RUN npm run build

FROM docker.io/ubuntu:$UBUNTU_VERSION AS build

RUN apt update && apt install -y \
    git build-essential cmake curl libssl-dev libcurl4-openssl-dev \
    libvulkan-dev glslc spirv-headers

WORKDIR /app
COPY . .
COPY --from=web /app/tools/ui/dist tools/ui/dist

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-server llama-cli llama-batched-bench

RUN mkdir -p /out/bin /out/lib && \
    cp build/bin/llama-server build/bin/llama-cli build/bin/llama-batched-bench /out/bin/ && \
    find build -name "*.so*" -exec cp -P {} /out/lib \;

FROM docker.io/ubuntu:$UBUNTU_VERSION AS runtime

RUN apt update && apt install -y \
    mesa-vulkan-drivers libvulkan1 libcurl4t64 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/bin/ /app/
COPY --from=build /out/lib/ /app/lib/

ENV LD_LIBRARY_PATH=/app/lib
EXPOSE 8095

ENTRYPOINT ["/app/llama-server", "--host", "0.0.0.0", "--port", "8095"]

# Recommended serving configuration for Strix Halo (128 GB unified memory).
# Override by appending your own arguments to `docker run ... qwen4exp-llama`.
CMD ["-m", "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf", \
     "--device", "Vulkan0", "-ngl", "999", "--flash-attn", "on", "--no-mmap", \
     "--jinja", "-c", "262144", "--parallel", "1", \
     "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", \
     "--spec-type", "draft-mtp", \
     "--spec-draft-model", "/models/mtp-Qwen-Qwen3.8-Flash-Next-Q8_0.gguf", \
     "--spec-draft-ngl", "999", "--spec-draft-n-max", "6", "--spec-draft-p-min", "0.75"]
