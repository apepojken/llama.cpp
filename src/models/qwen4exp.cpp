#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP (same convention as qwen35): the GGUF block count includes the MTP
    // block(s); n_layer() = n_layer_all - n_layer_nextn is the trunk
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);


    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0 && "qwen4exp needs a hyper-connection count");
    GGML_ASSERT(hparams.hc_low_rank  > 0 && "qwen4exp needs a hyper-connection low rank");
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;


    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    std::fill(hparams.is_ple_impl.begin(), hparams.is_ple_impl.end(), 0);
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        for (uint32_t il : ple_layers) {
            GGML_ASSERT(il < hparams.n_layer_all);
            hparams.is_ple_impl[il] = 1;
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        GGML_ASSERT(hparams.ple_ngram_size >= 2 && hparams.ple_ngram_size <= LLAMA_MAX_PLE_NGRAM);
        GGML_ASSERT(hparams.ple_n_heads > 0 && hparams.ple_n_heads <= LLAMA_MAX_PLE_HEADS);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_head_vocab_sizes);
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // an mtp- sidecar carries only the MTP block plus embeddings/head; the trunk is absent
    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int tf        = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, tf);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, tf);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, tf);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back
    // (the MTP block has no PLE, so a sidecar ships no table)
    if (hparams.ple_n_heads > 0 && !mtp_only) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto * ple_w = ml.get_weight(ple_name.c_str());
        GGML_ASSERT(ple_w != nullptr && "qwen4exp is missing the PLE n-gram table");
        const int64_t ple_rows = ple_w->tensor->ne[1];
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, 0);
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, tf);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, tf);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, tf);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, tf);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, tf);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, tf);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, tf);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, tf);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, tf);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, tf);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, tf);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, tf);


            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, tf);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, tf);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, tf);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, tf);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, tf);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, tf);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, tf);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, tf);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, tf);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, tf);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, tf);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, tf);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, tf);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, tf);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, tf);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, tf);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, tf);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, tf);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, tf);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, tf);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, tf);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, tf);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, tf);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, tf);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, tf);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, tf);
    }

    // NextN/MTP block(s): one dense-attention hyper-connection block appended beyond the
    // trunk, plus the combiner (fc_embd/fc_hidden and their pre-norms) and the head's own
    // hyper-connection mixer. No indexer (the draft head runs dense) and no PLE.
    for (int il = n_layer; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const int mf = !ml.load_mtp ? TENSOR_SKIP : 0;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, mf);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, mf);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, mf);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, mf);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, mf);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, mf);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, mf);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, mf);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mf);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, mf);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mf);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mf);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, mf);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, mf);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, mf);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, mf);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, mf);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, mf);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, mf);

        // the block's indexer projections are converted along with everything else, but the
        // draft head runs dense attention: load them so the tensor count matches, unused
        {
            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, mf | TENSOR_NOT_REQUIRED);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, mf | TENSOR_NOT_REQUIRED);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, mf | TENSOR_NOT_REQUIRED);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, mf | TENSOR_NOT_REQUIRED);
        }

        layer.nextn.fc_embd   = create_tensor(tn(LLM_TENSOR_NEXTN_FC_EMBD,   "weight", il), { n_embd, n_embd }, mf);
        layer.nextn.fc_hidden = create_tensor(tn(LLM_TENSOR_NEXTN_FC_HIDDEN, "weight", il), { n_embd, n_embd }, mf);
        layer.nextn.enorm     = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,     "weight", il), { n_embd }, mf);
        layer.nextn.hnorm     = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,     "weight", il), { hc_dim }, mf);
        layer.nextn.hc_norm   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_NORM,   "weight", il), { hc_dim }, mf);
        layer.nextn.hc_down   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_DOWN,   "weight", il), { hc_dim, hc_lr }, mf);
        layer.nextn.hc_up     = create_tensor(tn(LLM_TENSOR_NEXTN_HC_UP,     "weight", il), { hc_lr, hc_dim }, mf);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.8-Flash-Next.
// Combiner: fc_embd(enorm(emb(x_{t+1}))) + fc_hidden(mean_hc(hnorm(h_wide_t))), widened
// into hc identical streams; one dense-attention hyper-connection block (the model's own
// QSA layer run dense - the draft context has no indexer cache); closed by the head's own
// low-rank mixer and the shared output head. All norm gammas are converter-folded (1 + w).
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.fc_embd   && "MTP block missing nextn.fc_embd");
    GGML_ASSERT(layer.nextn.fc_hidden && "MTP block missing nextn.fc_hidden");
    GGML_ASSERT(layer.nextn.enorm     && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm     && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_norm   && "MTP block missing nextn.hc_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_wide = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    // grouped RMSNorm over the hc streams, gamma over the whole [hc_dim] layout
    auto grouped_norm = [&](ggml_tensor * x2d, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x2d, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        return ggml_mul(ctx0, t, w);
    };

    // collapse the hc streams of a [hc_dim, T] tensor by their mean
    auto mean_hc = [&](ggml_tensor * x2d) {
        ggml_tensor * x3 = ggml_reshape_3d(ctx0, x2d, n_embd, hc, n_tokens);
        // the first slice stays a view: the first add produces the contiguous tensor,
        // saving a full-size copy (hc >= 2 for every qwen4exp checkpoint)
        ggml_tensor * m = ggml_view_2d(ctx0, x3, n_embd, n_tokens,
                ggml_row_size(x3->type, n_embd) * hc, 0);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * st = ggml_view_2d(ctx0, x3, n_embd, n_tokens,
                    ggml_row_size(x3->type, n_embd) * hc,
                    ggml_row_size(x3->type, n_embd) * c);
            m = ggml_add(ctx0, m, st);
        }
        return ggml_scale(ctx0, m, 1.0f / (float) hc);
    };

    // hyper-connection read gate: norm -> low-rank silu/sigmoid gate -> mean collapse
    // (mirror of llama_model_qwen4exp::graph::build_hc_mix)
    auto hc_mix = [&](ggml_tensor * x3d, ggml_tensor * w_norm, ggml_tensor * w_down,
                      ggml_tensor * w_up, ggml_tensor * w_inject, ggml_tensor ** inject) {
        ggml_tensor * xn = ggml_rms_norm(ctx0, x3d, hparams.f_norm_rms_eps);
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, n_tokens);
        xn = ggml_mul(ctx0, xn, w_norm);

        ggml_tensor * lo = build_lora_mm(w_down, xn);
        lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
        ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));

        ggml_tensor * gated = ggml_mul(ctx0, xn, gate);

        if (inject) {
            // fold the combine-gate activation onto the inject mat-vec so the
            // backend can fuse mat-vec + scale + sigmoid + scale into one dispatch
            ggml_tensor * w = build_lora_mm(w_inject, xn);
            w = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, w, 1.0f / (float) hc)), 2.0f);
            *inject = w;
        }

        return mean_hc(gated);
    };

    // hyper-connection write gate (mirror of build_hc_combine)
    auto hc_combine = [&](ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject) {
        // inject already carries the 2*sigmoid(x/hc) activation (see hc_mix)
        ggml_tensor * w = ggml_reshape_3d(ctx0, inject, 1, hc, n_tokens);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, n_tokens);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, n_tokens, 1);

        return ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    };

    // -- combiner --------------------------------------------------------

    ggml_tensor * e = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e = build_lora_mm(layer.nextn.fc_embd, e);
    cb(e, "mtp_fc_embd", il);

    // combiner variant B: keep the per-stream structure of the target's widened residual.
    // fc_hidden is applied to each stream independently (it is [n_embd, n_embd]) and the
    // embedding path is broadcast into every stream, so the hc streams stay distinct -
    // collapsing them by mean first (DeepSeek-style) threw that information away and
    // measured lower draft acceptance.
    ggml_tensor * hn = grouped_norm(h_wide, layer.nextn.hnorm);
    ggml_tensor * hs = ggml_reshape_2d(ctx0, hn, n_embd, hc * n_tokens);
    hs = build_lora_mm(layer.nextn.fc_hidden, hs);
    hs = ggml_reshape_3d(ctx0, hs, n_embd, hc, n_tokens);
    cb(hs, "mtp_fc_hidden", il);

    ggml_tensor * e_wide = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);

    ggml_tensor * res_hc = ggml_add(ctx0, hs, e_wide);
    cb(res_hc, "mtp_hc_init", il);

    // -- attention (dense; the draft context carries no indexer cache) ---

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = hc_mix(res_hc, layer.hc_attn_norm, layer.hc_attn_down,
                               layer.hc_attn_up, layer.hc_attn_inject, &inject);
    cb(cur, "mtp_hc_attn_mix", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    res_hc = hc_combine(res_hc, cur, inject);
    cb(res_hc, "mtp_hc_attn_combine", il);

    // -- MoE FFN ---------------------------------------------------------

    cur = hc_mix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down,
                 layer.hc_ffn_up, layer.hc_ffn_inject, &inject);
    cb(cur, "mtp_hc_ffn_mix", il);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_out = moe_out;
    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, NULL, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);

        ggml_tensor * shared_gate = ggml_sigmoid(ctx0, build_lora_mm(layer.ffn_gate_inp_shexp, cur));
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);

        ffn_out = ggml_add(ctx0, moe_out, ffn_shexp);
    }
    cb(ffn_out, "mtp_ffn_out", il);

    res_hc = hc_combine(res_hc, ffn_out, inject);
    cb(res_hc, "mtp_hc_ffn_combine", il);

    // -- head ------------------------------------------------------------

    // the next chained draft step reads the widened stream, like the trunk export
    ggml_tensor * h_next = ggml_reshape_2d(ctx0, res_hc, hc_dim, n_tokens);
    cb(h_next, "h_nextn", -1);
    ggml_build_forward_expand(gf, h_next);
    res->t_h_nextn = h_next;

    // the head's own mixer is its output norm, as in the trunk
    ggml_tensor * final = hc_mix(res_hc, layer.nextn.hc_norm, layer.nextn.hc_down,
                                 layer.nextn.hc_up, nullptr, nullptr);
    cb(final, "mtp_head_mix", il);

    if (inp_out_ids) {
        final = ggml_get_rows(ctx0, final, inp_out_ids);
    }

    ggml_tensor * logits = build_lora_mm(model.output, final, model.output_s);
    cb(logits, "result_output", -1);

    res->t_logits = logits;
    ggml_build_forward_expand(gf, logits);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    // the first slice stays a view: the first add produces the contiguous tensor,
    // saving a full-size copy per hc_mix (~1ms x 95 per 2048-token prefill chunk)
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        // decode: plain mat-vec, so the backend fuses mat-vec + scale + sigmoid + scale.
        // batched prefill: m=4 wastes 15/16 of every MM tile (measured 77 GFLOPS beside
        // 6-17 TFLOPS neighbors); computing the transpose routes through the mat-vec
        // shaders (4 columns <= the 8-column limit) and a tiny [nt,4] transpose restores
        // the layout.
        ggml_tensor * w;
        // the transpose path puts w_inject on the mul_mat B side, which the Vulkan mat-vec
        // accepts only as f32/f16/q8_1 - a quantized inject weight asserts there. Only take
        // it when w_inject is unquantized; otherwise use the normal weight-as-A path.
        const bool w_inject_unquant = w_inject->type == GGML_TYPE_F32 || w_inject->type == GGML_TYPE_F16;
        if (nt > 8 && w_inject_unquant) {
            w = ggml_mul_mat(ctx0, xn, w_inject);
            w = ggml_cont(ctx0, ggml_transpose(ctx0, w));
        } else {
            w = build_lora_mm(w_inject, xn);
        }
        w = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, w, 1.0f / (float) hc)), 2.0f);
        *inject = w;
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid (applied in build_hc_mix, fused with the inject mat-vec) centres the
    // scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_reshape_3d(ctx0, inject, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // the widened pre-mixer stream seeds the MTP draft head: its combiner applies
    // pre_fc_norm_hidden on the raw hc-wide residual, so export it before the mixer
    {
        ggml_tensor * h_wide = ggml_reshape_2d(ctx0, res_hc, hc * n_embd, n_tokens);
        if (inp_out_ids && cparams.embeddings_nextn_masked) {
            h_wide = ggml_get_rows(ctx0, h_wide, inp_out_ids);
        }
        cb(h_wide, "h_nextn", -1);
        // the view has no consumer in the graph: expand it so the scheduler assigns a backend
        ggml_build_forward_expand(gf, h_wide);
        res->t_h_nextn = h_wide;
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, each scored by one
// mean-pooled indexer key, plus the incomplete tail. set_input resolves the cache layout.
class llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio) :
        mctx(mctx), ratio(ratio) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio,
                dirty_cells, dirty_pos, dirty_rows);
    }

    // the idx cache pads n_kv like the attention cache, so the tensor shapes are stable
    // across long stretches of decoding; without this override the default can_reuse
    // returns false and the whole graph is rebuilt and re-recorded every token, which
    // dominates decode wall time (measured: ~3.8 ms of GPU work inside a ~52 ms token)
    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_new = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        if (mctx_new->get_idx() == nullptr) {
            return false;
        }

        mctx = mctx_new;

        bool res = true;

        res &= k_idxs->ne[0]   == (int64_t) params.ubatch.n_tokens;
        res &= cell_blk->ne[0] == (int64_t) mctx_new->get_idx()->get_n_kv();
        res &= cell_blk->ne[1] == (int64_t) mctx_new->get_n_stream();
        res &= bias->ne[1] * bias->ne[2] == (int64_t) params.ubatch.n_tokens;

        // [TAG_QSA_POOLED_CACHE] the dirty tables must hold this ubatch's (re)pool range;
        // steady decode needs at most one block, so the capacity is stable at 1 there
        if (dirty_rows != nullptr) {
            res &= dirty_rows->ne[0] == (int64_t) mctx_new->qsa_pooled_n_dirty_max(params.ubatch, ratio);
        }

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_kv, n_tokens/n_stream, n_stream]

    // [TAG_QSA_POOLED_CACHE] present only when the pooled cache path is active
    ggml_tensor * dirty_cells = nullptr; // I32 [ratio*n_dirty_max, 1]
    ggml_tensor * dirty_pos   = nullptr; // I32 [4*n_dirty_max]
    ggml_tensor * dirty_rows  = nullptr; // I64 [n_dirty_max]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // the tables and bias depend only on the cells and the ubatch: every QSA layer in the
    // graph shares one input, so the host fills them once per batch instead of once per layer
    llm_graph_input_qsa * inp = (llm_graph_input_qsa *) qsa_shared;
    if (inp == nullptr) {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->bias);

        // [TAG_QSA_POOLED_CACHE] complete blocks' summaries are cached; per ubatch only the
        // freshly completed blocks (plus any pending refill after a full state load) are
        // pooled/normed/roped, and the score reads the cache. The full-recompute tables are
        // then dead graph inputs, so they are not created at all (an unreferenced input is
        // never allocated, and filling it would write through a null pointer).
        // Kill switch for A/B testing.
        if (mctx_hyb->get_pooled_k(il) != nullptr && n_stream == 1 &&
            getenv("LLAMA_QSA_NO_POOLED_CACHE") == nullptr) {
            const int64_t n_dirty_max = mctx_hyb->qsa_pooled_n_dirty_max(ubatch, (uint32_t) r);

            qsa->dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_dirty_max, 1);
            qsa->dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_dirty_max);
            qsa->dirty_rows  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_dirty_max);

            ggml_set_input(qsa->dirty_cells);
            ggml_set_input(qsa->dirty_pos);
            ggml_set_input(qsa->dirty_rows);
        } else {
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);

            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
        }

        inp = qsa.get();
        qsa_shared = inp;
        res->add_input(std::move(qsa));
    } else {
        // the ratio is per-layer in the hparams but uniform in practice; sharing requires it
        GGML_ASSERT(inp->ratio == (uint32_t) r && "QSA input sharing assumes a uniform compress ratio");
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    ggml_tensor * pooled = nullptr;

    if (inp->dirty_rows != nullptr) {
        // [TAG_QSA_POOLED_CACHE] pool only this ubatch's dirty blocks and scatter them into
        // the cache; the score then reads the cache. Rows of incomplete blocks hold stale
        // (finite) data and are masked by the -inf bias, exactly like the garbage partial
        // pools of the full recompute below.
        ggml_tensor * store = mctx_hyb->get_pooled_k(il);
        GGML_ASSERT(store != nullptr);

        const int64_t n_dirty_max = inp->dirty_rows->ne[0];

        ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->dirty_cells);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_dirty_max, 1);

        ggml_tensor * fresh = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_view_3d(ctx0, members, idx_dim, n_dirty_max, 1,
                    members->nb[2], members->nb[3], i*members->nb[1]);
            // r >= 2, so the first add materializes the contiguous sum
            fresh = fresh ? ggml_add(ctx0, fresh, slice) : slice;
        }
        GGML_ASSERT(r >= 2);
        fresh = ggml_scale(ctx0, fresh, 1.0f/(float) r);
        cb(fresh, "indexer_k_pooled", il);

        fresh = ggml_reshape_3d(ctx0, fresh, idx_dim, 1, n_dirty_max);
        fresh = build_norm(fresh, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
        fresh = ggml_rope_multi(ctx0, fresh, inp->dirty_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        fresh = ggml_reshape_2d(ctx0, fresh, idx_dim, n_dirty_max);

        ggml_tensor * store_view = ggml_view_2d(ctx0, store,
                idx_dim, store->ne[1], store->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, store_view, fresh, inp->dirty_rows));

        pooled = ggml_view_3d(ctx0, store, idx_dim, n_blocks, 1,
                store->nb[1], store->nb[1]*n_blocks, 0);
        cb(pooled, "indexer_k", il);
    } else {
        // gathers per stream: blk_cells row s indexes stream s's own cells
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

        // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                    members->nb[2], members->nb[3], i*members->nb[1]);
            pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
        }
        GGML_ASSERT(r >= 2);
        pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
        cb(pooled, "indexer_k_pooled", il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
        pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
        pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        cb(pooled, "indexer_k", il);
    }

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    // q is a fresh rope output and always contiguous; reshape asserts if that ever changes
    ggml_tensor * q_flat = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream);

    ggml_tensor * expanded;
    if (n_tps > 8) {
        // batched prefill: with q as the LEFT operand the heads land in dim 0 (h fastest),
        // so the head sum needs no permute+cont, and the summed score comes out directly
        // in the [n_tps, n_blocks] layout the expansion gather wants. This removes two of
        // the three full-size permute copies (~335MB/layer/chunk at 32k); the transpose
        // feeding the row-wise top-k below is the only one that survives.
        ggml_tensor * score = ggml_mul_mat(ctx0, q_flat, pooled);      // [nh*n_tps, n_blocks, ns]
        score = ggml_relu(ctx0, score);
        score = ggml_reshape_2d(ctx0, score, n_idx_h, n_tps*n_blocks*n_stream);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_reshape_3d(ctx0, score, n_tps, n_blocks, n_stream);
        cb(score, "indexer_score", il);

        // give every token of a block the block score; the budget is a whole number of
        // blocks, so the top-k cut still lands on a block boundary
        expanded = ggml_get_rows(ctx0, score, inp->cell_blk);          // [n_tps, n_kv, ns]
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));
        expanded = ggml_add(ctx0, expanded, inp->bias);
    } else {
        // decode keeps pooled as the left operand: the swap would put the score matmul
        // on the tiled MM path with nh=4 rows (the skinny-m pathology); at n_tps <= 8 the
        // permutes are tiny anyway
        ggml_tensor * score = ggml_mul_mat(ctx0, pooled, q_flat);
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
        score = ggml_relu(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
        score = ggml_sum_rows(ctx0, score);
        score = ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream);
        cb(score, "indexer_score", il);

        expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // [TAG_QSA_GATHER] (from EngramHalo.cpp) the decode gather path needs the selection padded
    // to the flash attention KV granularity; build_attn_qsa_gather masks the extra entries out
    // again, so the visible set stays `width`
    const int64_t n_sel = std::max<int64_t>(width, qsa_gather_n_sel(n_kv, width));

    // ggml_top_k (radix selection on the Vulkan backend for large k) returns the top
    // `width` indices in no particular order — every consumer treats the selection as a
    // set. When the gather path runs, the row count must be padded to n_sel for the flash
    // attention granularity: repeat the first index; the pad entries are re-masked to
    // -inf in build_attn_qsa_gather, so duplicates are never visible.
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    if (n_sel > width) {
        ggml_tensor * first = ggml_view_3d(ctx0, top_k, 1, top_k->ne[1], top_k->ne[2],
                top_k->nb[1], top_k->nb[2], 0);
        ggml_tensor * pad = ggml_repeat_4d(ctx0, first, n_sel - width, top_k->ne[1], top_k->ne[2], 1);
        top_k = ggml_concat(ctx0, top_k, pad, 0);
    }

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, n_sel, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// [TAG_QSA_GATHER] ported from EngramHalo.cpp (github.com/Aristo94/EngramHalo.cpp).
// Decides whether build_attn_qsa may gather the selected rows instead of masking all n_kv cells.
// Returns the padded row count for the gathered attention, or 0 to keep the dense path.
int64_t llama_model_qwen4exp::graph::qsa_gather_n_sel(int64_t n_kv, int64_t width) const {
    // opt-out / tuning: LLAMA_QSA_GATHER=0 disables the gather, an integer sets the
    // minimum n_kv it activates at; "1" forces it on for testing. Default 32768: on RADV
    // the dense masked path is strong enough that the gather only wins past ~32k
    // (measured: -1% at 16k, tie at 32k, +2% at 64k, growing with depth)
    static const int64_t min_kv = [] {
        const char * env = getenv("LLAMA_QSA_GATHER");
        if (env == nullptr) {
            return (int64_t) 32768;
        }
        const int64_t v = atoll(env);
        return v <= 0 ? INT64_MAX : v;
    }();

    // the gather relies on flash attention: a non-transposed V cache and an f16 mask.
    // alibi encodes distances in the mask; the gathered mask stays value-faithful, but the
    // dense reference path is the only one exercised with it, so do not diverge from it here.
    if (!cparams.flash_attn || hparams.use_alibi) {
        return 0;
    }

    // decode-sized batches only: a prefill ubatch amortizes the dense pass over many queries,
    // while the gather cost scales with n_tokens (each token gathers its own n_sel rows)
    if (n_tokens > 16 || n_kv < min_kv) {
        return 0;
    }

    // FATTN_KQ_STRIDE: the fastest kernels need the KV length padded to 256
    const int64_t n_sel = GGML_PAD(width, 256);

    // nothing to gain unless the gather actually shrinks the attended range
    if (n_sel >= n_kv) {
        return 0;
    }

    return n_sel;
}

// [TAG_QSA_GATHER] ported from EngramHalo.cpp.
// Decode fast path for QSA: instead of unmasking the top-k cells inside a dense [n_kv] attention,
// gather exactly those K/V rows out of the cache and attend over [n_sel] cells. KV bandwidth and
// attention work per token drop from O(n_kv) to O(n_sel). The result matches the dense path:
//   - the rows come from the same top_k tensor; entries [width, n_sel) only pad the row count
//     to the flash attention granularity and are re-masked below
//   - every gathered row keeps its original visibility: mask_sel[j] = kq_mask[top_k[j]]
// Each token attends to its own selection, so the tokens ride in ne3 with one query per slice.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_gather(
        ggml_tensor * k,        // [n_embd_head_k, n_head_kv, n_kv, ns] view of the K cache
        ggml_tensor * v,        // [n_embd_head_v, n_head_kv, n_kv, ns] view of the V cache
        ggml_tensor * kq_mask,  // F16 [n_kv, n_tps, 1, ns]
        ggml_tensor * q_cur,    // F32 [n_embd_head_k, n_head, n_tokens]
        ggml_tensor * top_k,    // I32 [n_sel, n_tps, 1, ns] cell indices, per stream
        int64_t       width,    // leading entries of top_k that carry the reference selection
        float         kq_scale,
        int           il) {
    const int64_t d_k   = k->ne[0];
    const int64_t d_v   = v->ne[0];
    const int64_t hkv   = k->ne[1];
    const int64_t n_kv  = k->ne[2];
    const int64_t ns    = k->ne[3];

    const int64_t n_sel = top_k->ne[0];
    const int64_t n_tps = top_k->ne[1];
    const int64_t nt    = n_tps*ns;

    GGML_ASSERT(top_k->ne[3] == ns);
    GGML_ASSERT(nt == q_cur->ne[2]);
    GGML_ASSERT(width <= n_sel && n_sel <= n_kv);
    GGML_ASSERT(v->nb[1] <= v->nb[2] && "QSA gather needs a non-transposed V cache");
    GGML_ASSERT(kq_mask->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(top_k));
    GGML_ASSERT(ggml_is_contiguous(q_cur));

    // every stream's indices in one flat list; get_rows picks stream s's cells for row list s
    ggml_tensor * idx = ggml_view_2d(ctx0, top_k, n_sel*n_tps, ns, top_k->nb[3], 0);

    // a cell is one contiguous row of d*hkv values in the cache, so fold the head dim away
    ggml_tensor * k_rows = ggml_view_3d(ctx0, k, d_k*hkv, n_kv, ns, k->nb[2], k->nb[3], 0);
    ggml_tensor * v_rows = ggml_view_3d(ctx0, v, d_v*hkv, n_kv, ns, v->nb[2], v->nb[3], 0);

    // note: get_rows always returns F32; a fused quant->F16 gather would halve this intermediate
    ggml_tensor * k_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, k_rows, idx), GGML_TYPE_F16);
    ggml_tensor * v_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, v_rows, idx), GGML_TYPE_F16);
    cb(k_sel, "qsa_k_sel", il);
    cb(v_sel, "qsa_v_sel", il);

    // [d*hkv, n_sel*n_tps, ns] -> [d, n_sel, hkv, nt]: token t of stream s is ne3 slice s*n_tps + t,
    // the same order the ubatch lays its tokens out in
    ggml_tensor * k_g = ggml_view_4d(ctx0, k_sel, d_k, n_sel, hkv, nt,
            ggml_row_size(k_sel->type, d_k*hkv),
            ggml_row_size(k_sel->type, d_k),
            ggml_row_size(k_sel->type, d_k*hkv*n_sel), 0);
    ggml_tensor * v_g = ggml_view_4d(ctx0, v_sel, d_v, n_sel, hkv, nt,
            ggml_row_size(v_sel->type, d_v*hkv),
            ggml_row_size(v_sel->type, d_v),
            ggml_row_size(v_sel->type, d_v*hkv*n_sel), 0);

    // per-row visibility: gather each selected cell's original mask value (rows of size 1)
    ggml_tensor * idx_w  = ggml_view_3d(ctx0, top_k, width, n_tps, ns, top_k->nb[1], top_k->nb[3], 0);
    ggml_tensor * m_sel  = ggml_get_rows(ctx0,
            ggml_reshape_4d(ctx0, kq_mask, 1, n_kv, n_tps, ns), idx_w);

    // the entries past `width` only pad n_sel for the kernel: mask them back out
    if (width < n_sel) {
        ggml_tensor * m_pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_sel - width, n_tps, ns);
        m_pad = ggml_fill(ctx0, m_pad, -INFINITY);
        m_sel = ggml_concat(ctx0, m_sel, m_pad, 1);
    }

    ggml_tensor * m = ggml_cast(ctx0, m_sel, GGML_TYPE_F16);
    m = ggml_reshape_4d(ctx0, m, n_sel, 1, 1, nt);
    cb(m, "qsa_kq_mask_sel", il);

    // one query per ne3 slice against that token's own n_sel rows
    ggml_tensor * q = ggml_reshape_4d(ctx0, q_cur, d_k, 1, q_cur->ne[1], nt);

    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q, k_g, v_g, m, kq_scale,
            hparams.f_max_alibi_bias,
            hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
    res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

    // [d_v, n_head, 1, nt] -> [d_v*n_head, n_tokens], token order unchanged
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    ggml_build_forward_expand(gf, cur);

    return cur;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // [TAG_QSA_GATHER] decode fast path: attend over the gathered top-k rows instead of
    // masking all n_kv cells
    {
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);

        const int64_t n_kv  = k->ne[2];
        const int64_t r     = hparams.dsv4_compress_ratios[il];
        const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

        // build_qsa_top_k took the same decision, so top_k already has n_sel entries
        const int64_t n_sel = qsa_gather_n_sel(n_kv, width);
        if (n_sel > 0) {
            GGML_ASSERT(top_k->ne[0] == n_sel);

            ggml_tensor * cur = build_attn_qsa_gather(k, v, kq_mask, q_cur, top_k, width, kq_scale, il);
            cb(cur, "kqv_out", il);

            // the rotation is its own inverse, so undo it on the value side of the output
            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    // offset 0: delta-net history first, PLE history (if any) after it
    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, 0, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);



    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_memory_hybrid_idx_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        return rows->ne[0] == (int64_t) (pmodel.hparams.ple_n_heads * params.ubatch.n_tokens);
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the token history lives on the memory, so it is per context and part of the state blob
    const llama_memory_hybrid_idx_context * mctx;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // An image is decoded as an embeddings-only batch, so ubatch->token is null and the
    // placeholder ids are not available. The hash must still give every position a row,
    // because this input feeds ggml_get_rows. Stand in the configured image token id, as
    // the reference hashes the placeholder, or EOS if the file has no such key.
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    std::vector<int32_t> idx(n_heads * n_tokens);

    // missing predecessors come from the per-sequence history, but only when it is
    // contiguous with the incoming position; otherwise the window is EOS-padded
    GGML_ASSERT(mctx != nullptr);

    // snapshot the history first, so a token cannot read an earlier token of this same ubatch
    // the snapshot is always n_gram - 1 long and EOS-padded at the front: prev() puts the most recent token last
    std::unordered_map<llama_seq_id, std::vector<llama_token>> snap;
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq = ubatch->seq_id[i][0];
        if (snap.count(seq)) {
            continue;
        }
        auto & h = mctx->get_ple_hist(seq);
        if (h.next_pos != ubatch->pos[i]) {
            h.next_pos = ubatch->pos[i];
            h.toks.clear();
        }
        // keep extra history beyond the hash window so ring rollbacks can truncate losslessly
        const int64_t keep = std::max<int64_t>(n_gram - 1, (int64_t) mctx->get_ple_hist_keep());
        if ((int64_t) h.toks.size() > keep) {
            h.toks.erase(h.toks.begin(), h.toks.end() - keep);
        }

        // the hash window is the most recent n_gram - 1 of the (possibly deeper) history
        const int64_t n_win = std::min<int64_t>((int64_t) h.toks.size(), n_gram - 1);

        std::vector<llama_token> padded(n_gram - 1, (llama_token) eos);
        std::copy(h.toks.end() - n_win, h.toks.end(), padded.end() - n_win);
        snap[seq] = std::move(padded);
    }

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq = ubatch->seq_id[i][0];
        const llama_pos    pos = ubatch->pos[i];

        const auto & hist = snap[seq];

        // predecessor s (1-based) of this token, EOS past a segment boundary
        auto prev = [&](int64_t s) -> int64_t {
            const int64_t j = i - s;
            if (j >= 0 && ubatch->seq_id[j][0] == seq && ubatch->pos[j] == pos - s) {
                return tok_of(j);
            }
            // s - i positions before this ubatch started, most recent last
            const int64_t back = s - i;
            const int64_t k    = (int64_t) hist.size() - back;
            if (back > 0 && k >= 0 && k < (int64_t) hist.size() && pos - s >= 0) {
                return hist[k];
            }
            return eos;
        };

        // an EOS in the window resets everything at or before it
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            ctx[s] = cut ? eos : prev(s);
            if (ctx[s] == eos) {
                cut = true;
            }
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }

        auto & h = mctx->get_ple_hist(seq);
        h.toks.push_back(tok_of(i));
        // keep extra history beyond the hash window so ring rollbacks can truncate losslessly
        const int64_t keep = std::max<int64_t>(n_gram - 1, (int64_t) mctx->get_ple_hist_keep());
        if ((int64_t) h.toks.size() > keep) {
            h.toks.erase(h.toks.begin(), h.toks.end() - keep);
        }
        h.next_pos = pos + 1;
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read one conv history from the recurrent row at row_offset and write the new tail back.
// The shared build_conv_state cannot do this: the row holds the delta-net history and the PLE one.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int64_t              row_offset,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = hparams.n_embd_r();

    // the gather needs the whole row, then this convolution takes its slice
    auto it = rs_rows.find(il);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(il, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    const size_t esz = ggml_element_size(rows);

    ggml_tensor * state = ggml_cont(ctx0,
            ggml_view_2d(ctx0, rows, state_cols * channels, n_seqs,
                    rows->nb[1], row_offset * esz));
    state = ggml_reshape_3d(ctx0, state, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    // cont the transposed activations first: concat's non-contiguous path costs ~13ms
    // per layer on 2048-token prefill chunks (measured), the cont+contiguous concat is
    // an order of magnitude cheaper
    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 0);

    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);

    const auto mem_size = mctx_cur->get_size();

    if (cparams.n_rs_seq == 0) {
        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                kv_head * row_size + row_offset * ggml_element_size(conv_states_all));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // mirror llm_build_delta_net_base::build_conv: with the rollback ring enabled the
        // cache holds (1 + n_rs_seq) banks of mem_size cells, and the state as of each of
        // the last K = n_rs_seq + 1 tokens must be written to its bank (snapshot slot
        // K - t -> rollback group K - t). without this, banks 1..K stay zeroed and every
        // rollback feeds the GDN and PLE convolutions a zeroed history, corrupting
        // generation (found via the greedy spec-on == spec-off identity oracle).
        // split_equal() keeps the trailing K tokens of a sequence in one ubatch, so the
        // slices below are always available.
        const int64_t K = (int64_t) cparams.n_rs_seq + 1;

        for (int64_t t = 1; t <= K; ++t) {
            const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - state_cols - K + t);
            const int64_t s_slot = K - t;

            ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                    state_cols, channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                    state_cols * channels, n_seqs,
                    conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size + row_offset * ggml_element_size(conv_states_all));

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
        }
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *        hidden,
        int                  il) {
    GGML_UNUSED(inp);

    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t n_heads = hparams.ple_n_heads;

    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb);

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // Depthwise causal conv dilated by the n-gram size, as a sum of shifted copies, because
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_r_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim,
            hparams.n_embd_r() - hparams.ple_conv_state(), il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
