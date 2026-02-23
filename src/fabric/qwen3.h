#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

// ============================================================
// Qwen3MLP
// Mirrors: transformers.models.qwen3.modeling_qwen3.Qwen3MLP
// ============================================================
struct Qwen3MLP {
    ggml_tensor * gate_proj = nullptr;  // w1
    ggml_tensor * up_proj   = nullptr;  // w3
    ggml_tensor * down_proj = nullptr;  // w2

    Qwen3MLP() = default;
    Qwen3MLP(const llama_layer & layer)
        : gate_proj(layer.ffn_gate)
        , up_proj(layer.ffn_up)
        , down_proj(layer.ffn_down) {}

    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up_proj, gate_proj, down_proj,
                       LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
};

// ============================================================
// Qwen3Attention
// Mirrors: transformers.models.qwen3.modeling_qwen3.Qwen3Attention
// Q/K/V projection -> reshape -> Q/K RMSNorm -> RoPE -> attention
// ============================================================
struct Qwen3Attention {
    Linear  q_proj, k_proj, v_proj;
    RMSNorm q_norm, k_norm;

    ggml_tensor * o_proj   = nullptr;
    ggml_tensor * o_proj_b = nullptr;

    int64_t head_dim     = 0;
    int64_t num_heads    = 0;
    int64_t num_kv_heads = 0;

    Qwen3Attention() = default;
    Qwen3Attention(const llama_layer & layer,
                   int64_t head_dim, int64_t nh, int64_t nh_kv)
        : q_proj(layer.wq)
        , k_proj(layer.wk)
        , v_proj(layer.wv)
        , q_norm(layer.attn_q_norm)
        , k_norm(layer.attn_k_norm)
        , o_proj(layer.wo)
        , o_proj_b(layer.bo)
        , head_dim(head_dim)
        , num_heads(nh)
        , num_kv_heads(nh_kv) {}

    Tensor forward(GraphContext & ctx, Tensor hidden_states,
                   Tensor position, llm_graph_input_attn_kv * kv_cache,
                   int il) const {
        // Project Q, K, V
        Tensor Q = q_proj.forward(ctx, hidden_states);
        ctx.name(Q, "Qcur", il);

        Tensor K = k_proj.forward(ctx, hidden_states);
        ctx.name(K, "Kcur", il);

        Tensor V = v_proj.forward(ctx, hidden_states);
        ctx.name(V, "Vcur", il);

        // Reshape to [head_dim, num_heads, seq_len]
        Q = ctx.reshape_3d(Q, head_dim, num_heads,    ctx.n_tokens());
        K = ctx.reshape_3d(K, head_dim, num_kv_heads, ctx.n_tokens());
        V = ctx.reshape_3d(V, head_dim, num_kv_heads, ctx.n_tokens());

        // Q/K normalization
        Q = q_norm.forward(ctx, Q, il);
        ctx.name(Q, "Qcur_normed", il);

        K = k_norm.forward(ctx, K, il);
        ctx.name(K, "Kcur_normed", il);

        // Rotary position embeddings
        Q = ctx.rope(Q, position);
        K = ctx.rope(K, position);

        ctx.name(Q, "Qcur", il);
        ctx.name(K, "Kcur", il);
        ctx.name(V, "Vcur", il);

        // Scaled dot-product attention with KV cache
        float scale = 1.0f / sqrtf(float(head_dim));
        return ctx.attn(kv_cache, o_proj, o_proj_b, Q, K, V, scale, il);
    }
};

// ============================================================
// Qwen3DecoderLayer
// Mirrors: transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer
// ============================================================
struct Qwen3DecoderLayer {
    Qwen3Attention self_attn;
    Qwen3MLP       mlp;
    RMSNorm        input_layernorm;
    RMSNorm        post_attention_layernorm;

    Qwen3DecoderLayer() = default;
    Qwen3DecoderLayer(const llama_layer & layer,
                      int64_t head_dim, int64_t n_head, int64_t n_head_kv)
        : self_attn(layer, head_dim, n_head, n_head_kv)
        , mlp(layer)
        , input_layernorm(layer.attn_norm)
        , post_attention_layernorm(layer.ffn_norm) {}

    Tensor forward(GraphContext & ctx, Tensor hidden_states,
                   Tensor position, llm_graph_input_attn_kv * kv_cache,
                   Tensor out_ids, int il, bool is_last) const {
        Tensor residual = hidden_states;

        // Self Attention
        hidden_states = input_layernorm.forward(ctx, hidden_states, il);
        ctx.name(hidden_states, "attn_norm", il);

        hidden_states = self_attn.forward(ctx, hidden_states, position, kv_cache, il);

        // Output token selection (last layer optimization)
        if (is_last && out_ids) {
            hidden_states = ctx.get_rows(hidden_states, out_ids);
            residual      = ctx.get_rows(residual, out_ids);
        }

        hidden_states = hidden_states + residual;
        ctx.name(hidden_states, "ffn_inp", il);

        // Feed-forward
        residual = hidden_states;

        hidden_states = post_attention_layernorm.forward(ctx, hidden_states, il);
        ctx.name(hidden_states, "ffn_norm", il);

        hidden_states = mlp.forward(ctx, hidden_states, il);
        ctx.name(hidden_states, "ffn_out", il);

        hidden_states = hidden_states + residual;

        hidden_states = ctx.cvec(hidden_states, il);
        ctx.name(hidden_states, "l_out", il);

        return hidden_states;
    }
};

// ============================================================
// Qwen3Model
// Mirrors: transformers.models.qwen3.modeling_qwen3.Qwen3Model
// ============================================================
struct Qwen3Model {
    ggml_tensor * embed_tokens = nullptr;
    std::vector<Qwen3DecoderLayer> layers;
    RMSNorm norm;

    Qwen3Model() = default;
    Qwen3Model(const llama_model & model,
               int64_t head_dim, int64_t n_head, int64_t n_head_kv) {
        embed_tokens = model.tok_embd;
        norm = RMSNorm(model.output_norm);

        const int n_layer = (int)model.layers.size();
        layers.reserve(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            layers.emplace_back(model.layers[i], head_dim, n_head, n_head_kv);
        }
    }

    Tensor forward(GraphContext & ctx) const {
        // Token embeddings
        Tensor hidden_states = ctx.inp_embd(embed_tokens);

        // Position input
        Tensor position = ctx.inp_pos();

        // KV cache
        auto * kv_cache = ctx.inp_attn_kv();

        // Output token selection
        Tensor out_ids = ctx.inp_out_ids();

        // Decoder layers
        const int n_layer = (int)layers.size();
        for (int il = 0; il < n_layer; ++il) {
            bool is_last = (il == n_layer - 1);
            hidden_states = layers[il].forward(
                ctx, hidden_states, position, kv_cache, out_ids, il, is_last);
        }

        // Final layer norm
        hidden_states = norm.forward(ctx, hidden_states, -1);
        ctx.name(hidden_states, "result_norm", -1);

        return hidden_states;
    }
};

// ============================================================
// Qwen3ForCausalLM
// Mirrors: transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM
// ============================================================
struct Qwen3ForCausalLM {
    Qwen3Model model;
    ggml_tensor * lm_head = nullptr;

    Qwen3ForCausalLM() = default;
    Qwen3ForCausalLM(const llama_model & m,
                     int64_t head_dim, int64_t n_head, int64_t n_head_kv)
        : model(m, head_dim, n_head, n_head_kv)
        , lm_head(m.output) {}

    void forward(GraphContext & ctx) const {
        // Decoder
        Tensor hidden_states = model.forward(ctx);
        ctx.res()->t_embd = hidden_states;

        // Language model head
        Tensor logits = ctx.lora_mm(lm_head, hidden_states);
        ctx.name(logits, "result_output", -1);
        ctx.res()->t_logits = logits;

        ctx.finalize(logits);
    }
};

} // namespace fabric
