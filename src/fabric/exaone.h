#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

// ============================================================
// Exaone — Like InternLM2 but with per-layer rope_factors
// ============================================================

struct ExaoneMLP {
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    ExaoneMLP() = default;
    ExaoneMLP(const llama_layer & l) : gate(l.ffn_gate), up(l.ffn_up), down(l.ffn_down) {}
    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up, gate, down, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
};

struct ExaoneAttention {
    Linear q_proj, k_proj, v_proj;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * o_proj_b = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    ExaoneAttention() = default;
    ExaoneAttention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq, l.bq), k_proj(l.wk, l.bk), v_proj(l.wv, l.bv)
        , o_proj(l.wo), o_proj_b(l.bo)
        , head_dim(hd), num_heads(nh), num_kv_heads(nhkv) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv,
                   ggml_tensor * rope_factors, int il) const {
        Tensor Q = q_proj.forward(ctx, x); ctx.name(Q, "Qcur", il);
        Tensor K = k_proj.forward(ctx, x); ctx.name(K, "Kcur", il);
        Tensor V = v_proj.forward(ctx, x); ctx.name(V, "Vcur", il);
        Q = ctx.reshape_3d(Q, head_dim, num_heads,    ctx.n_tokens());
        K = ctx.reshape_3d(K, head_dim, num_kv_heads, ctx.n_tokens());
        V = ctx.reshape_3d(V, head_dim, num_kv_heads, ctx.n_tokens());
        Q = ctx.rope(Q, pos, rope_factors); K = ctx.rope(K, pos, rope_factors);
        ctx.name(Q, "Qcur", il); ctx.name(K, "Kcur", il); ctx.name(V, "Vcur", il);
        return ctx.attn(kv, o_proj, o_proj_b, Q, K, V, 1.0f/sqrtf(float(head_dim)), il);
    }
};

struct ExaoneDecoderLayer {
    ExaoneAttention self_attn;
    ExaoneMLP mlp;
    RMSNorm attn_norm, ffn_norm;

    ExaoneDecoderLayer() = default;
    ExaoneDecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv), mlp(l)
        , attn_norm(l.attn_norm), ffn_norm(l.ffn_norm) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   ggml_tensor * rope_factors, int il, bool is_last) const {
        Tensor residual = x;
        x = attn_norm.forward(ctx, x, il); ctx.name(x, "attn_norm", il);
        x = self_attn.forward(ctx, x, pos, kv, rope_factors, il);
        if (is_last && out_ids) { x = ctx.get_rows(x, out_ids); residual = ctx.get_rows(residual, out_ids); }
        x = x + residual; ctx.name(x, "ffn_inp", il);
        residual = x;
        x = ffn_norm.forward(ctx, x, il); ctx.name(x, "ffn_norm", il);
        x = mlp.forward(ctx, x, il); ctx.name(x, "ffn_out", il);
        x = x + residual;
        x = ctx.cvec(x, il); ctx.name(x, "l_out", il);
        return x;
    }
};

struct ExaoneModel {
    const llama_model * mdl = nullptr;
    ggml_tensor * embed_tokens = nullptr;
    std::vector<ExaoneDecoderLayer> layers;
    RMSNorm norm;

    ExaoneModel() = default;
    ExaoneModel(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) : mdl(&m) {
        embed_tokens = m.tok_embd; norm = RMSNorm(m.output_norm);
        layers.reserve(m.layers.size());
        for (size_t i = 0; i < m.layers.size(); ++i)
            layers.emplace_back(m.layers[i], hd, nh, nhkv);
    }

    Tensor forward(GraphContext & ctx) const {
        Tensor x = ctx.inp_embd(embed_tokens);
        Tensor pos = ctx.inp_pos();
        auto * kv = ctx.inp_attn_kv();
        Tensor out_ids = ctx.inp_out_ids();
        for (int il = 0; il < (int)layers.size(); ++il) {
            ggml_tensor * rope_factors = ctx.get_rope_factors(*mdl, il);
            x = layers[il].forward(ctx, x, pos, kv, out_ids, rope_factors, il, il == (int)layers.size()-1);
        }
        x = norm.forward(ctx, x, -1); ctx.name(x, "result_norm", -1);
        return x;
    }
};

struct ExaoneForCausalLM {
    ExaoneModel model;
    ggml_tensor * lm_head = nullptr;

    ExaoneForCausalLM() = default;
    ExaoneForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
        : model(m, hd, nh, nhkv), lm_head(m.output) {}

    void forward(GraphContext & ctx) const {
        Tensor x = model.forward(ctx);
        ctx.res()->t_embd = x;
        Tensor logits = ctx.lora_mm(lm_head, x);
        ctx.name(logits, "result_output", -1);
        ctx.res()->t_logits = logits;
        ctx.finalize(logits);
    }
};

} // namespace fabric
