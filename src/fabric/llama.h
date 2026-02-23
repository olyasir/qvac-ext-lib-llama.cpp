#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

// ============================================================
// LLaMA — The most complete standard transformer
// Features: optional biases, rope_factors, optional QK-norm,
//           MoE FFN branch, configurable attention scale
// ============================================================

struct LlamaMLP {
    ggml_tensor * gate = nullptr, * gate_b = nullptr;
    ggml_tensor * up = nullptr, * up_b = nullptr;
    ggml_tensor * down = nullptr, * down_b = nullptr;

    LlamaMLP() = default;
    LlamaMLP(const llama_layer & l)
        : gate(l.ffn_gate), gate_b(l.ffn_gate_b)
        , up(l.ffn_up), up_b(l.ffn_up_b)
        , down(l.ffn_down), down_b(l.ffn_down_b) {}

    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up, up_b, gate, gate_b, down, down_b,
                       LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
};

struct LlamaMoE {
    ggml_tensor * gate_inp = nullptr;
    ggml_tensor * up_exps = nullptr, * gate_exps = nullptr, * down_exps = nullptr;

    LlamaMoE() = default;
    LlamaMoE(const llama_layer & l)
        : gate_inp(l.ffn_gate_inp)
        , up_exps(l.ffn_up_exps)
        , gate_exps(l.ffn_gate_exps)
        , down_exps(l.ffn_down_exps) {}

    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.moe_ffn(x, gate_inp, up_exps, gate_exps, down_exps,
                           nullptr, ctx.n_expert(), ctx.n_expert_used(),
                           LLM_FFN_SILU, true, false, 0.0,
                           LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    }
};

struct LlamaAttention {
    Linear q_proj, k_proj, v_proj;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * o_proj_b = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    LlamaAttention() = default;
    LlamaAttention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq, l.bq), k_proj(l.wk, l.bk), v_proj(l.wv, l.bv)
        , o_proj(l.wo), o_proj_b(l.bo)
        , head_dim(hd), num_heads(nh), num_kv_heads(nhkv) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv,
                   ggml_tensor * rope_factors,
                   float kq_scale, bool use_kq_norm, int il) const {
        Tensor Q = q_proj.forward(ctx, x); ctx.name(Q, "Qcur", il);
        Tensor K = k_proj.forward(ctx, x); ctx.name(K, "Kcur", il);
        Tensor V = v_proj.forward(ctx, x); ctx.name(V, "Vcur", il);
        Q = ctx.reshape_3d(Q, head_dim, num_heads,    ctx.n_tokens());
        K = ctx.reshape_3d(K, head_dim, num_kv_heads, ctx.n_tokens());
        V = ctx.reshape_3d(V, head_dim, num_kv_heads, ctx.n_tokens());
        Q = ctx.rope(Q, pos, rope_factors); K = ctx.rope(K, pos, rope_factors);
        ctx.name(Q, "Qcur", il); ctx.name(K, "Kcur", il); ctx.name(V, "Vcur", il);
        if (use_kq_norm) {
            Q = ctx.rms_norm(Q); ctx.name(Q, "Qcur_normed", il);
            K = ctx.rms_norm(K); ctx.name(K, "Kcur_normed", il);
        }
        return ctx.attn(kv, o_proj, o_proj_b, Q, K, V, kq_scale, il);
    }
};

struct LlamaDecoderLayer {
    LlamaAttention self_attn;
    LlamaMLP mlp;
    LlamaMoE moe;
    bool has_moe = false;
    RMSNorm attn_norm, ffn_norm;

    LlamaDecoderLayer() = default;
    LlamaDecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv)
        , mlp(l), moe(l)
        , has_moe(l.ffn_gate_inp != nullptr)
        , attn_norm(l.attn_norm), ffn_norm(l.ffn_norm) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   ggml_tensor * rope_factors,
                   float kq_scale, bool use_kq_norm,
                   int il, bool is_last) const {
        Tensor residual = x;
        x = attn_norm.forward(ctx, x, il); ctx.name(x, "attn_norm", il);
        x = self_attn.forward(ctx, x, pos, kv, rope_factors, kq_scale, use_kq_norm, il);
        if (is_last && out_ids) { x = ctx.get_rows(x, out_ids); residual = ctx.get_rows(residual, out_ids); }
        x = x + residual; ctx.name(x, "ffn_inp", il);
        residual = x;
        x = ffn_norm.forward(ctx, x, il); ctx.name(x, "ffn_norm", il);
        if (!has_moe) {
            x = mlp.forward(ctx, x, il); ctx.name(x, "ffn_out", il);
        } else {
            x = moe.forward(ctx, x, il); ctx.name(x, "ffn_moe_out", il);
        }
        x = x + residual;
        x = ctx.cvec(x, il); ctx.name(x, "l_out", il);
        return x;
    }
};

struct LlamaModel {
    const llama_model * mdl = nullptr;
    ggml_tensor * embed_tokens = nullptr;
    std::vector<LlamaDecoderLayer> layers;
    RMSNorm norm;
    float kq_scale = 0.0f;
    bool use_kq_norm = false;

    LlamaModel() = default;
    LlamaModel(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) : mdl(&m) {
        embed_tokens = m.tok_embd; norm = RMSNorm(m.output_norm);
        float f_attn_scale = m.hparams.f_attention_scale;
        kq_scale = (f_attn_scale == 0.0f) ? 1.0f/sqrtf(float(hd)) : f_attn_scale;
        use_kq_norm = m.hparams.use_kq_norm;
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
            x = layers[il].forward(ctx, x, pos, kv, out_ids, rope_factors,
                                   kq_scale, use_kq_norm, il, il == (int)layers.size()-1);
        }
        x = norm.forward(ctx, x, -1); ctx.name(x, "result_norm", -1);
        return x;
    }
};

struct LlamaForCausalLM {
    LlamaModel model;
    ggml_tensor * lm_head = nullptr;

    LlamaForCausalLM() = default;
    LlamaForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
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
