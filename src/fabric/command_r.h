#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

// ============================================================
// Command-R — Parallel attn+FFN, LayerNorm, optional Q/K norm,
//             logit scaling
// ============================================================

struct CommandRMLP {
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    CommandRMLP() = default;
    CommandRMLP(const llama_layer & l) : gate(l.ffn_gate), up(l.ffn_up), down(l.ffn_down) {}
    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up, gate, down, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
};

struct CommandRAttention {
    Linear q_proj, k_proj, v_proj;
    // Optional Q/K LayerNorm (applied before RoPE)
    ggml_tensor * q_norm_w = nullptr;
    ggml_tensor * k_norm_w = nullptr;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * o_proj_b = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    CommandRAttention() = default;
    CommandRAttention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq, l.bq), k_proj(l.wk, l.bk), v_proj(l.wv, l.bv)
        , q_norm_w(l.attn_q_norm), k_norm_w(l.attn_k_norm)
        , o_proj(l.wo), o_proj_b(l.bo)
        , head_dim(hd), num_heads(nh), num_kv_heads(nhkv) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, int il) const {
        Tensor Q = q_proj.forward(ctx, x); ctx.name(Q, "Qcur", il);
        Tensor K = k_proj.forward(ctx, x); ctx.name(K, "Kcur", il);
        Tensor V = v_proj.forward(ctx, x); ctx.name(V, "Vcur", il);
        Q = ctx.reshape_3d(Q, head_dim, num_heads,    ctx.n_tokens());
        K = ctx.reshape_3d(K, head_dim, num_kv_heads, ctx.n_tokens());
        V = ctx.reshape_3d(V, head_dim, num_kv_heads, ctx.n_tokens());
        // Optional Q/K LayerNorm before RoPE
        if (q_norm_w) {
            Q = ctx.norm(Q, q_norm_w, nullptr, LLM_NORM, il);
            ctx.name(Q, "Qcur", il);
        }
        Q = ctx.rope(Q, pos);
        if (k_norm_w) {
            K = ctx.norm(K, k_norm_w, nullptr, LLM_NORM, il);
            ctx.name(K, "Kcur", il);
        }
        K = ctx.rope(K, pos);
        ctx.name(Q, "Qcur", il); ctx.name(K, "Kcur", il); ctx.name(V, "Vcur", il);
        return ctx.attn(kv, o_proj, o_proj_b, Q, K, V, 1.0f/sqrtf(float(head_dim)), il);
    }
};

struct CommandRDecoderLayer {
    CommandRAttention self_attn;
    CommandRMLP mlp;
    LayerNorm attn_norm;

    CommandRDecoderLayer() = default;
    CommandRDecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv), mlp(l)
        , attn_norm(l.attn_norm) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   int il, bool is_last) const {
        // Single norm feeds both attention and FFN (parallel)
        Tensor normed = attn_norm.forward(ctx, x, il);
        ctx.name(normed, "attn_norm", il);

        Tensor attn_out = self_attn.forward(ctx, normed, pos, kv, il);

        if (is_last && out_ids) {
            attn_out = ctx.get_rows(attn_out, out_ids);
            x        = ctx.get_rows(x, out_ids);
            normed   = ctx.get_rows(normed, out_ids);
        }

        // FFN (parallel: same normed input as attention)
        Tensor ffn_out = mlp.forward(ctx, normed, il);
        ctx.name(ffn_out, "ffn_out", il);

        // residual + attn + ffn
        Tensor out = x + ffn_out;
        out = out + attn_out;

        out = ctx.cvec(out, il);
        ctx.name(out, "l_out", il);
        return out;
    }
};

struct CommandRModel {
    ggml_tensor * embed_tokens = nullptr;
    std::vector<CommandRDecoderLayer> layers;
    LayerNorm norm;

    CommandRModel() = default;
    CommandRModel(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) {
        embed_tokens = m.tok_embd;
        norm = LayerNorm(m.output_norm);
        layers.reserve(m.layers.size());
        for (size_t i = 0; i < m.layers.size(); ++i)
            layers.emplace_back(m.layers[i], hd, nh, nhkv);
    }

    Tensor forward(GraphContext & ctx) const {
        Tensor x = ctx.inp_embd(embed_tokens);
        Tensor pos = ctx.inp_pos();
        auto * kv = ctx.inp_attn_kv();
        Tensor out_ids = ctx.inp_out_ids();
        for (int il = 0; il < (int)layers.size(); ++il)
            x = layers[il].forward(ctx, x, pos, kv, out_ids, il, il == (int)layers.size()-1);
        x = norm.forward(ctx, x, -1); ctx.name(x, "result_norm", -1);
        return x;
    }
};

struct CommandRForCausalLM {
    CommandRModel model;
    ggml_tensor * lm_head = nullptr;
    float logit_scale = 0.0f;

    CommandRForCausalLM() = default;
    CommandRForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
        : model(m, hd, nh, nhkv), lm_head(m.output)
        , logit_scale(m.hparams.f_logit_scale) {}

    void forward(GraphContext & ctx) const {
        Tensor x = model.forward(ctx);
        ctx.res()->t_embd = x;
        Tensor logits = ctx.lora_mm(lm_head, x);
        if (logit_scale != 0.0f) {
            logits = ctx.scale(logits, logit_scale);
        }
        ctx.name(logits, "result_output", -1);
        ctx.res()->t_logits = logits;
        ctx.finalize(logits);
    }
};

} // namespace fabric
