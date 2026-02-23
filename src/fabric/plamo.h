#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

// ============================================================
// PLaMo — Parallel attention + FFN architecture
// Both attention and FFN take the normalized input,
// outputs are summed with residual: out = attn + ffn + residual
// ============================================================

struct PLaMoMLP {
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    PLaMoMLP() = default;
    PLaMoMLP(const llama_layer & l) : gate(l.ffn_gate), up(l.ffn_up), down(l.ffn_down) {}
    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up, gate, down, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
};

struct PLaMoAttention {
    Linear q_proj, k_proj, v_proj;
    ggml_tensor * o_proj = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    PLaMoAttention() = default;
    PLaMoAttention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq), k_proj(l.wk), v_proj(l.wv)
        , o_proj(l.wo), head_dim(hd), num_heads(nh), num_kv_heads(nhkv) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, int il) const {
        Tensor Q = q_proj.forward(ctx, x); ctx.name(Q, "Qcur", il);
        Tensor K = k_proj.forward(ctx, x); ctx.name(K, "Kcur", il);
        Tensor V = v_proj.forward(ctx, x); ctx.name(V, "Vcur", il);
        Q = ctx.reshape_3d(Q, head_dim, num_heads,    ctx.n_tokens());
        K = ctx.reshape_3d(K, head_dim, num_kv_heads, ctx.n_tokens());
        V = ctx.reshape_3d(V, head_dim, num_kv_heads, ctx.n_tokens());
        Q = ctx.rope(Q, pos); K = ctx.rope(K, pos);
        ctx.name(Q, "Qcur", il); ctx.name(K, "Kcur", il); ctx.name(V, "Vcur", il);
        return ctx.attn(kv, o_proj, nullptr, Q, K, V, 1.0f/sqrtf(float(head_dim)), il);
    }
};

struct PLaMoDecoderLayer {
    PLaMoAttention self_attn;
    PLaMoMLP mlp;
    RMSNorm attn_norm;

    PLaMoDecoderLayer() = default;
    PLaMoDecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv), mlp(l)
        , attn_norm(l.attn_norm) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   int il, bool is_last) const {
        // Single normalization shared by attention and FFN
        Tensor normed = attn_norm.forward(ctx, x, il);
        ctx.name(normed, "attn_norm", il);

        // Attention path
        Tensor attn_out = self_attn.forward(ctx, normed, pos, kv, il);

        // Output token selection
        if (is_last && out_ids) {
            attn_out = ctx.get_rows(attn_out, out_ids);
            normed   = ctx.get_rows(normed, out_ids);
            x        = ctx.get_rows(x, out_ids);
        }

        // FFN path (parallel: feeds from same normed input)
        Tensor ffn_out = mlp.forward(ctx, normed, il);
        ctx.name(ffn_out, "ffn_out", il);

        // Combine: attn + ffn + residual
        Tensor out = attn_out + ffn_out;
        out = out + x;

        out = ctx.cvec(out, il);
        ctx.name(out, "l_out", il);
        return out;
    }
};

struct PLaMoModel {
    ggml_tensor * embed_tokens = nullptr;
    std::vector<PLaMoDecoderLayer> layers;
    RMSNorm norm;

    PLaMoModel() = default;
    PLaMoModel(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) {
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
        for (int il = 0; il < (int)layers.size(); ++il)
            x = layers[il].forward(ctx, x, pos, kv, out_ids, il, il == (int)layers.size()-1);
        x = norm.forward(ctx, x, -1); ctx.name(x, "result_norm", -1);
        return x;
    }
};

struct PLaMoForCausalLM {
    PLaMoModel model;
    ggml_tensor * lm_head = nullptr;

    PLaMoForCausalLM() = default;
    PLaMoForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
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
