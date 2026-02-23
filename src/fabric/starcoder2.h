#pragma once
#include "fabric.h"
#include <cmath>

namespace fabric {

struct Starcoder2MLP {
    ggml_tensor * up = nullptr, * up_b = nullptr;
    ggml_tensor * down = nullptr, * down_b = nullptr;
    Starcoder2MLP() = default;
    Starcoder2MLP(const llama_layer & l)
        : up(l.ffn_up), up_b(l.ffn_up_b), down(l.ffn_down), down_b(l.ffn_down_b) {}
    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, up, up_b, nullptr, nullptr, down, down_b,
                       LLM_FFN_GELU, LLM_FFN_SEQ, il);
    }
};

struct Starcoder2Attention {
    Linear q_proj, k_proj, v_proj;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * o_proj_b = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    Starcoder2Attention() = default;
    Starcoder2Attention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq, l.bq), k_proj(l.wk, l.bk), v_proj(l.wv, l.bv)
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
        Q = ctx.rope(Q, pos); K = ctx.rope(K, pos);
        ctx.name(Q, "Qcur", il); ctx.name(K, "Kcur", il); ctx.name(V, "Vcur", il);
        return ctx.attn(kv, o_proj, o_proj_b, Q, K, V, 1.0f/sqrtf(float(head_dim)), il);
    }
};

struct Starcoder2DecoderLayer {
    Starcoder2Attention self_attn;
    Starcoder2MLP mlp;
    LayerNorm attn_norm, ffn_norm;

    Starcoder2DecoderLayer() = default;
    Starcoder2DecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv), mlp(l)
        , attn_norm(l.attn_norm, l.attn_norm_b), ffn_norm(l.ffn_norm, l.ffn_norm_b) {}

    Tensor forward(GraphContext & ctx, Tensor x, Tensor pos,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   int il, bool is_last) const {
        Tensor residual = x;
        x = attn_norm.forward(ctx, x, il); ctx.name(x, "attn_norm", il);
        x = self_attn.forward(ctx, x, pos, kv, il);
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

struct Starcoder2Model {
    ggml_tensor * embed_tokens = nullptr;
    std::vector<Starcoder2DecoderLayer> layers;
    LayerNorm norm;

    Starcoder2Model() = default;
    Starcoder2Model(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) {
        embed_tokens = m.tok_embd;
        norm = LayerNorm(m.output_norm, m.output_norm_b);
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

struct Starcoder2ForCausalLM {
    Starcoder2Model model;
    ggml_tensor * lm_head = nullptr;

    Starcoder2ForCausalLM() = default;
    Starcoder2ForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
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
