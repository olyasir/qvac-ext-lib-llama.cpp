#pragma once
#include <fabric/fabric.h>
#include <cmath>

namespace fabric {

// ============================================================
// XGLM (Meta) — Cross-lingual Generative Language Model
// Decoder-only transformer with sinusoidal positional embeddings,
// pre-norm LayerNorm, GELU FFN, weight-tied lm_head.
// ============================================================

struct XGLMMLP : Module<XGLMMLP> {
    ggml_tensor * fc1   = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2   = nullptr;
    ggml_tensor * fc2_b = nullptr;

    XGLMMLP() = default;
    XGLMMLP(const llama_layer & l)
        : fc1(l.ffn_up), fc1_b(l.ffn_up_b)
        , fc2(l.ffn_down), fc2_b(l.ffn_down_b) {}

    Tensor forward(GraphContext & ctx, Tensor x, int il) const {
        return ctx.ffn(x, fc1, fc1_b, nullptr, nullptr, fc2, fc2_b,
                       LLM_FFN_GELU, LLM_FFN_SEQ, il);
    }
};

struct XGLMAttention : Module<XGLMAttention> {
    Linear q_proj, k_proj, v_proj;
    ggml_tensor * o_proj   = nullptr;
    ggml_tensor * o_proj_b = nullptr;
    int64_t head_dim = 0, num_heads = 0, num_kv_heads = 0;

    XGLMAttention() = default;
    XGLMAttention(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : q_proj(l.wq, l.bq), k_proj(l.wk, l.bk), v_proj(l.wv, l.bv)
        , o_proj(l.wo), o_proj_b(l.bo)
        , head_dim(hd), num_heads(nh), num_kv_heads(nhkv) {}

    Tensor forward(GraphContext & ctx, Tensor x,
                   llm_graph_input_attn_kv * kv, int il) const {
        Tensor Q = q_proj(ctx, x);  ctx.name(Q, "Qcur", il);
        Tensor K = k_proj(ctx, x);  ctx.name(K, "Kcur", il);
        Tensor V = v_proj(ctx, x);  ctx.name(V, "Vcur", il);

        Q = Q.reshape(head_dim, num_heads,    ctx.n_tokens());
        K = K.reshape(head_dim, num_kv_heads, ctx.n_tokens());
        V = V.reshape(head_dim, num_kv_heads, ctx.n_tokens());

        ctx.name(Q, "Qcur", il);
        ctx.name(K, "Kcur", il);
        ctx.name(V, "Vcur", il);

        return ctx.attn(kv, o_proj, o_proj_b, Q, K, V, 1.0f / sqrtf(float(head_dim)), il);
    }
};

struct XGLMDecoderLayer : Module<XGLMDecoderLayer> {
    XGLMAttention self_attn;
    XGLMMLP       mlp;
    LayerNorm      attn_norm, ffn_norm;

    XGLMDecoderLayer() = default;
    XGLMDecoderLayer(const llama_layer & l, int64_t hd, int64_t nh, int64_t nhkv)
        : self_attn(l, hd, nh, nhkv), mlp(l)
        , attn_norm(l.attn_norm, l.attn_norm_b)
        , ffn_norm(l.ffn_norm, l.ffn_norm_b) {}

    Tensor forward(GraphContext & ctx, Tensor x,
                   llm_graph_input_attn_kv * kv, Tensor out_ids,
                   int il, bool is_last) const {
        Tensor residual = x;
        x = self_attn(ctx, attn_norm(ctx, x, il), kv, il);

        if (is_last && out_ids) {
            x        = ctx.get_rows(x, out_ids);
            residual = ctx.get_rows(residual, out_ids);
        }
        x = x + residual;

        residual = x;
        x = mlp(ctx, ffn_norm(ctx, x, il), il);
        x = x + residual;

        return ctx.cvec(x, il);
    }
};

struct XGLMModel : Module<XGLMModel> {
    ggml_tensor * embed_tokens    = nullptr;
    ggml_tensor * embed_positions = nullptr;
    std::vector<XGLMDecoderLayer> layers;
    LayerNorm norm;
    float embed_scale = 1.0f;

    XGLMModel() = default;
    XGLMModel(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv) {
        embed_tokens    = m.tok_embd;
        embed_positions = m.pos_embd;
        norm = LayerNorm(m.output_norm, m.output_norm_b);
        embed_scale = sqrtf(float(m.hparams.n_embd));

        layers.reserve(m.layers.size());
        for (auto & l : m.layers)
            layers.emplace_back(l, hd, nh, nhkv);
    }

    Tensor forward(GraphContext & ctx) const {
        Tensor x   = ctx.inp_embd(embed_tokens) * embed_scale;
        Tensor pos = ctx.get_rows(ctx.wrap(embed_positions), ctx.inp_pos());
        x = x + pos;

        auto * kv     = ctx.inp_attn_kv();
        Tensor outids = ctx.inp_out_ids();

        for (int il = 0; il < (int)layers.size(); ++il)
            x = layers[il](ctx, x, kv, outids, il, il == (int)layers.size() - 1);

        return norm(ctx, x, -1);
    }
};

struct XGLMForCausalLM : Module<XGLMForCausalLM> {
    XGLMModel model;
    ggml_tensor * lm_head = nullptr;

    XGLMForCausalLM() = default;
    XGLMForCausalLM(const llama_model & m, int64_t hd, int64_t nh, int64_t nhkv)
        : model(m, hd, nh, nhkv), lm_head(m.output) {}

    void forward(GraphContext & ctx) const {
        Tensor x = model(ctx);
        ctx.res()->t_embd = x;

        Tensor logits = ctx.lora_mm(lm_head, x);
        ctx.res()->t_logits = logits;
        ctx.finalize(logits);
    }
};

} // namespace fabric
