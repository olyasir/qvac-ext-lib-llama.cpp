#pragma once
#include "context.h"
#include "../llama-model.h"

namespace fabric {

struct RMSNorm : Module<RMSNorm> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;

    RMSNorm() = default;
    RMSNorm(ggml_tensor * w, ggml_tensor * b = nullptr) : weight(w), bias(b) {}

    Tensor forward(GraphContext & ctx, Tensor input, int il) const {
        return ctx.norm(input, weight, bias, LLM_NORM_RMS, il);
    }
};

struct LayerNorm : Module<LayerNorm> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;

    LayerNorm() = default;
    LayerNorm(ggml_tensor * w, ggml_tensor * b = nullptr) : weight(w), bias(b) {}

    Tensor forward(GraphContext & ctx, Tensor input, int il) const {
        return ctx.norm(input, weight, bias, LLM_NORM, il);
    }
};

struct Linear : Module<Linear> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;

    Linear() = default;
    Linear(ggml_tensor * w, ggml_tensor * b = nullptr) : weight(w), bias(b) {}

    Tensor forward(GraphContext & ctx, Tensor input) const {
        Tensor out = ctx.lora_mm(weight, input);
        if (bias) {
            out = out + Tensor(bias, &ctx);
        }
        return out;
    }
};

struct FusedQKVLinear : Module<FusedQKVLinear> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    int64_t q_size = 0;
    int64_t k_size = 0;
    int64_t v_size = 0;

    FusedQKVLinear() = default;
    FusedQKVLinear(ggml_tensor * w, ggml_tensor * b,
                   int64_t qs, int64_t ks, int64_t vs)
        : weight(w), bias(b), q_size(qs), k_size(ks), v_size(vs) {}

    struct QKV { Tensor Q, K, V; };

    QKV forward(GraphContext & ctx, Tensor input) const {
        Tensor qkv = ctx.lora_mm(weight, input);
        if (bias) {
            qkv = qkv + Tensor(bias, &ctx);
        }
        const size_t es = ggml_element_size(qkv.ptr);
        QKV result;
        result.Q = ctx.view_2d(qkv, q_size, ctx.n_tokens(), qkv.ptr->nb[1], 0);
        result.K = ctx.view_2d(qkv, k_size, ctx.n_tokens(), qkv.ptr->nb[1], es * q_size);
        result.V = ctx.view_2d(qkv, v_size, ctx.n_tokens(), qkv.ptr->nb[1], es * (q_size + k_size));
        return result;
    }
};

} // namespace fabric
