#pragma once

// Fabric SDK — Generic Modules
//
// Pre-built neural network modules using the generic Context.
// No LLM dependencies — only requires nn.h.
//
//   auto conv1 = fabric::Conv2d::load(m, "conv1", {.padding=1});
//   auto y = conv1(ctx, x);       // operator() inherited from Module
//   auto y = conv1.forward(ctx, x); // also works
//

#include "nn.h"
#include "model.h"

namespace fabric {

// --- ConvParams ---

struct ConvParams {
    int stride   = 1;
    int padding  = 0;
    int dilation = 1;
};

// --- Conv2d ---

struct Conv2d : Module<Conv2d> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    int stride  = 1;
    int padding = 0;
    int dilation = 1;

    Conv2d() = default;
    Conv2d(ggml_tensor * w, ggml_tensor * b, int s = 1, int p = 0, int d = 1)
        : weight(w), bias(b), stride(s), padding(p), dilation(d) {}
    Conv2d(ggml_tensor * w, ggml_tensor * b, ConvParams p)
        : weight(w), bias(b), stride(p.stride), padding(p.padding), dilation(p.dilation) {}

    Tensor forward(Context & ctx, Tensor x) const {
        Tensor out = ctx.conv2d(x, wrap_w(ctx), stride, stride, padding, padding, dilation, dilation);
        if (bias) {
            Tensor b = ctx.wrap(bias);
            out = out + ctx.reshape_4d(b, 1, 1, b.dim(0), 1);
        }
        return out;
    }

    static Conv2d load(Model & m, const std::string & prefix, ConvParams p = {}) {
        return Conv2d(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"), p);
    }

private:
    Tensor wrap_w(Context & ctx) const { return ctx.wrap(weight); }
};

// --- Conv1d ---

struct Conv1d : Module<Conv1d> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    int stride   = 1;
    int padding  = 0;
    int dilation = 1;

    Conv1d() = default;
    Conv1d(ggml_tensor * w, ggml_tensor * b, int s = 1, int p = 0, int d = 1)
        : weight(w), bias(b), stride(s), padding(p), dilation(d) {}
    Conv1d(ggml_tensor * w, ggml_tensor * b, ConvParams p)
        : weight(w), bias(b), stride(p.stride), padding(p.padding), dilation(p.dilation) {}

    Tensor forward(Context & ctx, Tensor x) const {
        Tensor out = ctx.conv1d(x, ctx.wrap(weight), stride, padding, dilation);
        if (bias) {
            Tensor b = ctx.wrap(bias);
            out = out + ctx.reshape_3d(b, 1, b.dim(0), 1);
        }
        return out;
    }

    static Conv1d load(Model & m, const std::string & prefix, ConvParams p = {}) {
        return Conv1d(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"), p);
    }
};

// --- GenericLinear (no LoRA) ---

struct GenericLinear : Module<GenericLinear> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;

    GenericLinear() = default;
    GenericLinear(ggml_tensor * w, ggml_tensor * b = nullptr) : weight(w), bias(b) {}

    Tensor forward(Context & ctx, Tensor x) const {
        Tensor out = ctx.matmul(ctx.wrap(weight), x);
        if (bias) {
            out = out + ctx.wrap(bias);
        }
        return out;
    }

    static GenericLinear load(Model & m, const std::string & prefix) {
        return GenericLinear(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"));
    }
};

// --- GenericLayerNorm ---

struct GenericLayerNorm : Module<GenericLayerNorm> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    float eps = 1e-5f;

    GenericLayerNorm() = default;
    GenericLayerNorm(ggml_tensor * w, ggml_tensor * b = nullptr, float e = 1e-5f)
        : weight(w), bias(b), eps(e) {}

    Tensor forward(Context & ctx, Tensor x) const {
        return ctx.layer_norm(x, ctx.wrap(weight), ctx.wrap(bias), eps);
    }

    static GenericLayerNorm load(Model & m, const std::string & prefix, float eps = 1e-5f) {
        return GenericLayerNorm(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"), eps);
    }
};

// --- GenericRMSNorm ---

struct GenericRMSNorm : Module<GenericRMSNorm> {
    ggml_tensor * weight = nullptr;
    float eps = 1e-5f;

    GenericRMSNorm() = default;
    GenericRMSNorm(ggml_tensor * w, float e = 1e-5f) : weight(w), eps(e) {}

    Tensor forward(Context & ctx, Tensor x) const {
        return ctx.rms_norm(x, ctx.wrap(weight), eps);
    }

    static GenericRMSNorm load(Model & m, const std::string & prefix, float eps = 1e-5f) {
        return GenericRMSNorm(m.require(prefix + ".weight"), eps);
    }
};

// --- ConvTranspose2d ---

struct ConvTranspose2d : Module<ConvTranspose2d> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    int stride = 1;

    ConvTranspose2d() = default;
    ConvTranspose2d(ggml_tensor * w, ggml_tensor * b, int s = 1) : weight(w), bias(b), stride(s) {}

    Tensor forward(Context & ctx, Tensor x) const {
        Tensor out = ctx.conv_transpose_2d(x, ctx.wrap(weight), stride);
        if (out.dtype() != GGML_TYPE_F32) {
            out = ctx.cast(out, GGML_TYPE_F32);
        }
        if (bias) {
            Tensor b = ctx.wrap(bias);
            out = out + ctx.reshape_4d(b, 1, 1, b.dim(0), 1);
        }
        return out;
    }

    static ConvTranspose2d load(Model & m, const std::string & prefix, int stride = 1) {
        return ConvTranspose2d(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"), stride);
    }
};

// --- Conv2dDW (depthwise convolution) ---

struct Conv2dDW : Module<Conv2dDW> {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    int stride   = 1;
    int padding  = 0;
    int dilation = 1;

    Conv2dDW() = default;
    Conv2dDW(ggml_tensor * w, ggml_tensor * b, ConvParams p)
        : weight(w), bias(b), stride(p.stride), padding(p.padding), dilation(p.dilation) {}

    Tensor forward(Context & ctx, Tensor x) const {
        Tensor out = ctx.conv2d_dw(x, ctx.wrap(weight), stride, stride, padding, padding, dilation, dilation);
        if (bias) {
            Tensor b = ctx.wrap(bias);
            out = out + ctx.reshape_4d(b, 1, 1, b.dim(0), 1);
        }
        return out;
    }

    static Conv2dDW load(Model & m, const std::string & prefix, ConvParams p = {}) {
        return Conv2dDW(m.require(prefix + ".weight"), m.tensor(prefix + ".bias"), p);
    }
};

// --- Dropout (identity at inference) ---

struct Dropout : Module<Dropout> {
    Tensor forward(Context & /*ctx*/, Tensor x) const {
        return x;
    }
};

} // namespace fabric
