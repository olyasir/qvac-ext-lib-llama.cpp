# Fabric C++ DSL — Phase 1: Qwen3 Proof-of-Concept

## Context

QVAC Fabric's goal is to make defining new model architectures in llama.cpp dramatically simpler. Today, adding an architecture requires touching 5-6 files with raw ggml operations and deep C++ expertise. The DSL provides a PyTorch-like C++ API that wraps ggml, so a new architecture can be defined in a single readable file while producing an identical computation graph.

Qwen3 is the first proof-of-concept model. It's a clean standard transformer with one interesting feature (Q/K norm before RoPE) that exercises the DSL's composability.

### Key differences between Qwen3 and Qwen2
- **No bias** on Q, K, V projections (Qwen2 had biases)
- **Q/K RMSNorm** applied after reshape, before RoPE (`attn_q_norm`, `attn_k_norm`)
- **No output bias** (`model.output_b` not used)
- Otherwise identical: RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual

## Approach

The DSL is a **thin C++ wrapper** over `llm_graph_context`. It does NOT replace any infrastructure — it reuses tensor loading, KV cache, LoRA, tensor naming, and all backend dispatch. Only the graph builder (currently `llm_build_qwen3` in `src/models/qwen3.cpp`) gets a DSL equivalent.

## New Files

### 1. `src/fabric/tensor.h` — Tensor wrapper + operators

```cpp
#pragma once
#include "ggml.h"

namespace fabric {

struct GraphContext; // forward decl

struct Tensor {
    ggml_tensor * ptr = nullptr;
    GraphContext * gctx = nullptr;

    Tensor() = default;
    explicit Tensor(ggml_tensor * p, GraphContext * g = nullptr) : ptr(p), gctx(g) {}

    operator ggml_tensor*() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }

    int64_t dim(int i) const { return ptr->ne[i]; }
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

} // namespace fabric
```

### 2. `src/fabric/tensor.cpp` — Operator implementations

```cpp
#include "tensor.h"
#include "context.h"

namespace fabric {

Tensor operator+(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.gctx);
    return Tensor(ggml_add(a.gctx->ggml_ctx(), a.ptr, b.ptr), a.gctx);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.gctx);
    return Tensor(ggml_mul(a.gctx->ggml_ctx(), a.ptr, b.ptr), a.gctx);
}

} // namespace fabric
```

### 3. `src/fabric/context.h` — GraphContext (wraps `llm_graph_context`)

```cpp
#pragma once
#include "../llama-graph.h"
#include "tensor.h"

namespace fabric {

struct GraphContext {
    llm_graph_context & llm_ctx;

    explicit GraphContext(llm_graph_context & ctx) : llm_ctx(ctx) {}

    Tensor wrap(ggml_tensor * t) { return Tensor(t, this); }

    ggml_context * ggml_ctx() const { return llm_ctx.ctx0; }
    ggml_cgraph  * graph()    const { return llm_ctx.gf; }

    // Input builders
    Tensor inp_embd(ggml_tensor * tok_embd) {
        return wrap(llm_ctx.build_inp_embd(tok_embd));
    }
    Tensor inp_pos() { return wrap(llm_ctx.build_inp_pos()); }
    auto * inp_attn_kv() { return llm_ctx.build_attn_inp_kv(); }
    Tensor inp_out_ids() { return wrap(llm_ctx.build_inp_out_ids()); }

    // Operations
    Tensor norm(Tensor cur, ggml_tensor * w, ggml_tensor * b,
                llm_norm_type type, int il) {
        return wrap(llm_ctx.build_norm(cur, w, b, type, il));
    }

    Tensor lora_mm(ggml_tensor * w, Tensor cur) {
        return wrap(llm_ctx.build_lora_mm(w, cur));
    }

    Tensor reshape_3d(Tensor t, int64_t d0, int64_t d1, int64_t d2) {
        return wrap(ggml_reshape_3d(ggml_ctx(), t, d0, d1, d2));
    }

    Tensor rope(Tensor t, Tensor pos, ggml_tensor * factors = nullptr) {
        return wrap(ggml_rope_ext(
            ggml_ctx(), t, pos, factors,
            llm_ctx.n_rot, llm_ctx.rope_type, llm_ctx.n_ctx_orig,
            llm_ctx.freq_base, llm_ctx.freq_scale,
            llm_ctx.ext_factor, llm_ctx.attn_factor,
            llm_ctx.beta_fast, llm_ctx.beta_slow));
    }

    Tensor attn(llm_graph_input_attn_kv * inp,
                ggml_tensor * wo, ggml_tensor * wo_b,
                Tensor Q, Tensor K, Tensor V,
                float kq_scale, int il) {
        return wrap(llm_ctx.build_attn(inp, wo, wo_b, Q, K, V,
                    nullptr, nullptr, nullptr, kq_scale, il));
    }

    Tensor ffn(Tensor cur,
               ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down,
               llm_ffn_op_type op, llm_ffn_gate_type gate_type, int il) {
        return wrap(llm_ctx.build_ffn(cur,
                    up, nullptr, nullptr,
                    gate, nullptr, nullptr,
                    down, nullptr, nullptr,
                    nullptr, op, gate_type, il));
    }

    Tensor cvec(Tensor cur, int il) {
        return wrap(llm_ctx.build_cvec(cur, il));
    }

    Tensor get_rows(Tensor t, Tensor ids) {
        return wrap(ggml_get_rows(ggml_ctx(), t, ids));
    }

    // Naming / callback
    void name(Tensor cur, const char * label, int il) {
        llm_ctx.cb(cur, label, il);
    }

    void finalize(Tensor cur) {
        ggml_build_forward_expand(graph(), cur);
    }

    // Parameter accessors
    int64_t n_layer()    const { return llm_ctx.n_layer; }
    int64_t n_head()     const { return llm_ctx.n_head; }
    int64_t n_head_kv()  const { return llm_ctx.n_head_kv; }
    int64_t n_tokens()   const { return llm_ctx.n_tokens; }

    llm_graph_result * res() { return llm_ctx.res; }
};

} // namespace fabric
```

### 4. `src/fabric/modules.h` — Reusable building blocks (PyTorch-style `forward()`)

```cpp
#pragma once
#include "context.h"
#include "../llama-model.h"

namespace fabric {

struct RMSNorm {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;

    RMSNorm() = default;
    RMSNorm(ggml_tensor * w, ggml_tensor * b = nullptr) : weight(w), bias(b) {}

    Tensor forward(GraphContext & ctx, Tensor input, int il) const {
        return ctx.norm(input, weight, bias, LLM_NORM_RMS, il);
    }
};

struct Linear {
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

} // namespace fabric
```

Architecture-specific modules (like `Qwen3MLP`) are defined in the model's own header, not in `modules.h`.

### 5. `src/fabric/fabric.h` — Single include header

```cpp
#pragma once
#include "tensor.h"
#include "context.h"
#include "modules.h"
```

### 6. `src/fabric/qwen3.h` — Full Qwen3 architecture (HuggingFace style)

Mirrors HuggingFace's `modeling_qwen3.py` class hierarchy: `Qwen3MLP`, `Qwen3Attention`, `Qwen3DecoderLayer`, `Qwen3Model`, `Qwen3ForCausalLM`.

```cpp
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
// Q/K/V projection → reshape → Q/K RMSNorm → RoPE → attention
// ============================================================
struct Qwen3Attention {
    Linear  q_proj, k_proj, v_proj;
    RMSNorm q_norm, k_norm;

    ggml_tensor * o_proj   = nullptr;
    ggml_tensor * o_proj_b = nullptr;

    int64_t head_dim   = 0;
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
```

### 7. `src/fabric/qwen3.cpp` — Graph builder (thin entry point)

```cpp
#include "qwen3.h"
#include "../models/models.h"

fabric_build_qwen3::fabric_build_qwen3(
        const llama_model & model,
        const llm_graph_params & params)
    : llm_graph_context(params)
{
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    fabric::GraphContext ctx(*this);
    fabric::Qwen3ForCausalLM qwen3(model, n_embd_head, n_head, n_head_kv);
    qwen3.forward(ctx);
}
```

## Modified Files

### 8. `src/llama-model.cpp` (~line 7305)

One-line change in `build_graph()`:
```cpp
case LLM_ARCH_QWEN3:
    llm = std::make_unique<fabric_build_qwen3>(*this, params);
```

Everything else unchanged (hparams loading, tensor creation, rope_type, memory type).

### 9. `src/models/models.h`

Add forward declaration:
```cpp
// Fabric DSL builders
struct fabric_build_qwen3 : public llm_graph_context {
    fabric_build_qwen3(const llama_model & model, const llm_graph_params & params);
};
```

### 10. `src/CMakeLists.txt`

Add after the existing model sources:
```cmake
            fabric/tensor.cpp
            fabric/qwen3.cpp
```

## Side-by-Side: Current vs DSL

### Current `src/models/qwen3.cpp` — 117 lines, raw ggml, single monolithic constructor

### DSL `src/fabric/qwen3.cpp` — **10 lines** (thin entry point)
```cpp
fabric::GraphContext ctx(*this);
fabric::Qwen3ForCausalLM qwen3(model, n_embd_head, n_head, n_head_kv);
qwen3.forward(ctx);
```

### DSL `src/fabric/qwen3.h` — HuggingFace-style class hierarchy

| DSL Class | HuggingFace Equivalent | Role |
|-----------|----------------------|------|
| `Qwen3MLP` | `Qwen3MLP` | SwiGLU feed-forward |
| `Qwen3Attention` | `Qwen3Attention` | Q/K/V + Q/K norm + RoPE + attention |
| `Qwen3DecoderLayer` | `Qwen3DecoderLayer` | Norm → Attn → Residual → Norm → FFN → Residual |
| `Qwen3Model` | `Qwen3Model` | Embedding + layers + final norm |
| `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | Full model + lm_head |

**Key wins**:
- Reads like the HuggingFace Python code — ML engineers can understand it immediately
- Each component is reusable — adding LLaMA would reuse `Linear`, `RMSNorm` and only define new Attention/Layer
- The full model is 10 lines in qwen3.cpp; architecture logic is in readable structs with `forward()` methods
- Zero performance overhead — it's the same ggml graph underneath

## Implementation Order

1. Create `src/fabric/` directory
2. Write `tensor.h` + `tensor.cpp` (Tensor wrapper, operators)
3. Write `context.h` (GraphContext wrapping llm_graph_context)
4. Write `modules.h` (RMSNorm, Linear)
5. Write `fabric.h` (unified include)
6. Write `qwen3.h` (Qwen3MLP, Qwen3Attention, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM)
7. Write `qwen3.cpp` (fabric_build_qwen3)
8. Add forward decl to `src/models/models.h`
9. Swap dispatch in `src/llama-model.cpp` (line ~7305)
10. Add sources to `src/CMakeLists.txt`

## Verification

1. **Build**: `cmake --build` must succeed with no warnings
2. **Graph equivalence**: Keep old `llm_build_qwen3` code, build both graphs, compare node count, op types, tensor shapes
3. **Output equivalence**: Run a Qwen3 model through both builders, verify identical logits
4. **Performance**: Benchmark to confirm zero overhead (the DSL is just pointer wrapping)

## Critical Files Reference

| File | Role |
|------|------|
| `src/models/qwen3.cpp` | Current Qwen3 builder (117 lines) — reference for exact graph |
| `src/llama-graph.h:537-690` | `llm_graph_context` — base class we wrap |
| `src/llama-model.h:197-416` | `llama_layer` — tensor pointer storage |
| `src/llama-model.h:425-548` | `llama_model` — model-level tensors |
| `src/llama-model.cpp:1064-1075` | Qwen3 hparams loading (unchanged) |
| `src/llama-model.cpp:3461-3495` | Qwen3 tensor creation (unchanged) |
| `src/llama-model.cpp:7305-7308` | Qwen3 build_graph dispatch (one-line change) |
| `src/llama-arch.cpp:796-813` | Qwen3 tensor name table (unchanged) |
