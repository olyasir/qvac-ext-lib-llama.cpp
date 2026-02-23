// XGLM — Self-registering external architecture using the Fabric SDK
//
// This is the pattern for adding new architectures to llama.cpp:
//
//   1. Define your model in a header using the Fabric DSL  (xglm.h)
//   2. Create a registration file that wires it up          (this file)
//
// That's it. No need to modify llama-arch.h, llama-arch.cpp,
// llama-model.cpp, or any other llama.cpp source file.

#include "xglm.h"

#include <llama-registry.h>
#include <llama-model.h>
#include <llama-hparams.h>
#include <llama-model-loader.h>

// ---------------------------------------------------------------------------
// Graph builder — constructs the computation graph using the Fabric DSL
// ---------------------------------------------------------------------------

struct fabric_build_xglm : public llm_graph_context {
    fabric_build_xglm(
            const llama_model & model,
            const llm_graph_params & params)
        : llm_graph_context(params)
    {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        fabric::GraphContext ctx(*this);
        fabric::XGLMForCausalLM xglm(model, n_embd_head, n_head, n_head_kv);
        xglm(ctx);
    }
};

// ---------------------------------------------------------------------------
// Custom hparams loader (optional — reads layernorm epsilon)
// ---------------------------------------------------------------------------

static void xglm_load_hparams(llama_model_loader & ml, llama_hparams & hparams) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
}

// ---------------------------------------------------------------------------
// Architecture registration — runs at static initialization, before main()
// ---------------------------------------------------------------------------

static llama_arch_info xglm_info() {
    llama_arch_info info;
    info.tensor_names = {
        { LLM_TENSOR_TOKEN_EMBD,  "token_embd" },
        { LLM_TENSOR_POS_EMBD,    "position_embd" },
        { LLM_TENSOR_OUTPUT_NORM,  "output_norm" },
        { LLM_TENSOR_OUTPUT,       "output" },
        { LLM_TENSOR_ATTN_NORM,    "blk.%d.attn_norm" },
        { LLM_TENSOR_ATTN_Q,       "blk.%d.attn_q" },
        { LLM_TENSOR_ATTN_K,       "blk.%d.attn_k" },
        { LLM_TENSOR_ATTN_V,       "blk.%d.attn_v" },
        { LLM_TENSOR_ATTN_OUT,     "blk.%d.attn_output" },
        { LLM_TENSOR_FFN_NORM,     "blk.%d.ffn_norm" },
        { LLM_TENSOR_FFN_UP,       "blk.%d.ffn_up" },
        { LLM_TENSOR_FFN_DOWN,     "blk.%d.ffn_down" },
    };
    info.rope_type    = LLAMA_ROPE_TYPE_NONE;
    info.build_graph  = [](const llama_model & m, const llm_graph_params & p)
                            -> std::unique_ptr<llm_graph_context> {
        return std::make_unique<fabric_build_xglm>(m, p);
    };
    info.load_hparams = xglm_load_hparams;
    return info;
}

static llama_arch_registration xglm_reg("xglm", xglm_info());
