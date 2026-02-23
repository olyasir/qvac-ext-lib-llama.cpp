#include "exaone.h"
#include "../models/models.h"

fabric_build_exaone::fabric_build_exaone(
        const llama_model & model,
        const llm_graph_params & params)
    : llm_graph_context(params)
{
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    fabric::GraphContext ctx(*this);
    fabric::ExaoneForCausalLM m(model, n_embd_head, n_head, n_head_kv);
    m.forward(ctx);
}
