#pragma once
#include "../llama-graph.h"
#include "../llama-model.h"
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

    // Normalization
    Tensor norm(Tensor cur, ggml_tensor * w, ggml_tensor * b,
                llm_norm_type type, int il) {
        return wrap(llm_ctx.build_norm(cur, w, b, type, il));
    }

    // Linear
    Tensor lora_mm(ggml_tensor * w, Tensor cur) {
        return wrap(llm_ctx.build_lora_mm(w, cur));
    }

    // Reshape / View
    Tensor reshape_3d(Tensor t, int64_t d0, int64_t d1, int64_t d2) {
        return wrap(ggml_reshape_3d(ggml_ctx(), t, d0, d1, d2));
    }

    Tensor view_2d(Tensor t, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
        return wrap(ggml_view_2d(ggml_ctx(), t, ne0, ne1, nb1, offset));
    }

    // Positional encoding
    Tensor rope(Tensor t, Tensor pos, ggml_tensor * factors = nullptr) {
        return wrap(ggml_rope_ext(
            ggml_ctx(), t, pos, factors,
            llm_ctx.n_rot, llm_ctx.rope_type, llm_ctx.n_ctx_orig,
            llm_ctx.freq_base, llm_ctx.freq_scale,
            llm_ctx.ext_factor, llm_ctx.attn_factor,
            llm_ctx.beta_fast, llm_ctx.beta_slow));
    }

    // Get per-layer rope factors from model (used by LLaMA3, Exaone, etc.)
    ggml_tensor * get_rope_factors(const llama_model & model, int il) {
        return model.get_rope_factors(llm_ctx.cparams, il);
    }

    // Raw RMS norm (for QK-norm without learned weights)
    Tensor rms_norm(Tensor t) {
        return wrap(ggml_rms_norm(ggml_ctx(), t, llm_ctx.hparams.f_norm_rms_eps));
    }

    // Attention
    Tensor attn(llm_graph_input_attn_kv * inp,
                ggml_tensor * wo, ggml_tensor * wo_b,
                Tensor Q, Tensor K, Tensor V,
                float kq_scale, int il) {
        return wrap(llm_ctx.build_attn(inp, wo, wo_b, Q, K, V,
                    nullptr, nullptr, nullptr, kq_scale, il));
    }

    // FFN (no biases)
    Tensor ffn(Tensor cur,
               ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down,
               llm_ffn_op_type op, llm_ffn_gate_type gate_type, int il) {
        return wrap(llm_ctx.build_ffn(cur,
                    up, nullptr, nullptr,
                    gate, nullptr, nullptr,
                    down, nullptr, nullptr,
                    nullptr, op, gate_type, il));
    }

    // FFN (with biases)
    Tensor ffn(Tensor cur,
               ggml_tensor * up, ggml_tensor * up_b,
               ggml_tensor * gate, ggml_tensor * gate_b,
               ggml_tensor * down, ggml_tensor * down_b,
               llm_ffn_op_type op, llm_ffn_gate_type gate_type, int il) {
        return wrap(llm_ctx.build_ffn(cur,
                    up, up_b, nullptr,
                    gate, gate_b, nullptr,
                    down, down_b, nullptr,
                    nullptr, op, gate_type, il));
    }

    // MoE FFN
    Tensor moe_ffn(Tensor cur,
                   ggml_tensor * gate_inp,
                   ggml_tensor * up_exps, ggml_tensor * gate_exps, ggml_tensor * down_exps,
                   ggml_tensor * exp_probs_b,
                   int64_t n_exp, int64_t n_exp_used,
                   llm_ffn_op_type op, bool norm_w, bool scale_w, float w_scale,
                   llama_expert_gating_func_type gating_op, int il) {
        return wrap(llm_ctx.build_moe_ffn(cur, gate_inp, up_exps, gate_exps, down_exps,
                    exp_probs_b, n_exp, n_exp_used, op, norm_w, scale_w, w_scale,
                    gating_op, il));
    }

    // Misc operations
    Tensor cvec(Tensor cur, int il) {
        return wrap(llm_ctx.build_cvec(cur, il));
    }

    Tensor get_rows(Tensor t, Tensor ids) {
        return wrap(ggml_get_rows(ggml_ctx(), t, ids));
    }

    Tensor scale(Tensor t, float s) {
        return wrap(ggml_scale(ggml_ctx(), t, s));
    }

    Tensor clamp(Tensor t, float min_val, float max_val) {
        return wrap(ggml_clamp(ggml_ctx(), t, min_val, max_val));
    }

    // Naming / callback
    void name(Tensor cur, const char * label, int il) {
        llm_ctx.cb(cur, label, il);
    }

    void finalize(Tensor cur) {
        ggml_build_forward_expand(graph(), cur);
    }

    // Parameter accessors
    int64_t n_layer()        const { return llm_ctx.n_layer; }
    int64_t n_head()         const { return llm_ctx.n_head; }
    int64_t n_head_kv()      const { return llm_ctx.n_head_kv; }
    int64_t n_tokens()       const { return llm_ctx.n_tokens; }
    int64_t n_embd()         const { return llm_ctx.n_embd; }
    int64_t n_embd_head_k()  const { return llm_ctx.n_embd_head_k; }
    int64_t n_embd_head_v()  const { return llm_ctx.n_embd_head_v; }
    int64_t n_embd_k_gqa()   const { return llm_ctx.n_embd_k_gqa; }
    int64_t n_embd_v_gqa()   const { return llm_ctx.n_embd_v_gqa; }
    int64_t n_rot()          const { return llm_ctx.n_rot; }
    int64_t n_expert()       const { return llm_ctx.n_expert; }
    int64_t n_expert_used()  const { return llm_ctx.n_expert_used; }

    const llama_hparams & hparams() const { return llm_ctx.hparams; }
    const llama_cparams & cparams() const { return llm_ctx.cparams; }

    llm_graph_result * res() { return llm_ctx.res; }
};

} // namespace fabric
