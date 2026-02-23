#include "tensor.h"
#include "context.h"

namespace fabric {

Tensor::Tensor(ggml_tensor * p, GraphContext * g)
    : ptr(p), ctx_(g ? g->ggml_ctx() : nullptr), gctx(g) {}

// --- Tensor-Tensor operators ---

Tensor operator+(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.ctx_);
    return Tensor(ggml_add(a.ctx_, a.ptr, b.ptr), a.ctx_);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.ctx_);
    return Tensor(ggml_sub(a.ctx_, a.ptr, b.ptr), a.ctx_);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.ctx_);
    return Tensor(ggml_mul(a.ctx_, a.ptr, b.ptr), a.ctx_);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    GGML_ASSERT(a.ctx_);
    return Tensor(ggml_div(a.ctx_, a.ptr, b.ptr), a.ctx_);
}

// --- Scalar arithmetic (1a) ---

Tensor Tensor::operator*(float s) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_scale(ctx_, ptr, s), ctx_);
}

Tensor Tensor::operator+(float s) const {
    GGML_ASSERT(ctx_);
    // scale_bias(a, 1.0, b) = 1.0 * a + b
    return Tensor(ggml_scale_bias(ctx_, ptr, 1.0f, s), ctx_);
}

Tensor Tensor::operator-(float s) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_scale_bias(ctx_, ptr, 1.0f, -s), ctx_);
}

Tensor Tensor::operator/(float s) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_scale(ctx_, ptr, 1.0f / s), ctx_);
}

// --- Member methods: reshape / view (1c) ---

Tensor Tensor::reshape(int64_t ne0) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_reshape_1d(ctx_, ptr, ne0), ctx_);
}

Tensor Tensor::reshape(int64_t ne0, int64_t ne1) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_reshape_2d(ctx_, ptr, ne0, ne1), ctx_);
}

Tensor Tensor::reshape(int64_t ne0, int64_t ne1, int64_t ne2) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_reshape_3d(ctx_, ptr, ne0, ne1, ne2), ctx_);
}

Tensor Tensor::reshape(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_reshape_4d(ctx_, ptr, ne0, ne1, ne2, ne3), ctx_);
}

Tensor Tensor::flatten() const {
    GGML_ASSERT(ctx_);
    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        n *= ptr->ne[i];
    }
    return Tensor(ggml_reshape_1d(ctx_, ptr, n), ctx_);
}

Tensor Tensor::contiguous() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_cont(ctx_, ptr), ctx_);
}

Tensor Tensor::permute(int ax0, int ax1, int ax2, int ax3) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_permute(ctx_, ptr, ax0, ax1, ax2, ax3), ctx_);
}

Tensor Tensor::transpose() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_transpose(ctx_, ptr), ctx_);
}

// --- Member methods: activations (1c) ---

Tensor Tensor::relu() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_relu(ctx_, ptr), ctx_);
}

Tensor Tensor::gelu() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_gelu(ctx_, ptr), ctx_);
}

Tensor Tensor::sigmoid() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_sigmoid(ctx_, ptr), ctx_);
}

Tensor Tensor::silu() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_silu(ctx_, ptr), ctx_);
}

Tensor Tensor::tanh_() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_tanh(ctx_, ptr), ctx_);
}

Tensor Tensor::leaky_relu(float negative_slope) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_leaky_relu(ctx_, ptr, negative_slope, false), ctx_);
}

// --- Member methods: element-wise (1c) ---

Tensor Tensor::scale(float s) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_scale(ctx_, ptr, s), ctx_);
}

Tensor Tensor::clamp(float min_val, float max_val) const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_clamp(ctx_, ptr, min_val, max_val), ctx_);
}

Tensor Tensor::sqrt_() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_sqrt(ctx_, ptr), ctx_);
}

Tensor Tensor::exp_() const {
    GGML_ASSERT(ctx_);
    return Tensor(ggml_exp(ctx_, ptr), ctx_);
}

// --- Debug printing (1e) ---

void Tensor::print(FILE * f) const {
    if (!ptr) {
        fprintf(f, "Tensor[null]\n");
        return;
    }
    int nd = ndim();
    fprintf(f, "Tensor[");
    for (int i = 0; i < nd; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%lld", (long long)dim(i));
    }
    fprintf(f, "] %s", ggml_type_name(dtype()));
    if (name()[0] != '\0') fprintf(f, " \"%s\"", name());
    fprintf(f, "\n");
}

} // namespace fabric
