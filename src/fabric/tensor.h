#pragma once
#include "ggml.h"

#include <cstdio>
#include <ostream>
#include <utility>

namespace fabric {

// --- Module base (PyTorch nn.Module pattern) ---
// Define forward(), get operator() for free.
//
//   struct MyLayer : fabric::Module<MyLayer> {
//       Tensor forward(Context & ctx, Tensor x) const { ... }
//   };
//   layer.forward(ctx, x);  // explicit
//   layer(ctx, x);          // callable — delegates to forward()
//
template<typename Derived>
struct Module {
    template<typename... Args>
    auto operator()(Args&&... args) const {
        return static_cast<const Derived*>(this)->forward(std::forward<Args>(args)...);
    }
};

struct GraphContext; // forward decl

struct Tensor {
    ggml_tensor  * ptr  = nullptr;
    ggml_context * ctx_ = nullptr;  // for operators (+, -, *, /)
    GraphContext * gctx = nullptr;  // optional: LLM-specific

    Tensor() = default;
    explicit Tensor(ggml_tensor * p, ggml_context * c) : ptr(p), ctx_(c) {}
    explicit Tensor(ggml_tensor * p, GraphContext * g);  // sets both ctx_ and gctx

    operator ggml_tensor*() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }

    int64_t dim(int i) const { return ptr->ne[i]; }

    // --- Introspection (1b) ---

    ggml_type    dtype()     const { return ptr->type; }
    int64_t      numel()     const { return ggml_nelements(ptr); }
    size_t       nbytes()    const { return ggml_nbytes(ptr); }
    int          ndim()      const { return ggml_n_dims(ptr); }
    size_t       stride(int i) const { return ptr->nb[i]; }
    const char * name()      const { return ptr->name; }

    // --- Scalar arithmetic (1a) ---

    Tensor operator*(float s) const;
    Tensor operator+(float s) const;
    Tensor operator-(float s) const;
    Tensor operator/(float s) const;

    // --- Member methods: reshape / view (1c) ---

    Tensor reshape(int64_t ne0) const;
    Tensor reshape(int64_t ne0, int64_t ne1) const;
    Tensor reshape(int64_t ne0, int64_t ne1, int64_t ne2) const;
    Tensor reshape(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) const;
    Tensor flatten() const;
    Tensor contiguous() const;
    Tensor permute(int ax0, int ax1, int ax2, int ax3) const;
    Tensor transpose() const;

    // --- Member methods: activations (1c) ---

    Tensor relu() const;
    Tensor gelu() const;
    Tensor sigmoid() const;
    Tensor silu() const;
    Tensor tanh_() const;
    Tensor leaky_relu(float negative_slope = 0.01f) const;

    // --- Member methods: element-wise (1c) ---

    Tensor scale(float s) const;
    Tensor clamp(float min_val, float max_val) const;
    Tensor sqrt_() const;
    Tensor exp_() const;

    // --- Naming ---

    Tensor & set_name(const char * label) {
        ggml_set_name(ptr, label);
        return *this;
    }

    // --- Debug printing (1e) ---

    void print(FILE * f = stderr) const;
};

// --- Tensor-Tensor operators ---

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

// --- Commutative scalar operators ---

inline Tensor operator*(float s, const Tensor& t) { return t * s; }
inline Tensor operator+(float s, const Tensor& t) { return t + s; }

// --- Free-function activations (1d) ---

inline Tensor relu(const Tensor& x)    { return x.relu(); }
inline Tensor gelu(const Tensor& x)    { return x.gelu(); }
inline Tensor sigmoid(const Tensor& x) { return x.sigmoid(); }
inline Tensor silu(const Tensor& x)    { return x.silu(); }

// --- Debug printing (1e) ---

inline std::ostream & operator<<(std::ostream & os, const Tensor & t) {
    if (!t.ptr) { os << "Tensor[null]"; return os; }
    os << "Tensor[";
    int nd = t.ndim();
    for (int i = 0; i < nd; i++) {
        if (i > 0) os << ", ";
        os << t.dim(i);
    }
    os << "] " << ggml_type_name(t.dtype());
    if (t.name()[0] != '\0') os << " \"" << t.name() << "\"";
    return os;
}

} // namespace fabric
