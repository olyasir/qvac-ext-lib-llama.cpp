#pragma once

// Fabric SDK — Generic Context
//
// PyTorch-like computation context backed by ggml. No LLM dependencies.
//
//   fabric::Context ctx(ggml_ctx, graph);
//   auto y = ctx.relu(ctx.conv2d(x, w, 1, 1, 1, 1));
//

#include "ggml.h"
#include "tensor.h"

#include <initializer_list>
#include <cstddef>

namespace fabric {

struct Context {
    ggml_context * ctx0;
    ggml_cgraph  * gf;

    Context(ggml_context * c, ggml_cgraph * g) : ctx0(c), gf(g) {}

    Tensor wrap(ggml_tensor * t) { return Tensor(t, ctx0); }

    // --- Tensor creation / I/O ---

    Tensor new_input(const char * name, ggml_type type, std::initializer_list<int64_t> shape) {
        ggml_tensor * t = nullptr;
        auto it = shape.begin();
        size_t n = shape.size();
        if (n == 1) {
            t = ggml_new_tensor_1d(ctx0, type, it[0]);
        } else if (n == 2) {
            t = ggml_new_tensor_2d(ctx0, type, it[0], it[1]);
        } else if (n == 3) {
            t = ggml_new_tensor_3d(ctx0, type, it[0], it[1], it[2]);
        } else {
            t = ggml_new_tensor_4d(ctx0, type, it[0], it[1], it[2], n > 3 ? it[3] : 1);
        }
        ggml_set_name(t, name);
        ggml_set_input(t);
        return wrap(t);
    }

    void finalize(Tensor output) {
        ggml_build_forward_expand(gf, output.ptr);
    }

    // --- Convolution ---

    Tensor conv1d(Tensor x, Tensor w, int s0 = 1, int p0 = 0, int d0 = 1) {
        return wrap(ggml_conv_1d(ctx0, w.ptr, x.ptr, s0, p0, d0));
    }

    Tensor conv2d(Tensor x, Tensor w, int s0, int s1, int p0, int p1, int d0 = 1, int d1 = 1) {
        return wrap(ggml_conv_2d_direct(ctx0, w.ptr, x.ptr, s0, s1, p0, p1, d0, d1));
    }

    Tensor conv2d_dw(Tensor x, Tensor w, int s0, int s1, int p0, int p1, int d0 = 1, int d1 = 1) {
        if (w.dtype() != GGML_TYPE_F16) w = cast(w, GGML_TYPE_F16);
        return wrap(ggml_conv_2d_dw(ctx0, w.ptr, x.ptr, s0, s1, p0, p1, d0, d1));
    }

    Tensor conv_transpose_1d(Tensor x, Tensor w, int s0 = 1, int p0 = 0, int d0 = 1) {
        return wrap(ggml_conv_transpose_1d(ctx0, w.ptr, x.ptr, s0, p0, d0));
    }

    Tensor conv_transpose_2d(Tensor x, Tensor w, int stride) {
        if (w.dtype() != GGML_TYPE_F16) w = cast(w, GGML_TYPE_F16);
        return wrap(ggml_conv_transpose_2d_p0(ctx0, w.ptr, x.ptr, stride));
    }

    // --- Pooling ---

    Tensor max_pool2d(Tensor x, int k, int s, int p = 0) {
        return wrap(ggml_pool_2d(ctx0, x.ptr, GGML_OP_POOL_MAX, k, k, s, s, (float)p, (float)p));
    }

    // Asymmetric pooling: separate kernel/stride/padding per dimension
    // Note: ggml ne[0]=W, ne[1]=H, so k0/s0/p0 = width, k1/s1/p1 = height
    Tensor max_pool2d(Tensor x, int k0, int k1, int s0, int s1, int p0, int p1) {
        return wrap(ggml_pool_2d(ctx0, x.ptr, GGML_OP_POOL_MAX, k0, k1, s0, s1, (float)p0, (float)p1));
    }

    Tensor avg_pool2d(Tensor x, int k, int s, int p = 0) {
        return wrap(ggml_pool_2d(ctx0, x.ptr, GGML_OP_POOL_AVG, k, k, s, s, (float)p, (float)p));
    }

    // Global average pool: reduce spatial dims to 1x1
    Tensor adaptive_avg_pool2d(Tensor x) {
        int w = (int)x.ptr->ne[0];
        int h = (int)x.ptr->ne[1];
        return wrap(ggml_pool_2d(ctx0, x.ptr, GGML_OP_POOL_AVG, w, h, w, h, 0.0f, 0.0f));
    }

    Tensor max_pool1d(Tensor x, int k, int s, int p = 0) {
        return wrap(ggml_pool_1d(ctx0, x.ptr, GGML_OP_POOL_MAX, k, s, p));
    }

    Tensor avg_pool1d(Tensor x, int k, int s, int p = 0) {
        return wrap(ggml_pool_1d(ctx0, x.ptr, GGML_OP_POOL_AVG, k, s, p));
    }

    // --- Activations ---

    Tensor relu(Tensor x)    { return wrap(ggml_relu(ctx0, x.ptr)); }
    Tensor gelu(Tensor x)    { return wrap(ggml_gelu(ctx0, x.ptr)); }
    Tensor sigmoid(Tensor x) { return wrap(ggml_sigmoid(ctx0, x.ptr)); }
    Tensor silu(Tensor x)    { return wrap(ggml_silu(ctx0, x.ptr)); }
    Tensor tanh_(Tensor x)   { return wrap(ggml_tanh(ctx0, x.ptr)); }

    Tensor leaky_relu(Tensor x, float negative_slope = 0.01f) {
        return wrap(ggml_leaky_relu(ctx0, x.ptr, negative_slope, false));
    }

    // --- Normalization ---

    Tensor layer_norm(Tensor x, Tensor w, Tensor b, float eps = 1e-5f) {
        ggml_tensor * n = ggml_norm(ctx0, x.ptr, eps);
        n = ggml_mul(ctx0, n, w.ptr);
        if (b.ptr) {
            n = ggml_add(ctx0, n, b.ptr);
        }
        return wrap(n);
    }

    Tensor rms_norm(Tensor x, float eps = 1e-5f) {
        return wrap(ggml_rms_norm(ctx0, x.ptr, eps));
    }

    Tensor rms_norm(Tensor x, Tensor w, float eps = 1e-5f) {
        ggml_tensor * n = ggml_rms_norm(ctx0, x.ptr, eps);
        n = ggml_mul(ctx0, n, w.ptr);
        return wrap(n);
    }

    Tensor group_norm(Tensor x, int n_groups, float eps = 1e-5f) {
        return wrap(ggml_group_norm(ctx0, x.ptr, n_groups, eps));
    }

    // --- Linear algebra ---

    Tensor matmul(Tensor w, Tensor x) {
        return wrap(ggml_mul_mat(ctx0, w.ptr, x.ptr));
    }

    // --- Element-wise ---

    Tensor scale(Tensor x, float s) {
        return wrap(ggml_scale(ctx0, x.ptr, s));
    }

    Tensor clamp(Tensor x, float min_val, float max_val) {
        return wrap(ggml_clamp(ctx0, x.ptr, min_val, max_val));
    }

    Tensor sqrt_(Tensor x) { return wrap(ggml_sqrt(ctx0, x.ptr)); }
    Tensor exp_(Tensor x)  { return wrap(ggml_exp(ctx0, x.ptr)); }

    // --- Reshape / View ---

    Tensor reshape_1d(Tensor t, int64_t ne0) {
        return wrap(ggml_reshape_1d(ctx0, t.ptr, ne0));
    }

    Tensor reshape_2d(Tensor t, int64_t ne0, int64_t ne1) {
        return wrap(ggml_reshape_2d(ctx0, t.ptr, ne0, ne1));
    }

    Tensor reshape_3d(Tensor t, int64_t ne0, int64_t ne1, int64_t ne2) {
        return wrap(ggml_reshape_3d(ctx0, t.ptr, ne0, ne1, ne2));
    }

    Tensor reshape_4d(Tensor t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return wrap(ggml_reshape_4d(ctx0, t.ptr, ne0, ne1, ne2, ne3));
    }

    Tensor view_1d(Tensor t, int64_t ne0, size_t offset = 0) {
        return wrap(ggml_view_1d(ctx0, t.ptr, ne0, offset));
    }

    Tensor view_2d(Tensor t, int64_t ne0, int64_t ne1, size_t nb1, size_t offset = 0) {
        return wrap(ggml_view_2d(ctx0, t.ptr, ne0, ne1, nb1, offset));
    }

    Tensor permute(Tensor t, int axis0, int axis1, int axis2, int axis3) {
        return wrap(ggml_permute(ctx0, t.ptr, axis0, axis1, axis2, axis3));
    }

    Tensor transpose(Tensor t) {
        return wrap(ggml_transpose(ctx0, t.ptr));
    }

    Tensor cont(Tensor t) {
        return wrap(ggml_cont(ctx0, t.ptr));
    }

    Tensor flatten(Tensor t) {
        int64_t n = 1;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            n *= t.ptr->ne[i];
        }
        return wrap(ggml_reshape_1d(ctx0, t.ptr, n));
    }

    // --- Other ---

    Tensor concat(Tensor a, Tensor b, int dim = 0) {
        return wrap(ggml_concat(ctx0, a.ptr, b.ptr, dim));
    }

    Tensor pad(Tensor t, int p0, int p1, int p2 = 0, int p3 = 0) {
        return wrap(ggml_pad(ctx0, t.ptr, p0, p1, p2, p3));
    }

    Tensor upscale_nearest(Tensor t, int scale_factor) {
        return wrap(ggml_upscale(ctx0, t.ptr, scale_factor, GGML_SCALE_MODE_NEAREST));
    }

    // Resize spatial dims to target (ne0=W, ne1=H), preserving channels and batch
    Tensor interpolate(Tensor t, int64_t ne0, int64_t ne1, uint32_t mode = GGML_SCALE_MODE_NEAREST) {
        return wrap(ggml_interpolate(ctx0, t.ptr, ne0, ne1, t.ptr->ne[2], t.ptr->ne[3], mode));
    }

    // Softmax along first dimension
    Tensor softmax(Tensor x) {
        return wrap(ggml_soft_max(ctx0, x.ptr));
    }

    Tensor cast(Tensor t, ggml_type type) {
        return wrap(ggml_cast(ctx0, t.ptr, type));
    }

    Tensor get_rows(Tensor t, Tensor ids) {
        return wrap(ggml_get_rows(ctx0, t.ptr, ids.ptr));
    }

    // --- Debug / inspection ---

    Tensor name(Tensor t, const char * label) {
        ggml_set_name(t.ptr, label);
        return t;
    }

    Tensor trace(Tensor t, const char * label = nullptr) {
        if (label) fprintf(stderr, "[trace] %-30s ", label);
        t.print(stderr);
        return t;
    }
};

} // namespace fabric
