#pragma once

// MobileNetV3-Large backbone for text detection (DBNet).
//
// Features: depthwise separable convolution, squeeze-and-excitation blocks,
// hard-swish activation, inverted residual blocks.
//
// BatchNorm is folded into conv at export time.
// Returns 4 feature maps at strides {4, 8, 16, 32} for FPN.
//
// Input:  (3, 1024, 1024)  — ggml shape (1024, 1024, 3, 1)
// Output: 4 feature maps with channels {24, 40, 112, 960}

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"

namespace ocr {

// hard_sigmoid(x) = clamp(x + 3, 0, 6) / 6
inline fabric::Tensor hard_sigmoid(fabric::Context & ctx, fabric::Tensor x) {
    return (x + 3.0f).clamp(0.0f, 6.0f) * (1.0f / 6.0f);
}

// hard_swish(x) = x * hard_sigmoid(x)
inline fabric::Tensor hard_swish(fabric::Context & ctx, fabric::Tensor x) {
    return x * hard_sigmoid(ctx, x);
}

// Squeeze-and-Excitation block
struct SE : fabric::Module<SE> {
    fabric::GenericLinear fc1;  // reduce
    fabric::GenericLinear fc2;  // expand

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Global average pool to (C, 1, 1) then flatten to (C,)
        // x shape: (W, H, C, N)
        int64_t c = x.dim(2);
        fabric::Tensor scale = ctx.adaptive_avg_pool2d(x);  // (1, 1, C, N)
        scale = ctx.reshape_1d(scale, c);                   // (C,)

        scale = fc1(ctx, scale).relu();
        scale = hard_sigmoid(ctx, fc2(ctx, scale));

        // Broadcast multiply: reshape scale to (1, 1, C, 1)
        scale = ctx.reshape_4d(scale, 1, 1, c, 1);
        return x * scale;
    }

    static SE load(fabric::Model & m, const std::string & prefix) {
        SE se;
        se.fc1 = fabric::GenericLinear::load(m, prefix + ".0");
        se.fc2 = fabric::GenericLinear::load(m, prefix + ".1");
        return se;
    }
};

// Inverted Residual Block (MBConv / bneck)
struct InvertedResidual : fabric::Module<InvertedResidual> {
    // Expand phase: 1x1 conv (optional, skipped if expand_ratio == 1)
    fabric::Conv2d expand_conv;
    bool has_expand = false;

    // Depthwise phase: kxk depthwise conv
    fabric::Conv2dDW dw_conv;

    // Squeeze-and-Excitation (optional)
    SE se;
    bool has_se = false;

    // Project phase: 1x1 conv
    fabric::Conv2d project_conv;

    // Activation: true = hard_swish, false = relu
    bool use_hs = false;

    // Residual connection: only when stride=1 and in_ch == out_ch
    bool use_residual = false;

    fabric::Tensor activate(fabric::Context & ctx, fabric::Tensor x) const {
        return use_hs ? hard_swish(ctx, x) : x.relu();
    }

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        fabric::Tensor residual = x;

        // Expand
        if (has_expand) {
            x = activate(ctx, expand_conv(ctx, x));
        }

        // Depthwise
        x = activate(ctx, dw_conv(ctx, x));

        // Squeeze-and-Excitation
        if (has_se) {
            x = se(ctx, x);
        }

        // Project (no activation)
        x = project_conv(ctx, x);

        // Residual
        if (use_residual) {
            x = x + residual;
        }

        return x;
    }

    static InvertedResidual load(fabric::Model & m, const std::string & prefix,
                                  int in_ch, int exp_ch, int out_ch,
                                  int kernel, int stride, bool use_se, bool use_hs_) {
        InvertedResidual block;
        block.use_hs = use_hs_;
        block.use_residual = (stride == 1 && in_ch == out_ch);
        block.has_se = use_se;

        int pad = kernel / 2;
        int idx = 0;

        // Expand conv (only if expansion changes channels)
        if (exp_ch != in_ch) {
            block.has_expand = true;
            block.expand_conv = fabric::Conv2d::load(m, prefix + "." + std::to_string(idx));
            idx++;
        }

        // Depthwise conv
        block.dw_conv = fabric::Conv2dDW::load(m, prefix + "." + std::to_string(idx),
                                                {.stride = stride, .padding = pad});
        idx++;

        // SE block
        if (use_se) {
            block.se = SE::load(m, prefix + "." + std::to_string(idx));
            idx++;
        }

        // Project conv
        block.project_conv = fabric::Conv2d::load(m, prefix + "." + std::to_string(idx));

        return block;
    }
};

// MobileNetV3-Large backbone
// Returns 4 feature maps for FPN: {C2, C3, C4, C5}
struct MobileNetV3Large : fabric::Module<MobileNetV3Large> {
    // Stem: Conv 3->16, stride 2
    fabric::Conv2d stem;

    // 15 inverted residual blocks
    InvertedResidual bneck[15];

    // Final conv: 160 -> 960
    fabric::Conv2d final_conv;

    // Feature extraction points for FPN:
    // After bneck[2]:  24ch, stride 4  -> C2
    // After bneck[5]:  40ch, stride 8  -> C3
    // After bneck[11]: 112ch, stride 16 -> C4
    // After final_conv: 960ch, stride 32 -> C5

    struct Features {
        fabric::Tensor c2, c3, c4, c5;
    };

    Features extract_features(fabric::Context & ctx, fabric::Tensor x) const {
        Features feat;

        // Stem
        x = hard_swish(ctx, stem(ctx, x));  // stride 2 -> H/2

        // bneck 0: 16->16, k3, exp16, RE, s1
        x = bneck[0](ctx, x);

        // bneck 1-2: 16->24, 24->24, stride 2 then 1 -> H/4
        x = bneck[1](ctx, x);
        x = bneck[2](ctx, x);
        feat.c2 = x;  // 24ch at stride 4

        // bneck 3-5: 24->40, 40->40, 40->40, stride 2 then 1,1 -> H/8
        x = bneck[3](ctx, x);
        x = bneck[4](ctx, x);
        x = bneck[5](ctx, x);
        feat.c3 = x;  // 40ch at stride 8

        // bneck 6-11: various, stride 2 then 1s -> H/16
        x = bneck[6](ctx, x);
        x = bneck[7](ctx, x);
        x = bneck[8](ctx, x);
        x = bneck[9](ctx, x);
        x = bneck[10](ctx, x);
        x = bneck[11](ctx, x);
        feat.c4 = x;  // 112ch at stride 16

        // bneck 12-14: 112->160, stride 2 then 1s -> H/32
        x = bneck[12](ctx, x);
        x = bneck[13](ctx, x);
        x = bneck[14](ctx, x);

        // Final 1x1 conv
        x = hard_swish(ctx, final_conv(ctx, x));
        feat.c5 = x;  // 960ch at stride 32

        return feat;
    }

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        auto feat = extract_features(ctx, x);
        return feat.c5;
    }

    static MobileNetV3Large load(fabric::Model & m, const std::string & prefix) {
        MobileNetV3Large net;

        // Stem conv
        net.stem = fabric::Conv2d::load(m, prefix + ".0", {.stride = 2, .padding = 1});

        // MobileNetV3-Large bneck configuration:
        // idx: in_ch, exp_ch, out_ch, kernel, stride, SE, activation(HS=hardswish, RE=relu)
        struct BneckConfig {
            int in_ch, exp_ch, out_ch, kernel, stride;
            bool se, hs;
        };

        static const BneckConfig configs[15] = {
            { 16,  16,  16, 3, 1, false, false}, //  0
            { 16,  64,  24, 3, 2, false, false}, //  1
            { 24,  72,  24, 3, 1, false, false}, //  2
            { 24,  72,  40, 5, 2, true,  false}, //  3
            { 40, 120,  40, 5, 1, true,  false}, //  4
            { 40, 120,  40, 5, 1, true,  false}, //  5
            { 40, 240,  80, 3, 2, false, true }, //  6
            { 80, 200,  80, 3, 1, false, true }, //  7
            { 80, 184,  80, 3, 1, false, true }, //  8
            { 80, 184,  80, 3, 1, false, true }, //  9
            { 80, 480, 112, 3, 1, true,  true }, // 10
            {112, 672, 112, 3, 1, true,  true }, // 11
            {112, 672, 160, 5, 2, true,  true }, // 12
            {160, 960, 160, 5, 1, true,  true }, // 13
            {160, 960, 160, 5, 1, true,  true }, // 14
        };

        for (int i = 0; i < 15; i++) {
            auto & c = configs[i];
            std::string bp = prefix + ".1." + std::to_string(i);
            net.bneck[i] = InvertedResidual::load(m, bp,
                c.in_ch, c.exp_ch, c.out_ch, c.kernel, c.stride, c.se, c.hs);
        }

        // Final conv: 160 -> 960, 1x1
        net.final_conv = fabric::Conv2d::load(m, prefix + ".2");

        return net;
    }
};

} // namespace ocr
