#pragma once

// Feature Pyramid Network (FPN) for multi-scale feature fusion.
//
// Takes 4 feature maps from backbone (C2..C5), applies lateral 1x1 convs
// to project to common channel count, top-down pathway with upsampling,
// and output 3x3 convs. All outputs upsampled to C2 spatial size and
// concatenated along channel dimension.
//
// Output: 4 * out_channels (e.g. 4*64=256 for DocTR)

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"

namespace ocr {

struct FPN : fabric::Module<FPN> {
    // Lateral 1x1 convs (reduce backbone channels to fpn_channels)
    fabric::Conv2d lateral2, lateral3, lateral4, lateral5;

    // Output 3x3 convs (smooth after addition)
    fabric::Conv2d output2, output3, output4, output5;

    int out_channels = 64;  // per-level output channels

    // 4-way forward: takes C2, C3, C4, C5 feature maps
    fabric::Tensor forward_features(fabric::Context & ctx,
                                     fabric::Tensor c2, fabric::Tensor c3,
                                     fabric::Tensor c4, fabric::Tensor c5) const {
        // Lateral projections (Conv+BN+ReLU — BN folded at export, ReLU applied here)
        fabric::Tensor l5 = lateral5(ctx, c5).relu();
        fabric::Tensor l4 = lateral4(ctx, c4).relu();
        fabric::Tensor l3 = lateral3(ctx, c3).relu();
        fabric::Tensor l2 = lateral2(ctx, c2).relu();

        // Top-down pathway: upsample + add
        // l4 += upsample(l5) to match l4 spatial dims
        fabric::Tensor p4 = l4 + ctx.interpolate(l5, l4.dim(0), l4.dim(1));
        fabric::Tensor p3 = l3 + ctx.interpolate(p4, l3.dim(0), l3.dim(1));
        fabric::Tensor p2 = l2 + ctx.interpolate(p3, l2.dim(0), l2.dim(1));

        // Output convolutions (Conv+BN+ReLU — BN folded at export, ReLU applied here)
        fabric::Tensor o5 = output5(ctx, l5).relu();
        fabric::Tensor o4 = output4(ctx, p4).relu();
        fabric::Tensor o3 = output3(ctx, p3).relu();
        fabric::Tensor o2 = output2(ctx, p2).relu();

        // Upsample all to C2 spatial size
        int64_t tw = o2.dim(0);
        int64_t th = o2.dim(1);
        o3 = ctx.interpolate(o3, tw, th);
        o4 = ctx.interpolate(o4, tw, th);
        o5 = ctx.interpolate(o5, tw, th);

        // Concatenate along channel dimension (dim=2 in ggml WHCN)
        fabric::Tensor out = ctx.concat(o2, o3, 2);
        out = ctx.concat(out, o4, 2);
        out = ctx.concat(out, o5, 2);

        return out;  // (W, H, 4*out_channels, 1)
    }

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Not used directly — use forward_features
        return x;
    }

    static FPN load(fabric::Model & m, const std::string & prefix,
                    int c2_ch, int c3_ch, int c4_ch, int c5_ch,
                    int fpn_ch = 128, int out_ch = 64) {
        FPN fpn;
        fpn.out_channels = out_ch;

        // Lateral 1x1 convs (no padding needed)
        fpn.lateral2 = fabric::Conv2d::load(m, prefix + ".lateral2");
        fpn.lateral3 = fabric::Conv2d::load(m, prefix + ".lateral3");
        fpn.lateral4 = fabric::Conv2d::load(m, prefix + ".lateral4");
        fpn.lateral5 = fabric::Conv2d::load(m, prefix + ".lateral5");

        // Output 3x3 convs with padding=1
        fabric::ConvParams p1 = {.stride = 1, .padding = 1};
        fpn.output2 = fabric::Conv2d::load(m, prefix + ".output2", p1);
        fpn.output3 = fabric::Conv2d::load(m, prefix + ".output3", p1);
        fpn.output4 = fabric::Conv2d::load(m, prefix + ".output4", p1);
        fpn.output5 = fabric::Conv2d::load(m, prefix + ".output5", p1);

        return fpn;
    }
};

} // namespace ocr
