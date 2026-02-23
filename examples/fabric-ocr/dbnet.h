#pragma once

// DBNet (Differentiable Binarization Network) for text detection.
//
// Architecture: MobileNetV3-Large backbone -> FPN -> Probability head
//
// The probability head upsamples to 1/4 input resolution, then the
// post-processor handles final upsampling and thresholding.
//
// Input:  (3, 1024, 1024)  — ggml shape (1024, 1024, 3, 1)
// Output: probability map at (256, 256, 1, 1)

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"
#include "mobilenetv3.h"
#include "fpn.h"

namespace ocr {

struct DBNet : fabric::Module<DBNet> {
    MobileNetV3Large backbone;
    FPN fpn;

    // Probability head
    fabric::Conv2d    head_conv;     // 256->64, 3x3, p=1
    fabric::ConvTranspose2d head_up1;    // 64->64, 2x2, s=2
    fabric::ConvTranspose2d head_up2;    // 64->1, 2x2, s=2

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Extract multi-scale features
        auto feat = backbone.extract_features(ctx, x);

        // FPN fusion -> (W/4, H/4, 256, 1)
        fabric::Tensor fused = fpn.forward_features(ctx, feat.c2, feat.c3, feat.c4, feat.c5);

        // Probability head
        fabric::Tensor prob = head_conv(ctx, fused).relu();
        prob = head_up1(ctx, prob).relu();
        prob = head_up2(ctx, prob).sigmoid();

        return prob;  // probability map
    }

    static DBNet load(fabric::Model & m, const std::string & prefix) {
        DBNet net;

        net.backbone = MobileNetV3Large::load(m, prefix + ".feat_extractor");

        // MobileNetV3-Large feature channels: C2=24, C3=40, C4=112, C5=960
        net.fpn = FPN::load(m, prefix + ".fpn", 24, 40, 112, 960);

        // Probability head
        net.head_conv = fabric::Conv2d::load(m, prefix + ".probability_head.0",
                                              {.stride = 1, .padding = 1});
        net.head_up1 = fabric::ConvTranspose2d::load(m, prefix + ".probability_head.1", /*stride=*/2);
        net.head_up2 = fabric::ConvTranspose2d::load(m, prefix + ".probability_head.2", /*stride=*/2);

        return net;
    }
};

} // namespace ocr
