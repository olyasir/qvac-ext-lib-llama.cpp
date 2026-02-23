#pragma once

// VGG16-BN backbone for text recognition (DocTR variant).
//
// 13 Conv3x3 layers in 5 blocks. BatchNorm is folded into conv at export time.
// Key difference from standard VGG: last 3 max-pools are asymmetric (kH=2,kW=1)
// to preserve the width dimension for the sequence output.
//
// Input:  (3, 32, 128)  — ggml shape (128, 32, 3, 1)
// Output: (512, 1, 32)  — ggml shape (32, 1, 512, 1)

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"

namespace ocr {

struct VGG16BN : fabric::Module<VGG16BN> {
    // Block 1: 2 convs, 64 channels
    fabric::Conv2d conv0, conv1;
    // Block 2: 2 convs, 128 channels
    fabric::Conv2d conv2, conv3;
    // Block 3: 3 convs, 256 channels
    fabric::Conv2d conv4, conv5, conv6;
    // Block 4: 3 convs, 512 channels
    fabric::Conv2d conv7, conv8, conv9;
    // Block 5: 3 convs, 512 channels
    fabric::Conv2d conv10, conv11, conv12;

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Block 1: 64ch, pool (2,2) -> 16x64
        x = conv0(ctx, x).relu();
        x = conv1(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);

        // Block 2: 128ch, pool (2,2) -> 8x32
        x = conv2(ctx, x).relu();
        x = conv3(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);

        // Block 3: 256ch, asymmetric pool (kH=2,kW=1, sH=2,sW=1) -> 4x32
        // ggml: k0=kW=1, k1=kH=2, s0=sW=1, s1=sH=2
        x = conv4(ctx, x).relu();
        x = conv5(ctx, x).relu();
        x = conv6(ctx, x).relu();
        x = ctx.max_pool2d(x, /*k0*/1, /*k1*/2, /*s0*/1, /*s1*/2, /*p0*/0, /*p1*/0);

        // Block 4: 512ch, asymmetric pool -> 2x32
        x = conv7(ctx, x).relu();
        x = conv8(ctx, x).relu();
        x = conv9(ctx, x).relu();
        x = ctx.max_pool2d(x, 1, 2, 1, 2, 0, 0);

        // Block 5: 512ch, asymmetric pool -> 1x32
        x = conv10(ctx, x).relu();
        x = conv11(ctx, x).relu();
        x = conv12(ctx, x).relu();
        x = ctx.max_pool2d(x, 1, 2, 1, 2, 0, 0);

        return x;  // (32, 1, 512, 1) in ggml order
    }

    static VGG16BN load(fabric::Model & m, const std::string & prefix) {
        VGG16BN net;
        fabric::ConvParams p1 = {.stride = 1, .padding = 1};
        net.conv0  = fabric::Conv2d::load(m, prefix + ".0",  p1);
        net.conv1  = fabric::Conv2d::load(m, prefix + ".1",  p1);
        net.conv2  = fabric::Conv2d::load(m, prefix + ".2",  p1);
        net.conv3  = fabric::Conv2d::load(m, prefix + ".3",  p1);
        net.conv4  = fabric::Conv2d::load(m, prefix + ".4",  p1);
        net.conv5  = fabric::Conv2d::load(m, prefix + ".5",  p1);
        net.conv6  = fabric::Conv2d::load(m, prefix + ".6",  p1);
        net.conv7  = fabric::Conv2d::load(m, prefix + ".7",  p1);
        net.conv8  = fabric::Conv2d::load(m, prefix + ".8",  p1);
        net.conv9  = fabric::Conv2d::load(m, prefix + ".9",  p1);
        net.conv10 = fabric::Conv2d::load(m, prefix + ".10", p1);
        net.conv11 = fabric::Conv2d::load(m, prefix + ".11", p1);
        net.conv12 = fabric::Conv2d::load(m, prefix + ".12", p1);
        return net;
    }
};

} // namespace ocr
