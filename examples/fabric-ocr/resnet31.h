#pragma once

// ResNet31 backbone for text recognition (swappable alternative to VGG16-BN).
//
// Uses BasicBlock (two Conv3x3) with optional 1x1 downsample projection.
// Asymmetric pooling in later stages preserves width for sequence output.
//
// Input:  (3, 32, 128)  — ggml shape (128, 32, 3, 1)
// Output: (512, 4, 32)  — ggml shape (32, 4, 512, 1)
//
// Note: Produces 4-height output vs VGG16's 1-height, so flatten to
// (512*4=2048, 32) before LSTM or use an additional pool.

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"

namespace ocr {

struct BasicBlock : fabric::Module<BasicBlock> {
    fabric::Conv2d conv1;       // 3x3
    fabric::Conv2d conv2;       // 3x3
    fabric::Conv2d downsample;  // optional 1x1 projection

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        fabric::Tensor residual = x;

        x = conv1(ctx, x).relu();
        x = conv2(ctx, x);

        if (downsample.weight) {
            residual = downsample(ctx, residual);
        }

        return (x + residual).relu();
    }

    static BasicBlock load(fabric::Model & m, const std::string & prefix,
                            bool has_downsample = false) {
        BasicBlock blk;
        fabric::ConvParams p1 = {.stride = 1, .padding = 1};
        blk.conv1 = fabric::Conv2d::load(m, prefix + ".conv1", p1);
        blk.conv2 = fabric::Conv2d::load(m, prefix + ".conv2", p1);
        if (has_downsample) {
            blk.downsample = fabric::Conv2d::load(m, prefix + ".downsample");
        }
        return blk;
    }
};

struct ResNet31 : fabric::Module<ResNet31> {
    // Stem: two convolutions
    fabric::Conv2d stem_conv1;   // 3->64, 3x3, p=1
    fabric::Conv2d stem_conv2;   // 64->128, 3x3, p=1

    // Stage 0: 1 block, 128->256
    BasicBlock stage0[1];

    // Stage 1: 2 blocks, 256->256
    BasicBlock stage1[2];

    // Stage 2: 5 blocks, 256->512
    BasicBlock stage2[5];

    // Stage 3: 3 blocks, 512->512
    BasicBlock stage3[3];

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Stem + pool (2,2) -> 16x64
        x = stem_conv1(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);
        x = stem_conv2(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);

        // Stage 0 + pool (2,2) -> 8x32
        for (auto & blk : stage0) x = blk(ctx, x);
        x = ctx.max_pool2d(x, 2, 2);

        // Stage 1 + asymmetric pool (kH=2,kW=1) -> 4x32
        for (auto & blk : stage1) x = blk(ctx, x);
        x = ctx.max_pool2d(x, /*k0*/1, /*k1*/2, /*s0*/1, /*s1*/2, /*p0*/0, /*p1*/0);

        // Stage 2 (no pool)
        for (auto & blk : stage2) x = blk(ctx, x);

        // Stage 3 (no pool)
        for (auto & blk : stage3) x = blk(ctx, x);

        return x;  // (32, 4, 512, 1) in ggml order
    }

    static ResNet31 load(fabric::Model & m, const std::string & prefix) {
        ResNet31 net;
        fabric::ConvParams p1 = {.stride = 1, .padding = 1};

        net.stem_conv1 = fabric::Conv2d::load(m, prefix + ".stem.0", p1);
        net.stem_conv2 = fabric::Conv2d::load(m, prefix + ".stem.1", p1);

        // Stage 0: 1 block, first has downsample (128->256)
        net.stage0[0] = BasicBlock::load(m, prefix + ".stage0.0", true);

        // Stage 1: 2 blocks (256->256, no downsample needed)
        net.stage1[0] = BasicBlock::load(m, prefix + ".stage1.0", false);
        net.stage1[1] = BasicBlock::load(m, prefix + ".stage1.1", false);

        // Stage 2: 5 blocks, first has downsample (256->512)
        net.stage2[0] = BasicBlock::load(m, prefix + ".stage2.0", true);
        for (int i = 1; i < 5; i++) {
            net.stage2[i] = BasicBlock::load(m, prefix + ".stage2." + std::to_string(i), false);
        }

        // Stage 3: 3 blocks (512->512, no downsample)
        for (int i = 0; i < 3; i++) {
            net.stage3[i] = BasicBlock::load(m, prefix + ".stage3." + std::to_string(i), false);
        }

        return net;
    }
};

} // namespace ocr
