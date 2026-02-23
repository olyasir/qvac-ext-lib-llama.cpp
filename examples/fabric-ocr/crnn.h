#pragma once

// CRNN (Convolutional Recurrent Neural Network) for text recognition.
//
// Architecture: VGG16-BN backbone -> BiLSTM (2 layers) -> Linear -> CTC decode
//
// The BiLSTM is unrolled at graph build time since ggml uses static graphs.
// Fixed sequence length = 32 (width of VGG output).
//
// Graph size: ~2600 nodes (32 steps x 2 layers x 2 directions x ~19 ops).
// Requires Runner with graph_size=4096.

#include "../../src/fabric/nn.h"
#include "../../src/fabric/nn_modules.h"
#include "vgg16_bn.h"

namespace ocr {

// Single LSTM cell weights (bias_ih + bias_hh pre-merged at export)
struct LSTMCellWeights {
    ggml_tensor * W_ih = nullptr;  // (4*hidden, input_size)
    ggml_tensor * W_hh = nullptr;  // (4*hidden, hidden_size)
    ggml_tensor * bias = nullptr;  // (4*hidden,) — merged b_ih + b_hh
};

// One LSTM direction (forward or backward)
struct LSTMDirection {
    LSTMCellWeights weights;
    int hidden_size = 0;

    // Unroll one direction over the full sequence.
    // x_steps: vector of (input_size,) tensors, one per timestep
    // Returns: (hidden_size, seq_len) tensor
    fabric::Tensor unroll(fabric::Context & ctx,
                          const std::vector<fabric::Tensor> & x_steps,
                          fabric::Tensor h0,
                          fabric::Tensor c0,
                          bool reverse) const {
        int seq_len = (int)x_steps.size();
        fabric::Tensor W_ih = ctx.wrap(weights.W_ih);
        fabric::Tensor W_hh = ctx.wrap(weights.W_hh);
        fabric::Tensor b    = ctx.wrap(weights.bias);

        fabric::Tensor h = h0;
        fabric::Tensor c = c0;

        size_t gate_bytes = hidden_size * sizeof(float);
        std::vector<fabric::Tensor> outputs;
        outputs.reserve(seq_len);

        for (int t = 0; t < seq_len; t++) {
            int idx = reverse ? (seq_len - 1 - t) : t;
            fabric::Tensor xt = x_steps[idx];

            // gates = W_ih @ x_t + W_hh @ h + bias
            fabric::Tensor gates = ctx.matmul(W_ih, xt) + ctx.matmul(W_hh, h) + b;

            // Split into i, f, g, o (each hidden_size)
            fabric::Tensor i_gate = ctx.view_1d(gates, hidden_size, 0 * gate_bytes).sigmoid();
            fabric::Tensor f_gate = ctx.view_1d(gates, hidden_size, 1 * gate_bytes).sigmoid();
            fabric::Tensor g_gate = ctx.view_1d(gates, hidden_size, 2 * gate_bytes).tanh_();
            fabric::Tensor o_gate = ctx.view_1d(gates, hidden_size, 3 * gate_bytes).sigmoid();

            // c = f * c_prev + i * g
            c = f_gate * c + i_gate * g_gate;

            // h = o * tanh(c)
            h = o_gate * c.tanh_();

            outputs.push_back(h);
        }

        // If reversed, outputs are in reverse order — fix that
        if (reverse) {
            std::reverse(outputs.begin(), outputs.end());
        }

        // Build (hidden_size, seq_len) by chaining concat along dim=1
        fabric::Tensor seq = ctx.reshape_2d(outputs[0], hidden_size, 1);
        for (int t = 1; t < seq_len; t++) {
            fabric::Tensor ht = ctx.reshape_2d(outputs[t], hidden_size, 1);
            seq = ctx.concat(seq, ht, 1);
        }

        return seq;  // (hidden_size, seq_len)
    }
};

// Bidirectional LSTM layer
struct BiLSTMLayer {
    LSTMDirection forward_dir;
    LSTMDirection backward_dir;
    int hidden_size = 0;

    // Returns (2*hidden_size, seq_len)
    fabric::Tensor run(fabric::Context & ctx,
                       const std::vector<fabric::Tensor> & x_steps,
                       fabric::Tensor h0_fwd, fabric::Tensor c0_fwd,
                       fabric::Tensor h0_bwd, fabric::Tensor c0_bwd) const {
        fabric::Tensor fwd = forward_dir.unroll(ctx, x_steps, h0_fwd, c0_fwd, false);
        fabric::Tensor bwd = backward_dir.unroll(ctx, x_steps, h0_bwd, c0_bwd, true);
        return ctx.concat(fwd, bwd, 0);  // (2*hidden, seq_len)
    }
};

// Full CRNN model
struct CRNN : fabric::Module<CRNN> {
    VGG16BN backbone;
    BiLSTMLayer lstm0;  // layer 0
    BiLSTMLayer lstm1;  // layer 1
    fabric::GenericLinear linear;

    int hidden_size = 128;
    int seq_len     = 32;

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        // Backbone: (W=128, H=32, C=3, N=1) -> (W=32, H=1, C=512, N=1)
        x = backbone(ctx, x);

        // VGG output: (W=32, H=1, C=512, N=1) — W varies fastest in memory.
        // Reshape to (ne0=seq_len=32, ne1=feat_dim=512), then transpose+cont
        // so features become contiguous per timestep.
        int feat_dim = (int)x.dim(2);  // 512
        x = ctx.reshape_2d(x, seq_len, feat_dim);   // (32, 512): [time, feature]
        x = ctx.transpose(x);                        // (512, 32): [feature, time] non-contiguous
        x = ctx.cont(x);                             // contiguous: feature varies fastest

        // Create initial states (zeros) — 2 layers x 2 directions x {h, c}
        auto make_zero = [&](const char * name) {
            return ctx.new_input(name, GGML_TYPE_F32, {(int64_t)hidden_size});
        };

        fabric::Tensor h0_l0_f = make_zero("h0_l0_f");
        fabric::Tensor c0_l0_f = make_zero("c0_l0_f");
        fabric::Tensor h0_l0_b = make_zero("h0_l0_b");
        fabric::Tensor c0_l0_b = make_zero("c0_l0_b");
        fabric::Tensor h0_l1_f = make_zero("h0_l1_f");
        fabric::Tensor c0_l1_f = make_zero("c0_l1_f");
        fabric::Tensor h0_l1_b = make_zero("h0_l1_b");
        fabric::Tensor c0_l1_b = make_zero("c0_l1_b");

        // Extract per-timestep vectors from (feat_dim=512, seq_len=32)
        // Timestep t is contiguous at offset t*feat_dim*sizeof(float)
        std::vector<fabric::Tensor> steps;
        steps.reserve(seq_len);
        for (int t = 0; t < seq_len; t++) {
            steps.push_back(ctx.view_1d(x, feat_dim, t * feat_dim * sizeof(float)));
        }

        // Layer 0: input_size=512, hidden=128
        fabric::Tensor l0_out = lstm0.run(ctx, steps, h0_l0_f, c0_l0_f, h0_l0_b, c0_l0_b);
        // l0_out: (2*hidden=256, seq_len=32)

        // Extract timesteps from layer 0 output for layer 1
        int l1_input = 2 * hidden_size;  // 256
        std::vector<fabric::Tensor> steps1;
        steps1.reserve(seq_len);
        for (int t = 0; t < seq_len; t++) {
            steps1.push_back(ctx.view_1d(l0_out, l1_input, t * l1_input * sizeof(float)));
        }

        // Layer 1: input_size=256, hidden=128
        fabric::Tensor l1_out = lstm1.run(ctx, steps1, h0_l1_f, c0_l1_f, h0_l1_b, c0_l1_b);
        // l1_out: (256, 32)

        // Linear projection: (vocab_size, 256) @ (256, 32) = (vocab_size, 32)
        fabric::Tensor logits = linear(ctx, l1_out);

        return logits;  // (vocab_size+1, seq_len)
    }

    static CRNN load(fabric::Model & m, const std::string & prefix, int hidden = 128) {
        CRNN net;
        net.hidden_size = hidden;

        net.backbone = VGG16BN::load(m, prefix + ".feat_extractor");

        // LSTM layer 0
        net.lstm0.hidden_size = hidden;
        net.lstm0.forward_dir.hidden_size = hidden;
        net.lstm0.forward_dir.weights = {
            m.require(prefix + ".encoder.weight_ih_l0"),
            m.require(prefix + ".encoder.weight_hh_l0"),
            m.require(prefix + ".encoder.bias_l0"),
        };
        net.lstm0.backward_dir.hidden_size = hidden;
        net.lstm0.backward_dir.weights = {
            m.require(prefix + ".encoder.weight_ih_l0_reverse"),
            m.require(prefix + ".encoder.weight_hh_l0_reverse"),
            m.require(prefix + ".encoder.bias_l0_reverse"),
        };

        // LSTM layer 1
        net.lstm1.hidden_size = hidden;
        net.lstm1.forward_dir.hidden_size = hidden;
        net.lstm1.forward_dir.weights = {
            m.require(prefix + ".encoder.weight_ih_l1"),
            m.require(prefix + ".encoder.weight_hh_l1"),
            m.require(prefix + ".encoder.bias_l1"),
        };
        net.lstm1.backward_dir.hidden_size = hidden;
        net.lstm1.backward_dir.weights = {
            m.require(prefix + ".encoder.weight_ih_l1_reverse"),
            m.require(prefix + ".encoder.weight_hh_l1_reverse"),
            m.require(prefix + ".encoder.bias_l1_reverse"),
        };

        net.linear = fabric::GenericLinear::load(m, prefix + ".decoder");

        return net;
    }
};

} // namespace ocr
