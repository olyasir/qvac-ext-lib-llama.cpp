// Fabric SDK — CNN Example
//
// Usage:
//   1. python examples/fabric-cnn/train_mnist.py
//   2. python scripts/convert_pytorch_to_gguf.py mnist_cnn.pt mnist.gguf --transpose-conv
//   3. ./fabric-cnn mnist.gguf [image.bin ...]

#include <fabric/nn.h>
#include <fabric/nn_modules.h>
#include <fabric/runner.h>

#include <cstdio>
#include <fstream>
#include <vector>
#include <algorithm>

struct MnistCNN : fabric::Module<MnistCNN> {
    fabric::Conv2d        conv1, conv2;
    fabric::GenericLinear fc1, fc2;

    MnistCNN(fabric::Model & m)
        : conv1(fabric::Conv2d::load(m, "conv1", {.padding=1}))
        , conv2(fabric::Conv2d::load(m, "conv2", {.padding=1}))
        , fc1(fabric::GenericLinear::load(m, "fc1"))
        , fc2(fabric::GenericLinear::load(m, "fc2")) {}

    fabric::Tensor forward(fabric::Context & ctx, fabric::Tensor x) const {
        x = conv1(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);
        x = conv2(ctx, x).relu();
        x = ctx.max_pool2d(x, 2, 2);
        x = fc1(ctx, x.flatten()).relu();
        return fc2(ctx, x);
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.bin ...]\n", argv[0]);
        return 1;
    }

    // One line: load weights + build graph
    fabric::Runner model(argv[1], [](fabric::Context & ctx, fabric::Model & m) {
        auto x = ctx.new_input("image", GGML_TYPE_F32, {28, 28, 1, 1});
        return MnistCNN(m).forward(ctx, x);
    });

    // Just call it
    for (int i = 2; i < argc; i++) {
        std::vector<float> image(28 * 28);
        std::ifstream fin(argv[i], std::ios::binary);
        if (!fin) { fprintf(stderr, "ERROR: %s\n", argv[i]); continue; }
        fin.read(reinterpret_cast<char *>(image.data()), image.size() * sizeof(float));

        auto scores = model(image.data());

        auto it = std::max_element(scores.begin(), scores.end());
        printf("%s -> %zu (%.2f)\n", argv[i], std::distance(scores.begin(), it), *it);
    }

    return 0;
}
