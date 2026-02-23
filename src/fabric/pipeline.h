#pragma once

// Fabric SDK — High-level inference pipeline
//
// Provides a simple, Transformers-like API for text generation:
//
//   fabric::Pipeline pipe("model.gguf");
//   std::string text = pipe.generate("Hello", 64);
//
// With streaming:
//
//   pipe.generate("Hello", 64, [](const char * piece) {
//       printf("%s", piece); fflush(stdout);
//   });
//
// Only depends on the public llama.h API.

#include <llama.h>

#include <functional>
#include <string>
#include <vector>
#include <stdexcept>

namespace fabric {

class Pipeline {
public:
    explicit Pipeline(const std::string & model_path, int n_gpu_layers = 99) {
        ggml_backend_load_all();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = n_gpu_layers;

        model_ = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model_) {
            throw std::runtime_error("failed to load model: " + model_path);
        }

        vocab_ = llama_model_get_vocab(model_);
    }

    ~Pipeline() {
        if (model_) llama_model_free(model_);
    }

    Pipeline(const Pipeline &) = delete;
    Pipeline & operator=(const Pipeline &) = delete;

    using stream_fn_t = std::function<void(const char * piece)>;

    std::string generate(
            const std::string & prompt,
            int max_tokens = 64,
            stream_fn_t on_token = nullptr) {
        // Tokenize
        std::vector<llama_token> tokens = tokenize(prompt);

        // Create context
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx   = (int)tokens.size() + max_tokens;
        cp.n_batch = (int)tokens.size();

        llama_context * ctx = llama_init_from_model(model_, cp);
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }

        // Create sampler
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // Stream the prompt if callback provided
        if (on_token) on_token(prompt.c_str());

        std::string result = prompt;

        // Evaluate prompt
        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        if (llama_decode(ctx, batch)) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            throw std::runtime_error("failed to decode prompt");
        }

        // Generate
        for (int i = 0; i < max_tokens; i++) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab_, id)) break;

            std::string piece = detokenize(id);
            result += piece;

            if (on_token) on_token(piece.c_str());

            batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch)) break;
        }

        llama_sampler_free(smpl);
        llama_free(ctx);
        return result;
    }

    llama_model       * model() { return model_; }
    const llama_vocab * vocab() { return vocab_; }

private:
    llama_model       * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;

    std::vector<llama_token> tokenize(const std::string & text) {
        int n = -llama_tokenize(vocab_, text.c_str(), (int)text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n);
        if (llama_tokenize(vocab_, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), true, true) < 0) {
            throw std::runtime_error("failed to tokenize");
        }
        return tokens;
    }

    std::string detokenize(llama_token id) {
        char buf[256];
        int n = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
        if (n < 0) return "";
        return std::string(buf, n);
    }
};

} // namespace fabric
