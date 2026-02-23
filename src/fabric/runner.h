#pragma once

// Fabric SDK — Inference Runner
//
// Load weights + define architecture in one step, then just call it.
//
//   fabric::Runner model("mnist.gguf", [](auto & ctx, auto & m) {
//       auto x = ctx.new_input("image", GGML_TYPE_F32, {28, 28, 1, 1});
//       return MyCNN(m).forward(ctx, x);
//   });
//
//   auto scores = model(image_data);
//

#include "ggml.h"
#include "ggml-cpp.h"
#include "nn.h"
#include "model.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

namespace fabric {

class Runner {
public:
    using build_fn_t = std::function<Tensor(Context & ctx, Model & model)>;

    // Load weights + build graph in one step
    Runner(const std::string & gguf_path, build_fn_t build_fn, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : owned_model_(std::make_unique<Model>(gguf_path)), model_(*owned_model_), graph_size_(graph_size) {
        build(build_fn);
    }

    // Use existing Model (non-owning)
    Runner(Model & model, build_fn_t build_fn, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : model_(model), graph_size_(graph_size) {
        build(build_fn);
    }

    // --- PyTorch-like: just call it ---

    // Single input (when model has exactly one input)
    std::vector<float> operator()(const void * data) {
        if (input_tensors_.size() != 1) {
            throw std::runtime_error("fabric::Runner: operator()(data) requires exactly 1 input, "
                                     "got " + std::to_string(input_tensors_.size()));
        }
        alloc_graph();
        set_input(input_order_[0], data);
        compute();
        return get_output();
    }

    // Named inputs — unspecified inputs with defaults get auto-filled
    std::vector<float> operator()(const std::unordered_map<std::string, const void *> & inputs) {
        alloc_graph();
        for (auto & [name, data] : inputs) {
            set_input(name, data);
        }
        apply_defaults(inputs);
        compute();
        return get_output();
    }

    // Mark an input to be zero-filled when not explicitly provided
    void set_default_zeros(const std::string & name) {
        auto it = input_tensors_.find(name);
        if (it == input_tensors_.end()) {
            throw std::runtime_error("fabric::Runner: unknown input for default: " + name);
        }
        size_t nbytes = ggml_nbytes(it->second);
        default_zeros_.insert_or_assign(name, std::vector<uint8_t>(nbytes, 0));
    }

    // Access the underlying Model (e.g. for print_tensors())
    Model & weights() { return model_; }

    // --- Post-compute tensor inspection ---

    std::vector<float> get_tensor(const std::string & name) {
        ggml_tensor * t = find_graph_tensor(name);
        if (!t) throw std::runtime_error("fabric::Runner: tensor not found in graph: " + name);
        int64_t n = ggml_nelements(t);
        std::vector<float> result(n);
        ggml_backend_tensor_get(t, result.data(), 0, n * sizeof(float));
        return result;
    }

    std::vector<int64_t> get_tensor_shape(const std::string & name) {
        ggml_tensor * t = find_graph_tensor(name);
        if (!t) throw std::runtime_error("fabric::Runner: tensor not found in graph: " + name);
        int nd = ggml_n_dims(t);
        std::vector<int64_t> shape(nd);
        for (int i = 0; i < nd; i++) shape[i] = t->ne[i];
        return shape;
    }

    // --- Low-level API ---

    void alloc_graph() {
        if (!ggml_gallocr_alloc_graph(alloc_.get(), gf_)) {
            throw std::runtime_error("fabric::Runner: failed to alloc graph");
        }
    }

    void set_input(const std::string & name, const void * data) {
        auto it = input_tensors_.find(name);
        if (it == input_tensors_.end()) {
            throw std::runtime_error("fabric::Runner: unknown input: " + name);
        }
        ggml_tensor * t = it->second;
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }

    void compute() {
        ggml_status status = ggml_backend_graph_compute(model_.backend(), gf_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("fabric::Runner: compute failed");
        }
    }

    std::vector<float> get_output() {
        int64_t n = ggml_nelements(output_);
        std::vector<float> result(n);
        ggml_backend_tensor_get(output_, result.data(), 0, n * sizeof(float));
        return result;
    }

private:
    std::unique_ptr<Model> owned_model_;  // set when Runner owns the weights
    Model & model_;
    size_t graph_size_ = GGML_DEFAULT_GRAPH_SIZE;
    ggml_context_ptr ctx_;
    ggml_cgraph * gf_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_gallocr_ptr alloc_;
    std::unordered_map<std::string, ggml_tensor *> input_tensors_;
    std::vector<std::string> input_order_;
    std::unordered_map<std::string, std::vector<uint8_t>> default_zeros_;

    void apply_defaults(const std::unordered_map<std::string, const void *> & provided) {
        for (auto & [name, zeros] : default_zeros_) {
            if (provided.find(name) == provided.end()) {
                set_input(name, zeros.data());
            }
        }
    }

    ggml_tensor * find_graph_tensor(const std::string & name) {
        int n_nodes = ggml_graph_n_nodes(gf_);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(gf_, i);
            if (node->name[0] != '\0' && name == node->name) return node;
        }
        return nullptr;
    }

    void build(build_fn_t & build_fn) {
        size_t ctx_size = ggml_tensor_overhead() * graph_size_ + ggml_graph_overhead_custom(graph_size_, false);
        ggml_init_params params = { ctx_size, nullptr, true };
        ctx_.reset(ggml_init(params));
        if (!ctx_) {
            throw std::runtime_error("fabric::Runner: failed to create context");
        }

        gf_ = ggml_new_graph_custom(ctx_.get(), graph_size_, false);

        Context fctx(ctx_.get(), gf_);
        Tensor output = build_fn(fctx, model_);
        output_ = output.ptr;
        fctx.finalize(output);

        // Check graph capacity
        int n_nodes = ggml_graph_n_nodes(gf_);
        if ((size_t)n_nodes > graph_size_) {
            throw std::runtime_error("fabric::Runner: graph has " + std::to_string(n_nodes) +
                " nodes but capacity is " + std::to_string(graph_size_) +
                ". Increase graph_size, e.g. Runner(path, fn, " +
                std::to_string(graph_size_ * 2) + ")");
        }
        fprintf(stderr, "[fabric] graph: %d nodes, capacity %zu\n", n_nodes, graph_size_);

        // Collect named input tensors by scanning graph node sources (O(n) linear walk).
        // Every leaf tensor (input or weight) is a direct source of at least one graph node.
        // Only user inputs have GGML_TENSOR_FLAG_INPUT; model weights do not.
        std::unordered_set<ggml_tensor *> visited;
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = ggml_graph_node(gf_, i);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                ggml_tensor * src = node->src[s];
                if (!src) continue;
                if (!visited.insert(src).second) continue;
                if ((src->flags & GGML_TENSOR_FLAG_INPUT) && src->name[0] != '\0') {
                    input_tensors_[src->name] = src;
                }
            }
        }

        for (auto & [name, _] : input_tensors_) {
            input_order_.push_back(name);
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model_.backend());
        alloc_.reset(ggml_gallocr_new(buft));
        if (!ggml_gallocr_reserve(alloc_.get(), gf_)) {
            throw std::runtime_error("fabric::Runner: failed to reserve allocator");
        }
    }

};

} // namespace fabric
