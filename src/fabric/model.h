#pragma once

// Fabric SDK — Generic Model Loader
//
// Loads GGUF files directly using the ggml GGUF API.
// No LLM dependencies — does not use llama_model_loader.
//
//   fabric::Model model("mnist.gguf");
//   ggml_tensor * w = model.require("conv1.weight");
//   int n_classes = model.get_i32("n_classes");
//

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <string>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstdio>
#include <thread>

namespace fabric {

class Model {
public:
    explicit Model(const std::string & path) : path_(path) {
        // Phase 1: Load GGUF metadata + tensor descriptors (no data yet)
        ggml_context * meta = nullptr;
        gguf_init_params params = { /*no_alloc=*/ true, /*ctx=*/ &meta };

        gguf_.reset(gguf_init_from_file(path.c_str(), params));
        if (!gguf_) {
            throw std::runtime_error("fabric::Model: failed to load " + path);
        }
        meta_.reset(meta);

        // Phase 2: Create data context with duplicate tensors
        int64_t n_tensors = gguf_get_n_tensors(gguf_.get());
        size_t ctx_size = ggml_tensor_overhead() * n_tensors;
        ggml_init_params ctx_params = { ctx_size, nullptr, true };
        data_.reset(ggml_init(ctx_params));
        if (!data_) {
            throw std::runtime_error("fabric::Model: failed to create data context");
        }

        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_.get(), i);
            ggml_tensor * meta_t = ggml_get_tensor(meta_.get(), name);
            ggml_tensor * data_t = ggml_dup_tensor(data_.get(), meta_t);
            ggml_set_name(data_t, name);
        }

        // Phase 3: Allocate backend buffer and load tensor data from file
        // Try GPU first, fall back to CPU
        backend_.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
        if (backend_) {
            fprintf(stderr, "[fabric] using GPU backend\n");
        } else {
            backend_.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            if (!backend_) {
                throw std::runtime_error("fabric::Model: failed to init backend");
            }
            int hw_threads = (int)std::thread::hardware_concurrency();
            if (hw_threads > 0) {
                ggml_backend_cpu_set_n_threads(backend_.get(), hw_threads);
            }
            fprintf(stderr, "[fabric] using CPU backend (%d threads)\n", hw_threads);
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_.get());
        buf_.reset(ggml_backend_alloc_ctx_tensors_from_buft(data_.get(), buft));
        if (!buf_) {
            throw std::runtime_error("fabric::Model: failed to allocate buffer");
        }
        ggml_backend_buffer_set_usage(buf_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Read tensor data from file
        size_t data_offset = gguf_get_data_offset(gguf_.get());
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("fabric::Model: failed to open " + path);
        }

        std::vector<uint8_t> read_buf;
        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_.get(), i);
            size_t offset = data_offset + gguf_get_tensor_offset(gguf_.get(), i);
            ggml_tensor * cur = ggml_get_tensor(data_.get(), name);
            size_t nbytes = ggml_nbytes(cur);

            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                throw std::runtime_error(std::string("fabric::Model: failed to seek for tensor ") + name);
            }

            if (ggml_backend_buft_is_host(buft)) {
                fin.read(reinterpret_cast<char *>(cur->data), nbytes);
            } else {
                read_buf.resize(nbytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), nbytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
            }
        }
    }

    // Tensor lookup (returns nullptr if missing)
    ggml_tensor * tensor(const std::string & name) {
        return ggml_get_tensor(data_.get(), name.c_str());
    }

    // Tensor lookup (throws if missing)
    ggml_tensor * require(const std::string & name) {
        ggml_tensor * t = tensor(name);
        if (!t) {
            throw std::runtime_error("fabric::Model: missing tensor: " + name);
        }
        return t;
    }

    // GGUF metadata access
    int32_t get_i32(const std::string & key) {
        int64_t id = gguf_find_key(gguf_.get(), key.c_str());
        if (id < 0) throw std::runtime_error("fabric::Model: missing key: " + key);
        return gguf_get_val_i32(gguf_.get(), id);
    }

    uint32_t get_u32(const std::string & key) {
        int64_t id = gguf_find_key(gguf_.get(), key.c_str());
        if (id < 0) throw std::runtime_error("fabric::Model: missing key: " + key);
        return gguf_get_val_u32(gguf_.get(), id);
    }

    float get_f32(const std::string & key) {
        int64_t id = gguf_find_key(gguf_.get(), key.c_str());
        if (id < 0) throw std::runtime_error("fabric::Model: missing key: " + key);
        return gguf_get_val_f32(gguf_.get(), id);
    }

    std::string get_str(const std::string & key) {
        int64_t id = gguf_find_key(gguf_.get(), key.c_str());
        if (id < 0) throw std::runtime_error("fabric::Model: missing key: " + key);
        return gguf_get_val_str(gguf_.get(), id);
    }

    bool get_bool(const std::string & key) {
        int64_t id = gguf_find_key(gguf_.get(), key.c_str());
        if (id < 0) throw std::runtime_error("fabric::Model: missing key: " + key);
        return gguf_get_val_bool(gguf_.get(), id);
    }

    bool has_key(const std::string & key) {
        return gguf_find_key(gguf_.get(), key.c_str()) >= 0;
    }

    // Debug: print all tensors
    void print_tensors() {
        int64_t n = gguf_get_n_tensors(gguf_.get());
        fprintf(stderr, "fabric::Model: %s (%lld tensors)\n", path_.c_str(), (long long)n);
        for (int64_t i = 0; i < n; i++) {
            const char * name = gguf_get_tensor_name(gguf_.get(), i);
            ggml_tensor * t = ggml_get_tensor(data_.get(), name);
            fprintf(stderr, "  %-40s [%4lld, %4lld, %4lld, %4lld] %s\n",
                name,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                ggml_type_name(t->type));
        }
    }

    ggml_backend_t backend() { return backend_.get(); }

private:
    std::string path_;
    gguf_context_ptr          gguf_;
    ggml_context_ptr          meta_;
    ggml_context_ptr          data_;
    ggml_backend_ptr          backend_;
    ggml_backend_buffer_ptr   buf_;
};

} // namespace fabric
