#pragma once

#include "llama-arch.h"
#include "llama.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

struct llama_model;
struct llama_model_loader;
struct llama_hparams;
struct llm_graph_context;
struct llm_graph_params;

// Information needed to define an external model architecture.
// At minimum, tensor_names and build_graph must be provided.
struct llama_arch_info {
    // Identity — must match general.architecture in the GGUF file
    std::string name;

    // Tensor naming (maps llm_tensor enum → GGUF name strings)
    std::map<llm_tensor, const char *> tensor_names;

    // Properties
    llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;
    bool is_recurrent = false;
    bool is_hybrid    = false;
    bool is_diffusion = false;

    // Graph builder factory (REQUIRED)
    // Must return a fully-constructed llm_graph_context subclass
    using build_graph_fn_t = std::function<std::unique_ptr<llm_graph_context>(
        const llama_model &, const llm_graph_params &)>;
    build_graph_fn_t build_graph = nullptr;

    // Hparams loader (OPTIONAL — generic fallback reads standard keys)
    using load_hparams_fn_t = std::function<void(
        llama_model_loader & ml, llama_hparams & hparams)>;
    load_hparams_fn_t load_hparams = nullptr;

    // Tensor creator (OPTIONAL — generic fallback auto-maps from tensor_names)
    using create_tensors_fn_t = std::function<void(
        llama_model & model, llama_model_loader & ml)>;
    create_tensors_fn_t create_tensors = nullptr;
};

// RAII registration — use as a static global to register before main()
struct llama_arch_registration {
    llama_arch_registration(const char * name, llama_arch_info info);
    ~llama_arch_registration();

private:
    std::string reg_name;
};

// Query the registry for an external architecture by name
const llama_arch_info * llama_arch_lookup(const std::string & name);

// Generic fallback: load arch-specific hparams keys with sensible defaults.
// Called when no custom load_hparams callback is registered.
void llama_arch_load_hparams_generic(llama_model_loader & ml, llama_hparams & hparams);

// Set/get the active external architecture name for LLM_KV prefix resolution.
// Must be called before hparams/tensor loading for external architectures.
// Pass nullptr to clear. NOT thread-safe (model loading is single-threaded).
void llm_arch_set_external_arch_name(const char * name);
const char * llm_arch_get_external_arch_name();

// Set/get the active external tensor names for the LLM_TN system.
// Must be called before tensor creation for external architectures.
// Pass nullptr to clear. NOT thread-safe (model loading is single-threaded).
void llm_arch_set_external_tensor_names(const std::map<llm_tensor, const char *> * names);
const std::map<llm_tensor, const char *> * llm_arch_get_external_tensor_names();
