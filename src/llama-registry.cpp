#include "llama-registry.h"

#include "llama-impl.h"
#include "llama-hparams.h"
#include "llama-model-loader.h"

#include <map>
#include <mutex>
#include <string>

// ---------------------------------------------------------------------------
// Registry singleton
// ---------------------------------------------------------------------------

static std::mutex & registry_mutex() {
    static std::mutex m;
    return m;
}

static std::map<std::string, llama_arch_info> & registry_map() {
    static std::map<std::string, llama_arch_info> m;
    return m;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

llama_arch_registration::llama_arch_registration(const char * name, llama_arch_info info) : reg_name(name) {
    info.name = name;
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry_map()[reg_name] = std::move(info);
}

llama_arch_registration::~llama_arch_registration() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry_map().erase(reg_name);
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

const llama_arch_info * llama_arch_lookup(const std::string & name) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto & m = registry_map();
    auto it = m.find(name);
    if (it == m.end()) {
        return nullptr;
    }
    return &it->second;
}

// ---------------------------------------------------------------------------
// Generic hparams loader
// ---------------------------------------------------------------------------

void llama_arch_load_hparams_generic(llama_model_loader & ml, llama_hparams & hparams) {
    // Try RMS norm epsilon first, then regular layernorm epsilon
    if (!ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false)) {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);
    }
}

// ---------------------------------------------------------------------------
// External tensor names for LLM_TN system
// ---------------------------------------------------------------------------

static const char * s_external_arch_name = nullptr;
static const std::map<llm_tensor, const char *> * s_external_tensor_names = nullptr;

void llm_arch_set_external_arch_name(const char * name) {
    s_external_arch_name = name;
}

const char * llm_arch_get_external_arch_name() {
    return s_external_arch_name;
}

void llm_arch_set_external_tensor_names(const std::map<llm_tensor, const char *> * names) {
    s_external_tensor_names = names;
}

const std::map<llm_tensor, const char *> * llm_arch_get_external_tensor_names() {
    return s_external_tensor_names;
}
