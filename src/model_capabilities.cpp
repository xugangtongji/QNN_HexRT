#include "model_capabilities.hpp"

#include <algorithm>
#include <iterator>

namespace {

constexpr qhx_hexagon_architectures kAllSupportedArchitectures =
    QHX_HEXAGON_ARCH_V75 | QHX_HEXAGON_ARCH_V79 | QHX_HEXAGON_ARCH_V81;

#if defined(QHX_ENABLE_LFM2_5_350M_CANDIDATE) && \
    QHX_ENABLE_LFM2_5_350M_CANDIDATE
constexpr qhx_model_support_state kLfm2_5_350mSupport =
    QHX_MODEL_SUPPORT_EXECUTABLE;
#else
constexpr qhx_model_support_state kLfm2_5_350mSupport =
    QHX_MODEL_SUPPORT_DECLARED;
#endif

constexpr qhx_model_capability kModelCapabilities[] = {
    {
        sizeof(qhx_model_capability),
        QHX_MODEL_QWEN3_5_0_8B,
        QHX_MODEL_FAMILY_LLM,
        QHX_MODEL_OPERATION_GENERATE_TEXT,
        kAllSupportedArchitectures,
        QHX_MODEL_SUPPORT_DECLARED,
        QHX_RUNNER_QWEN3_5,
        QHX_GRAPH_CONTRACT_UNKNOWN,
        "qwen3_5_0_8b",
        "qwen3.5-0.8b-1024",
    },
    {
        sizeof(qhx_model_capability),
        QHX_MODEL_LFM2_5_350M,
        QHX_MODEL_FAMILY_LLM,
        QHX_MODEL_OPERATION_GENERATE_TEXT,
        QHX_HEXAGON_ARCH_V79,
        kLfm2_5_350mSupport,
        QHX_RUNNER_LFM2,
        QHX_GRAPH_CONTRACT_LFM2_SHARED_V1,
        "lfm2_5_350m",
        "lfm2-5-350m-2048",
    },
    {
        sizeof(qhx_model_capability),
        QHX_MODEL_QWEN3_0_6B,
        QHX_MODEL_FAMILY_LLM,
        QHX_MODEL_OPERATION_GENERATE_TEXT,
        QHX_HEXAGON_ARCH_V75 | QHX_HEXAGON_ARCH_V81,
        QHX_MODEL_SUPPORT_DECLARED,
        QHX_RUNNER_QWEN3,
        QHX_GRAPH_CONTRACT_UNKNOWN,
        "qwen3_0_6b",
        "qwen3-0.6b-1024final",
    },
    {
        sizeof(qhx_model_capability),
        QHX_MODEL_LFM2_5_230M,
        QHX_MODEL_FAMILY_LLM,
        QHX_MODEL_OPERATION_GENERATE_TEXT,
        kAllSupportedArchitectures,
        QHX_MODEL_SUPPORT_EXECUTABLE,
        QHX_RUNNER_LFM2,
        QHX_GRAPH_CONTRACT_LFM2_SPLIT_V1,
        "lfm2_5_230m",
        "lfm2-5-230m",
    },
};

struct ManifestBinding {
  std::string_view manifest_name;
  qhx_model_id model_id;
};

constexpr ManifestBinding kManifestBindings[] = {
    {"qwen3.5-0.8b-1024", QHX_MODEL_QWEN3_5_0_8B},
    {"lfm2-5-350m-2048", QHX_MODEL_LFM2_5_350M},
    {"qwen3-0.6b-1024final", QHX_MODEL_QWEN3_0_6B},
    {"lfm2-5-230m", QHX_MODEL_LFM2_5_230M},
};

const qhx_model_capability* find_by_id(qhx_model_id model_id) noexcept {
  const auto capability =
      std::find_if(std::begin(kModelCapabilities), std::end(kModelCapabilities),
                   [model_id](const qhx_model_capability& candidate) {
                     return candidate.model_id == model_id;
                   });
  return capability == std::end(kModelCapabilities) ? nullptr : capability;
}

}  // namespace

namespace qhx {

const qhx_model_capability* find_model_capability_by_manifest(
    std::string_view manifest_name) noexcept {
  const auto binding =
      std::find_if(std::begin(kManifestBindings), std::end(kManifestBindings),
                   [manifest_name](const ManifestBinding& candidate) {
                     return manifest_name == candidate.manifest_name;
                   });
  return binding == std::end(kManifestBindings) ? nullptr : find_by_id(binding->model_id);
}

}  // namespace qhx

extern "C" {

qhx_status qhx_model_capabilities(const qhx_model_capability** capabilities,
                                  size_t* count) {
  if (!capabilities || !count) return QHX_ERROR_INVALID_ARGUMENT;
  *capabilities = kModelCapabilities;
  *count = std::size(kModelCapabilities);
  return QHX_OK;
}

const qhx_model_capability* qhx_model_capability_find(qhx_model_id model_id) {
  return find_by_id(model_id);
}

}  // extern "C"
