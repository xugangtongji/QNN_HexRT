#include <qhexrt/qhexrt_c.h>

#include "lfm_manifest.hpp"
#include "lfm_runner.hpp"
#include "model_capabilities.hpp"
#include "qhexrt_internal.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <regex>
#include <string>
#include <string_view>

namespace fs = std::filesystem;
namespace {

thread_local std::string last_error;

std::string read_text(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return {};
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

bool json_string(const std::string& json, const char* key, std::string& value) {
  const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) return false;
  value = match[1].str();
  return true;
}

bool json_int(const std::string& json, const char* key, int& value) {
  const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) return false;
  try {
    value = std::stoi(match[1].str());
    return true;
  } catch (...) {
    return false;
  }
}

bool json_double(const std::string& json, const char* key, double& value) {
  const std::regex pattern(std::string("\\\"") + key +
                           "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) return false;
  try {
    value = std::stod(match[1].str());
    return true;
  } catch (...) {
    return false;
  }
}

bool artifact_string(const std::string& json, const char* section, const char* key,
                     std::string& value) {
  const std::regex section_pattern(std::string("\\\"") + section +
                                   "\\\"\\s*:\\s*\\{([^}]*)\\}");
  std::smatch section_match;
  if (!std::regex_search(json, section_match, section_pattern)) return false;
  return json_string(section_match[1].str(), key, value);
}

fs::path resolve_artifact(const qhx_model& model, const std::string& relative) {
  fs::path path(relative);
  return path.is_absolute() ? path : fs::path(model.artifacts_dir) / path;
}

bool regular_nonempty(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0;
}

qhx_hexagon_architectures architecture_mask(std::string_view architecture) {
  if (architecture == "v75") return QHX_HEXAGON_ARCH_V75;
  if (architecture == "v79") return QHX_HEXAGON_ARCH_V79;
  if (architecture == "v81") return QHX_HEXAGON_ARCH_V81;
  return QHX_HEXAGON_ARCH_NONE;
}

bool load_lfm_assets(qhx_model& model) {
  const fs::path tokenizer = resolve_artifact(model, model.tokenizer_path);
  const fs::path embedding = resolve_artifact(model, model.embedding_path);
  if (!regular_nonempty(tokenizer) || !regular_nonempty(embedding)) {
    last_error = "Tokenizer or embedding artifact is missing";
    return false;
  }
  std::error_code ec;
  const auto expected_embedding =
      static_cast<uintmax_t>(model.vocab) * static_cast<uintmax_t>(model.hidden) * 2U;
  if (fs::file_size(embedding, ec) != expected_embedding) {
    last_error = "Embedding table size does not match vocab*hidden*fp16";
    return false;
  }
  model.tokenizer_path = tokenizer.string();
  model.embedding_path = embedding.string();
  if (!model.embedding.open(model.embedding_path, qhx::MappingAccess::kRandom, last_error)) {
    return false;
  }
  if (model.embedding.size() != expected_embedding) {
    last_error = "Mapped embedding table size changed during model loading";
    return false;
  }
  model.tokenizer = std::make_unique<qhx::Tokenizer>();
  return model.tokenizer->load(model.tokenizer_path, last_error);
}

bool load_lfm2_split_v1(qhx_model& model, const std::string& json) {
  if (!json_int(json, "hidden", model.hidden) || !json_int(json, "vocab", model.vocab) ||
      !json_int(json, "max_ctx", model.max_ctx) || !json_int(json, "kv_dim", model.kv_dim) ||
      !json_int(json, "head_dim", model.head_dim) ||
      !json_int(json, "eos_token_id", model.eos_token_id) ||
      !json_double(json, "rope_theta", model.rope_theta)) {
    last_error = "Manifest is missing required LFM2 split-v1 fields";
    return false;
  }
  if (model.model_id != QHX_MODEL_LFM2_5_230M || model.hidden != 1024 ||
      model.vocab != 65536 || model.max_ctx <= 0 || model.kv_dim != 512 ||
      model.head_dim != 64) {
    last_error = "Manifest does not match the verified LFM2 split-v1 graph contract";
    return false;
  }
  model.bos_token_id = 1;
  std::string prefill_bin, decode_bin, lmhead_bin;
  if (!artifact_string(json, "prefill", "bin", prefill_bin) ||
      !artifact_string(json, "decode", "bin", decode_bin) ||
      !artifact_string(json, "lmhead", "bin", lmhead_bin) ||
      !json_string(json, "embed", model.embedding_path) ||
      !json_string(json, "tokenizer", model.tokenizer_path)) {
    last_error = "Manifest is missing LFM artifacts";
    return false;
  }
  if (!load_lfm_assets(model)) return false;
  model.prefill = std::make_unique<qhx::ContextGraph>();
  model.decode = std::make_unique<qhx::ContextGraph>();
  model.lmhead = std::make_unique<qhx::ContextGraph>();
  auto& runtime = model.runtime->impl;
  if (!model.prefill->load(runtime, resolve_artifact(model, prefill_bin).string(),
                           "lfm230_pf_512_w8", last_error) ||
      !model.decode->load(runtime, resolve_artifact(model, decode_bin).string(),
                          "lfm230_dec_512_w8", last_error) ||
      !model.lmhead->load(runtime, resolve_artifact(model, lmhead_bin).string(),
                          "lfm230_lmh_w8", last_error)) {
    return false;
  }
  return qhx::prepare_lfm_model(model, last_error);
}

bool load_lfm2_shared_v1(qhx_model& model, const std::string& json) {
  qhx::LfmManifest manifest;
  if (!qhx::parse_lfm_manifest_v1(json, manifest, last_error)) return false;
  if (model.model_id != QHX_MODEL_LFM2_5_350M ||
      manifest.name != "lfm2-5-350m-2048" || manifest.family != "llm" ||
      manifest.dsp_arch != "v79" ||
      manifest.context_layout != qhx::LfmContextLayout::kSharedBody ||
      manifest.prefill.binary_path != manifest.decode.binary_path ||
      manifest.prefill.graph_name != "lfm_pf_512_f16" ||
      manifest.decode.graph_name != "lfm_dec_2048_f16" ||
      manifest.lmhead.graph_name != "lfm_lmh_f16" || manifest.hidden != 1024 ||
      manifest.vocab != 65536 || manifest.layers != 16 ||
      manifest.max_context != 2048 || manifest.kv_dimension != 512 ||
      manifest.head_dimension != 64 || manifest.bos_token_id != 1 ||
      manifest.eos_token_id != 7) {
    last_error = "Manifest does not match the verified LFM2 shared-v1 graph contract";
    return false;
  }

  model.hidden = manifest.hidden;
  model.vocab = manifest.vocab;
  model.max_ctx = manifest.max_context;
  model.kv_dim = manifest.kv_dimension;
  model.head_dim = manifest.head_dimension;
  model.bos_token_id = manifest.bos_token_id;
  model.eos_token_id = manifest.eos_token_id;
  model.rope_theta = manifest.rope_theta;
  model.embedding_path = manifest.embedding_path;
  model.tokenizer_path = manifest.tokenizer_path;
  if (!load_lfm_assets(model)) return false;

  auto shared_context = std::make_shared<qhx::ContextBinary>();
  auto& runtime = model.runtime->impl;
  if (!shared_context->load(
          runtime, resolve_artifact(model, manifest.prefill.binary_path).string(),
          last_error)) {
    return false;
  }
  model.prefill = std::make_unique<qhx::ContextGraph>();
  model.decode = std::make_unique<qhx::ContextGraph>();
  model.lmhead = std::make_unique<qhx::ContextGraph>();
  if (!model.prefill->load(shared_context, manifest.prefill.graph_name, last_error) ||
      !model.decode->load(shared_context, manifest.decode.graph_name, last_error) ||
      !model.lmhead->load(
          runtime, resolve_artifact(model, manifest.lmhead.binary_path).string(),
          manifest.lmhead.graph_name, last_error)) {
    return false;
  }
  return qhx::prepare_lfm_model(model, last_error);
}

bool load_model_manifest(qhx_model& model, const std::string& json) {
  std::string family;
  if (!json_string(json, "name", model.name) || !json_string(json, "family", family) ||
      !json_string(json, "dsp_arch", model.dsp_arch)) {
    last_error = "Manifest is missing required model identity fields";
    return false;
  }

  const qhx_model_capability* capability =
      qhx::find_model_capability_by_manifest(model.name);
  if (!capability) {
    last_error = "Unsupported QHexRT manifest model: " + model.name;
    return false;
  }
  if (family != "llm" || capability->family != QHX_MODEL_FAMILY_LLM) {
    last_error = "Manifest family does not match the declared model capability";
    return false;
  }

  const qhx_hexagon_architectures manifest_architecture =
      architecture_mask(model.dsp_arch);
  if (manifest_architecture == QHX_HEXAGON_ARCH_NONE ||
      (capability->architectures & manifest_architecture) == 0U) {
    last_error = "Manifest architecture is not declared for model: " +
                 std::string(capability->catalog_id);
    return false;
  }
  if (model.dsp_arch != model.runtime->impl.arch) {
    last_error = "Model DSP architecture does not match runtime device";
    return false;
  }
  if (capability->support_state != QHX_MODEL_SUPPORT_EXECUTABLE) {
    last_error = "Model is declared but not executable in this QHexRT build: " +
                 std::string(capability->catalog_id);
    return false;
  }

  model.model_id = capability->model_id;
  model.runner = capability->runner;
  model.graph_contract = capability->graph_contract;
  switch (model.graph_contract) {
    case QHX_GRAPH_CONTRACT_LFM2_SPLIT_V1:
      return load_lfm2_split_v1(model, json);
    case QHX_GRAPH_CONTRACT_LFM2_SHARED_V1:
      return load_lfm2_shared_v1(model, json);
    default:
      last_error = "Executable model capability has no supported graph contract";
      return false;
  }
}

bool cancelled(qhx_session* session, const qhx_generate_options* options) {
  return session->cancelled.load(std::memory_order_acquire) ||
         (options && options->struct_size >= sizeof(qhx_generate_options) &&
          options->should_cancel && options->should_cancel(options->should_cancel_user));
}

}  // namespace

extern "C" {

qhx_runtime* qhx_runtime_create(const char* htp_library, const char* system_library) {
  auto runtime = std::unique_ptr<qhx_runtime>(new (std::nothrow) qhx_runtime());
  if (!runtime) return nullptr;
  if (!runtime->impl.initialize(htp_library, system_library)) {
    last_error = runtime->impl.error;
    return nullptr;
  }
  return runtime.release();
}

void qhx_runtime_free(qhx_runtime* runtime) { delete runtime; }

qhx_status qhx_runtime_device(qhx_runtime* runtime, char* arch, size_t arch_size,
                              int* soc_model, int* htp_device) {
  if (!runtime) return QHX_ERROR_INVALID_ARGUMENT;
  if (arch && arch_size) {
    std::strncpy(arch, runtime->impl.arch.c_str(), arch_size - 1);
    arch[arch_size - 1] = '\0';
  }
  if (soc_model) *soc_model = runtime->impl.soc_model;
  if (htp_device) *htp_device = 0;
  return QHX_OK;
}

qhx_model* qhx_model_load(qhx_runtime* runtime, const char* manifest_path,
                          const char* artifacts_dir) {
  if (!runtime || !manifest_path || !*manifest_path) return nullptr;
  last_error.clear();
  auto model = std::unique_ptr<qhx_model>(new (std::nothrow) qhx_model());
  if (!model) return nullptr;
  model->runtime = runtime;
  model->manifest_path = manifest_path;
  model->artifacts_dir = artifacts_dir && *artifacts_dir
                             ? artifacts_dir
                             : fs::path(manifest_path).parent_path().string();
  const std::string json = read_text(manifest_path);
  if (json.empty() || !load_model_manifest(*model, json)) return nullptr;
  return model.release();
}

void qhx_model_free(qhx_model* model) { delete model; }

qhx_session* qhx_session_create(qhx_model* model) {
  if (!model) return nullptr;
  auto* session = new (std::nothrow) qhx_session();
  if (session) session->model = model;
  return session;
}

void qhx_session_free(qhx_session* session) { delete session; }
void qhx_session_reset(qhx_session* session) {
  if (!session) return;
  session->cancelled.store(false, std::memory_order_release);
  session->output_text.clear();
  if (session->model && session->model->runner == QHX_RUNNER_LFM2)
    qhx::reset_lfm_session(*session);
}
void qhx_session_cancel(qhx_session* session) {
  if (session) session->cancelled.store(true, std::memory_order_release);
}

void qhx_gen_cfg_default(qhx_gen_cfg* config) {
  if (!config) return;
  *config = {};
  config->max_new_tokens = 64;
  config->temperature = 0.0F;
  config->top_p = 1.0F;
  config->top_k = 0;
  config->repetition_penalty = 1.0F;
}

void qhx_generate_options_default(qhx_generate_options* options) {
  if (!options) return;
  *options = {};
  options->struct_size = sizeof(*options);
}

qhx_status qhx_generate(qhx_session* session, const qhx_inputs* inputs,
                        const qhx_gen_cfg* config, qhx_token_cb callback,
                        void* callback_user, qhx_output* output) {
  qhx_generate_options options{};
  qhx_generate_options_default(&options);
  return qhx_generate_ex(session, inputs, config, &options, callback, callback_user, output);
}

qhx_status qhx_generate_ex(qhx_session* session, const qhx_inputs* inputs,
                           const qhx_gen_cfg* config, const qhx_generate_options* options,
                           qhx_token_cb callback, void* callback_user, qhx_output* output) {
  if (!session || !session->model || !inputs || !inputs->text || !output)
    return QHX_ERROR_INVALID_ARGUMENT;
  *output = {};
  if (cancelled(session, options)) return QHX_ERROR_CANCELLED;
  qhx_gen_cfg defaults;
  qhx_gen_cfg_default(&defaults);
  if (session->model->runner == QHX_RUNNER_LFM2 &&
      (session->model->graph_contract == QHX_GRAPH_CONTRACT_LFM2_SPLIT_V1 ||
       session->model->graph_contract == QHX_GRAPH_CONTRACT_LFM2_SHARED_V1)) {
    return qhx::run_lfm(*session, *inputs, config ? *config : defaults, options,
                        callback, callback_user, *output, last_error);
  }
  last_error = "Model has no executable runner in this QHexRT build";
  return QHX_ERROR_UNSUPPORTED;
}

const char* qhx_status_str(qhx_status status) {
  if (!last_error.empty() && status != QHX_OK) return last_error.c_str();
  switch (status) {
    case QHX_OK: return "ok";
    case QHX_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case QHX_ERROR_IO: return "I/O error";
    case QHX_ERROR_QNN_LOAD: return "QNN library load failed";
    case QHX_ERROR_QNN_API: return "QNN API failure";
    case QHX_ERROR_MODEL: return "model error";
    case QHX_ERROR_UNSUPPORTED: return "unsupported";
    case QHX_ERROR_CANCELLED: return "cancelled";
    case QHX_ERROR_OUT_OF_MEMORY: return "out of memory";
    default: return "internal error";
  }
}

const char* qhx_version(void) { return "QHexRT/0.1.0 ABI-1.3 QAIRT-2.48"; }

}  // extern "C"
