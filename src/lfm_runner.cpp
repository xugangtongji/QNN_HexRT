#include "lfm_runner.hpp"

#include "qhexrt_internal.hpp"

#include <android/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

namespace qhx {
namespace {

using Buffer = std::vector<uint8_t>;
using Buffers = std::vector<Buffer>;
using Clock = std::chrono::steady_clock;

struct TensorTransfer {
  size_t source = 0;
  size_t destination = 0;
};

struct LfmBindings {
  size_t prefill_x = 0;
  size_t prefill_cos = 0;
  size_t prefill_sin = 0;
  size_t prefill_mask = 0;
  size_t prefill_y = 0;
  size_t decode_x = 0;
  size_t decode_cos = 0;
  size_t decode_sin = 0;
  size_t decode_mask = 0;
  size_t decode_y = 0;
  size_t lmhead_input = 0;
  size_t lmhead_output = 0;
  std::vector<TensorTransfer> prefill_kv_to_decode;
  std::vector<TensorTransfer> prefill_conv_to_decode;
  std::vector<TensorTransfer> decode_kv_to_past;
  std::vector<TensorTransfer> decode_conv_to_past;
};

struct SampleCandidate {
  uint32_t token = 0;
  float logit = 0.0F;
  double weight = 0.0;
};

struct PerformanceTimings {
  double preparation_ms = 0.0;
  double prefill_graph_ms = 0.0;
  double decode_graph_ms = 0.0;
  double lmhead_graph_ms = 0.0;
  double sampling_ms = 0.0;
  double callback_ms = 0.0;
  int decode_executions = 0;
};

uint16_t fp16(float value) {
  _Float16 converted = static_cast<_Float16>(value);
  uint16_t bits;
  std::memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

float fp32(const uint8_t* value) {
  _Float16 converted;
  std::memcpy(&converted, value, sizeof(converted));
  return static_cast<float>(converted);
}

void set_half(Buffer& buffer, size_t index, float value) {
  const uint16_t bits = fp16(value);
  std::memcpy(buffer.data() + index * sizeof(bits), &bits, sizeof(bits));
}

std::optional<size_t> tensor_index(const std::vector<ContextGraph::TensorDef>& tensors,
                                   std::string_view name) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].name == name) return i;
  }
  return std::nullopt;
}

bool require_tensor(const std::vector<ContextGraph::TensorDef>& tensors, std::string_view name,
                    size_t& result, std::string& error) {
  const auto index = tensor_index(tensors, name);
  if (!index) {
    error = "LFM graph contract is missing tensor: " + std::string(name);
    return false;
  }
  result = *index;
  return true;
}

Buffers allocate_inputs(const std::vector<ContextGraph::TensorDef>& tensors,
                        const std::vector<size_t>& skipped = {}) {
  std::vector<bool> skip(tensors.size(), false);
  for (size_t index : skipped) {
    if (index < skip.size()) skip[index] = true;
  }
  Buffers result(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!skip[i]) result[i].resize(tensors[i].bytes);
  }
  return result;
}

bool validate_bytes(const ContextGraph::TensorDef& tensor, size_t expected,
                    std::string_view role, std::string& error) {
  if (tensor.bytes == expected) return true;
  error = "LFM graph tensor size mismatch for " + std::string(role);
  return false;
}

bool validate_fp16(const std::vector<ContextGraph::TensorDef>& tensors,
                   std::string_view graph, std::string& error) {
  for (const auto& tensor : tensors) {
    if (tensor.data_type != QNN_DATATYPE_FLOAT_16) {
      error = "LFM graph tensor is not FP16: " + std::string(graph) + "." + tensor.name;
      return false;
    }
  }
  return true;
}

bool create_bindings(const qhx_model& model, LfmBindings& bindings, std::string& error) {
  if (!require_tensor(model.prefill->inputs, "x", bindings.prefill_x, error) ||
      !require_tensor(model.prefill->inputs, "cos", bindings.prefill_cos, error) ||
      !require_tensor(model.prefill->inputs, "sin", bindings.prefill_sin, error) ||
      !require_tensor(model.prefill->inputs, "cmask", bindings.prefill_mask, error) ||
      !require_tensor(model.prefill->outputs, "y", bindings.prefill_y, error) ||
      !require_tensor(model.decode->inputs, "x", bindings.decode_x, error) ||
      !require_tensor(model.decode->inputs, "cos", bindings.decode_cos, error) ||
      !require_tensor(model.decode->inputs, "sin", bindings.decode_sin, error) ||
      !require_tensor(model.decode->inputs, "cmask", bindings.decode_mask, error) ||
      !require_tensor(model.decode->outputs, "y", bindings.decode_y, error)) {
    return false;
  }
  if (model.lmhead->inputs.size() != 1 || model.lmhead->outputs.size() != 1) {
    error = "LFM lmhead graph contract mismatch";
    return false;
  }
  if (!validate_fp16(model.prefill->inputs, "prefill.input", error) ||
      !validate_fp16(model.prefill->outputs, "prefill.output", error) ||
      !validate_fp16(model.decode->inputs, "decode.input", error) ||
      !validate_fp16(model.decode->outputs, "decode.output", error) ||
      !validate_fp16(model.lmhead->inputs, "lmhead.input", error) ||
      !validate_fp16(model.lmhead->outputs, "lmhead.output", error)) {
    return false;
  }

  const size_t hidden_bytes = static_cast<size_t>(model.hidden) * 2U;
  const size_t prefill_hidden_bytes = static_cast<size_t>(model.prefill_ctx) * hidden_bytes;
  const size_t rope_bytes = static_cast<size_t>(model.prefill_ctx) * model.head_dim * 2U;
  const size_t prefill_mask_bytes =
      static_cast<size_t>(model.prefill_ctx) * model.prefill_ctx * 2U;
  const size_t decode_mask_bytes = static_cast<size_t>(model.max_ctx) * 2U;
  const size_t logits_bytes = static_cast<size_t>(model.vocab) * 2U;
  if (!validate_bytes(model.prefill->inputs[bindings.prefill_x], prefill_hidden_bytes,
                      "prefill.x", error) ||
      !validate_bytes(model.prefill->inputs[bindings.prefill_cos], rope_bytes, "prefill.cos",
                      error) ||
      !validate_bytes(model.prefill->inputs[bindings.prefill_sin], rope_bytes, "prefill.sin",
                      error) ||
      !validate_bytes(model.prefill->inputs[bindings.prefill_mask], prefill_mask_bytes,
                      "prefill.cmask", error) ||
      !validate_bytes(model.prefill->outputs[bindings.prefill_y], prefill_hidden_bytes,
                      "prefill.y", error) ||
      !validate_bytes(model.decode->inputs[bindings.decode_x], hidden_bytes, "decode.x", error) ||
      !validate_bytes(model.decode->inputs[bindings.decode_cos],
                      static_cast<size_t>(model.head_dim) * 2U, "decode.cos", error) ||
      !validate_bytes(model.decode->inputs[bindings.decode_sin],
                      static_cast<size_t>(model.head_dim) * 2U, "decode.sin", error) ||
      !validate_bytes(model.decode->inputs[bindings.decode_mask], decode_mask_bytes,
                      "decode.cmask", error) ||
      !validate_bytes(model.decode->outputs[bindings.decode_y], hidden_bytes, "decode.y", error) ||
      !validate_bytes(model.lmhead->inputs[0], hidden_bytes, "lmhead.input", error) ||
      !validate_bytes(model.lmhead->outputs[0], logits_bytes, "lmhead.output", error)) {
    return false;
  }

  bindings.lmhead_input = 0;
  bindings.lmhead_output = 0;
  const size_t new_kv_bytes = static_cast<size_t>(model.kv_dim) * 2U;
  const size_t prefill_kv_bytes = static_cast<size_t>(model.prefill_ctx) * new_kv_bytes;
  const size_t past_kv_bytes = static_cast<size_t>(model.max_ctx) * new_kv_bytes;
  for (size_t destination = 0; destination < model.decode->inputs.size(); ++destination) {
    const auto& input = model.decode->inputs[destination];
    if (input.name.rfind("past_k_", 0) == 0 || input.name.rfind("past_v_", 0) == 0) {
      const std::string source_name = "new_" + input.name.substr(5);
      const auto source = tensor_index(model.prefill->outputs, source_name);
      if (!source || model.prefill->outputs[*source].bytes != prefill_kv_bytes ||
          input.bytes != past_kv_bytes || prefill_kv_bytes > past_kv_bytes) {
        error = "LFM prefill/decode KV contract mismatch: " + input.name;
        return false;
      }
      bindings.prefill_kv_to_decode.push_back({*source, destination});
    } else if (input.name.rfind("past_conv_", 0) == 0) {
      const std::string source_name = "conv_bx_" + input.name.substr(10);
      const auto source = tensor_index(model.prefill->outputs, source_name);
      const size_t expected_conv = static_cast<size_t>(model.hidden) * 3U * 2U;
      if (!source || input.bytes != expected_conv ||
          model.prefill->outputs[*source].bytes < prefill_hidden_bytes) {
        error = "LFM prefill/decode conv contract mismatch: " + input.name;
        return false;
      }
      bindings.prefill_conv_to_decode.push_back({*source, destination});
    }
  }

  for (size_t source = 0; source < model.decode->outputs.size(); ++source) {
    const auto& output = model.decode->outputs[source];
    if (output.name.rfind("new_k_", 0) == 0 || output.name.rfind("new_v_", 0) == 0) {
      const std::string destination_name = "past_" + output.name.substr(4);
      const auto destination = tensor_index(model.decode->inputs, destination_name);
      if (!destination || output.bytes != new_kv_bytes ||
          model.decode->inputs[*destination].bytes != past_kv_bytes) {
        error = "LFM decode KV contract mismatch: " + output.name;
        return false;
      }
      bindings.decode_kv_to_past.push_back({source, *destination});
    } else if (output.name.rfind("present_conv_", 0) == 0) {
      const std::string destination_name = "past_conv_" + output.name.substr(13);
      const auto destination = tensor_index(model.decode->inputs, destination_name);
      if (!destination || output.bytes != model.decode->inputs[*destination].bytes) {
        error = "LFM decode conv contract mismatch: " + output.name;
        return false;
      }
      bindings.decode_conv_to_past.push_back({source, *destination});
    }
  }

  const size_t mapped_decode_inputs = 4U + bindings.prefill_kv_to_decode.size() +
                                      bindings.prefill_conv_to_decode.size();
  const size_t mapped_prefill_outputs = 1U + bindings.prefill_kv_to_decode.size() +
                                        bindings.prefill_conv_to_decode.size();
  const size_t mapped_decode_outputs = 1U + bindings.decode_kv_to_past.size() +
                                       bindings.decode_conv_to_past.size();
  if (mapped_decode_inputs != model.decode->inputs.size() ||
      mapped_prefill_outputs != model.prefill->outputs.size() ||
      mapped_decode_outputs != model.decode->outputs.size() ||
      bindings.prefill_kv_to_decode.size() != bindings.decode_kv_to_past.size() ||
      bindings.prefill_conv_to_decode.size() != bindings.decode_conv_to_past.size()) {
    error = "LFM graph contract contains unsupported tensors";
    return false;
  }
  return true;
}

void fill_rope_table(Buffer& cos_buffer, Buffer& sin_buffer, int rows, int head_dim,
                     double theta) {
  const int half = head_dim / 2;
  std::vector<double> inverse_frequency(static_cast<size_t>(half));
  for (int d = 0; d < half; ++d)
    inverse_frequency[static_cast<size_t>(d)] = std::pow(theta, -2.0 * d / head_dim);
  for (int row = 0; row < rows; ++row) {
    for (int d = 0; d < half; ++d) {
      const double angle = row * inverse_frequency[static_cast<size_t>(d)];
      const float cosine = static_cast<float>(std::cos(angle));
      const float sine = static_cast<float>(std::sin(angle));
      const size_t offset = static_cast<size_t>(row) * head_dim;
      set_half(cos_buffer, offset + d, cosine);
      set_half(cos_buffer, offset + half + d, cosine);
      set_half(sin_buffer, offset + d, sine);
      set_half(sin_buffer, offset + half + d, sine);
    }
  }
}

void fill_prefill_mask(Buffer& mask, int context) {
  for (int query = 0; query < context; ++query) {
    for (int key = 0; key < context; ++key) {
      set_half(mask, static_cast<size_t>(query) * context + key,
               key <= query ? 0.0F : -std::numeric_limits<float>::infinity());
    }
  }
}

void fill_decode_mask_base(Buffer& mask, int context) {
  for (int key = 0; key < context; ++key)
    set_half(mask, static_cast<size_t>(key), -std::numeric_limits<float>::infinity());
}

bool put_embedding(const qhx_model& model, int32_t token, Buffer& destination, size_t row) {
  if (token < 0 || token >= model.vocab) return false;
  const size_t bytes = static_cast<size_t>(model.hidden) * 2U;
  if (model.embedding.size() < bytes || destination.size() < bytes) return false;
  const size_t source_offset = static_cast<size_t>(token) * bytes;
  const size_t destination_offset = row * bytes;
  if (source_offset > model.embedding.size() - bytes ||
      destination_offset > destination.size() - bytes) {
    return false;
  }
  std::memcpy(destination.data() + destination_offset, model.embedding.data() + source_offset,
              bytes);
  return true;
}

void initialize_conv_state(const qhx_model& model, const LfmBindings& bindings,
                           const Buffers& prefill_outputs, Buffers& decode_inputs,
                           size_t prompt_size) {
  for (const auto& transfer : bindings.prefill_conv_to_decode) {
    const Buffer& source = prefill_outputs[transfer.source];
    Buffer& destination = decode_inputs[transfer.destination];
    for (int hidden = 0; hidden < model.hidden; ++hidden) {
      for (int kernel = 0; kernel < 3; ++kernel) {
        const int row =
            std::max(0, static_cast<int>(prompt_size) - 3 + kernel);
        const size_t source_offset =
            (static_cast<size_t>(row) * model.hidden + hidden) * 2U;
        const size_t destination_offset =
            (static_cast<size_t>(hidden) * 3U + kernel) * 2U;
        std::memcpy(destination.data() + destination_offset, source.data() + source_offset, 2U);
      }
    }
  }
}

bool is_cancelled(qhx_session& session, const qhx_generate_options* options) {
  return session.cancelled.load(std::memory_order_acquire) ||
         (options && options->struct_size >= sizeof(qhx_generate_options) &&
          options->should_cancel && options->should_cancel(options->should_cancel_user));
}

std::string chat_prompt(const qhx_inputs& input) {
  if (input.no_template) return input.text ? input.text : "";
  size_t capacity = 64U + std::strlen(input.text ? input.text : "");
  capacity += input.system_prompt ? std::strlen(input.system_prompt) : 0U;
  for (int i = 0; input.history && i < input.n_history; ++i)
    capacity += 32U + (input.history[i] ? std::strlen(input.history[i]) : 0U);
  std::string text;
  text.reserve(capacity);
  if (input.system_prompt && *input.system_prompt)
    text += "<|im_start|>system\n" + std::string(input.system_prompt) + "<|im_end|>\n";
  for (int i = 0; input.history && i < input.n_history; ++i) {
    text += i % 2 == 0 ? "<|im_start|>user\n" : "<|im_start|>assistant\n";
    text += input.history[i] ? input.history[i] : "";
    text += "<|im_end|>\n";
  }
  text += "<|im_start|>user\n" + std::string(input.text ? input.text : "") +
          "<|im_end|>\n<|im_start|>assistant\n";
  return text;
}

bool stopped(const std::string& text, const qhx_gen_cfg& config) {
  for (int i = 0; config.stop_strings && i < config.n_stop_strings; ++i) {
    const char* stop = config.stop_strings[i];
    if (stop && *stop && text.size() >= std::strlen(stop) &&
        text.compare(text.size() - std::strlen(stop), std::strlen(stop), stop) == 0) {
      return true;
    }
  }
  return false;
}

bool enabled_value(const char* value) {
  return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
                   std::strcmp(value, "on") == 0);
}

bool performance_logging_enabled() {
  static const bool enabled = [] {
    if (enabled_value(std::getenv("QHEXRT_PROFILE"))) return true;
    char property[PROP_VALUE_MAX]{};
    return __system_property_get("debug.qhexrt.profile", property) > 0 && enabled_value(property);
  }();
  return enabled;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Callable>
auto measure(bool enabled, double& accumulator, Callable&& callable) {
  if (!enabled) return callable();
  const auto start = Clock::now();
  auto result = callable();
  accumulator += elapsed_ms(start, Clock::now());
  return result;
}

void log_performance(const PerformanceTimings& timings, int prompt_tokens, int generated_tokens) {
  __android_log_print(
      ANDROID_LOG_INFO, "QHexRTPerf",
      "prompt=%d generated=%d prep=%.3fms prefill_graph=%.3fms decode_graph=%.3fms "
      "lmhead=%.3fms sampling=%.3fms callback=%.3fms decode_exec=%d",
      prompt_tokens, generated_tokens, timings.preparation_ms, timings.prefill_graph_ms,
      timings.decode_graph_ms, timings.lmhead_graph_ms, timings.sampling_ms,
      timings.callback_ms, timings.decode_executions);
}

}  // namespace

struct LfmWorkspace {
  LfmBindings bindings;
  ContextGraph::Bindings prefill_qnn;
  ContextGraph::Bindings decode_qnn;
  ContextGraph::Bindings lmhead_qnn;
  Buffers prefill_inputs;
  Buffers prefill_outputs;
  Buffers decode_inputs;
  Buffers decode_outputs;
  Buffers lmhead_inputs;
  Buffers lmhead_outputs;
  std::vector<int32_t> tokens;
  std::vector<uint8_t> seen;
  std::vector<int32_t> seen_tokens;
  std::vector<SampleCandidate> candidates;
  PerformanceTimings timings;

  bool initialize(const qhx_model& model, std::string& error) {
    if (!create_bindings(model, bindings, error)) return false;
    prefill_qnn = model.prefill->create_bindings();
    decode_qnn = model.decode->create_bindings();
    lmhead_qnn = model.lmhead->create_bindings();
    prefill_inputs = allocate_inputs(model.prefill->inputs);
    prefill_outputs.resize(model.prefill->outputs.size());
    std::vector<size_t> skipped_decode_inputs;
    skipped_decode_inputs.reserve(bindings.prefill_kv_to_decode.size());
    for (const auto& transfer : bindings.prefill_kv_to_decode) {
      if (model.prefill->outputs[transfer.source].bytes ==
          model.decode->inputs[transfer.destination].bytes) {
        skipped_decode_inputs.push_back(transfer.destination);
      }
    }
    decode_inputs = allocate_inputs(model.decode->inputs, skipped_decode_inputs);
    decode_outputs.resize(model.decode->outputs.size());
    lmhead_inputs = allocate_inputs(model.lmhead->inputs);
    lmhead_outputs.resize(model.lmhead->outputs.size());
    tokens.reserve(static_cast<size_t>(model.max_ctx));
    seen.resize(static_cast<size_t>(model.vocab));
    seen_tokens.reserve(static_cast<size_t>(model.max_ctx));
    std::memcpy(prefill_inputs[bindings.prefill_cos].data(), model.rope_cos.data(),
                prefill_inputs[bindings.prefill_cos].size());
    std::memcpy(prefill_inputs[bindings.prefill_sin].data(), model.rope_sin.data(),
                prefill_inputs[bindings.prefill_sin].size());
    std::memcpy(prefill_inputs[bindings.prefill_mask].data(), model.prefill_mask.data(),
                model.prefill_mask.size());
    return true;
  }

  void clear_seen() noexcept {
    for (int32_t token : seen_tokens) seen[static_cast<size_t>(token)] = 0;
    seen_tokens.clear();
  }

  void mark_seen(int32_t token) {
    if (token < 0 || static_cast<size_t>(token) >= seen.size() || seen[token]) return;
    seen[token] = 1;
    seen_tokens.push_back(token);
  }

  void reset_logical_state() noexcept {
    clear_seen();
    tokens.clear();
    candidates.clear();
    timings = {};
  }

  void begin_generation(const qhx_model& model) {
    reset_logical_state();
    std::fill(prefill_inputs[bindings.prefill_x].begin(),
              prefill_inputs[bindings.prefill_x].end(), 0U);
    std::memcpy(decode_inputs[bindings.decode_mask].data(), model.decode_mask_base.data(),
                model.decode_mask_base.size());
  }
};

namespace {

class KvStateLease {
 public:
  KvStateLease(const LfmBindings& bindings, Buffers& prefill_outputs, Buffers& decode_inputs)
      : bindings_(bindings), prefill_outputs_(prefill_outputs), decode_inputs_(decode_inputs) {
    for (const auto& transfer : bindings_.prefill_kv_to_decode) {
      Buffer& source = prefill_outputs_[transfer.source];
      Buffer& destination = decode_inputs_[transfer.destination];
      if (source.size() == destination.size()) {
        source.swap(destination);
        swapped_.push_back(transfer);
      } else {
        std::memcpy(destination.data(), source.data(), source.size());
      }
    }
  }

  ~KvStateLease() {
    for (const auto& transfer : swapped_)
      prefill_outputs_[transfer.source].swap(decode_inputs_[transfer.destination]);
  }
  KvStateLease(const KvStateLease&) = delete;
  KvStateLease& operator=(const KvStateLease&) = delete;

 private:
  const LfmBindings& bindings_;
  Buffers& prefill_outputs_;
  Buffers& decode_inputs_;
  std::vector<TensorTransfer> swapped_;
};

float adjusted_logit(const Buffer& logits, size_t token, const qhx_gen_cfg& config,
                     const LfmWorkspace& workspace) {
  float value = fp32(logits.data() + token * 2U);
  if (config.repetition_penalty > 0.0F && config.repetition_penalty != 1.0F &&
      workspace.seen[token]) {
    value = value < 0.0F ? value * config.repetition_penalty
                         : value / config.repetition_penalty;
  }
  return value;
}

bool candidate_precedes(const SampleCandidate& lhs, const SampleCandidate& rhs) {
  if (lhs.logit > rhs.logit) return true;
  if (lhs.logit < rhs.logit) return false;
  return lhs.token < rhs.token;
}

int32_t greedy_sample(const Buffer& logits, const qhx_gen_cfg& config,
                      const LfmWorkspace& workspace) {
  const size_t count = logits.size() / 2U;
  size_t best_token = 0;
  float best_logit = adjusted_logit(logits, 0, config, workspace);
  for (size_t token = 1; token < count; ++token) {
    const float value = adjusted_logit(logits, token, config, workspace);
    if (value > best_logit) {
      best_logit = value;
      best_token = token;
    }
  }
  return static_cast<int32_t>(best_token);
}

int32_t draw_linear(const Buffer& logits, const qhx_gen_cfg& config,
                    const LfmWorkspace& workspace, std::mt19937_64& rng) {
  const size_t count = logits.size() / 2U;
  float maximum = -std::numeric_limits<float>::infinity();
  for (size_t token = 0; token < count; ++token)
    maximum = std::max(maximum, adjusted_logit(logits, token, config, workspace));
  double total = 0.0;
  for (size_t token = 0; token < count; ++token) {
    total += std::exp((adjusted_logit(logits, token, config, workspace) - maximum) /
                      config.temperature);
  }
  if (!(total > 0.0) || !std::isfinite(total))
    return greedy_sample(logits, config, workspace);
  const double target = std::uniform_real_distribution<double>(0.0, total)(rng);
  double cumulative = 0.0;
  for (size_t token = 0; token < count; ++token) {
    cumulative += std::exp((adjusted_logit(logits, token, config, workspace) - maximum) /
                           config.temperature);
    if (target <= cumulative) return static_cast<int32_t>(token);
  }
  return static_cast<int32_t>(count - 1U);
}

int32_t draw_candidates(std::vector<SampleCandidate>& candidates, double total,
                        std::mt19937_64& rng) {
  if (!(total > 0.0) || !std::isfinite(total))
    return static_cast<int32_t>(candidates.front().token);
  const double target = std::uniform_real_distribution<double>(0.0, total)(rng);
  double cumulative = 0.0;
  for (const auto& candidate : candidates) {
    cumulative += candidate.weight;
    if (target <= cumulative) return static_cast<int32_t>(candidate.token);
  }
  return static_cast<int32_t>(candidates.back().token);
}

int32_t filtered_sample(const Buffer& logits, const qhx_gen_cfg& config,
                        LfmWorkspace& workspace, std::mt19937_64& rng) {
  const size_t count = logits.size() / 2U;
  workspace.candidates.resize(count);
  for (size_t token = 0; token < count; ++token) {
    workspace.candidates[token] =
        {static_cast<uint32_t>(token), adjusted_logit(logits, token, config, workspace), 0.0};
  }

  const size_t top_k = config.top_k > 0
                           ? std::min(count, static_cast<size_t>(config.top_k))
                           : count;
  if (top_k < count) {
    std::nth_element(workspace.candidates.begin(), workspace.candidates.begin() + top_k,
                     workspace.candidates.end(), candidate_precedes);
    workspace.candidates.resize(top_k);
  }

  const bool nucleus = config.top_p > 0.0F && config.top_p < 1.0F;
  if (nucleus)
    std::sort(workspace.candidates.begin(), workspace.candidates.end(), candidate_precedes);
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto& candidate : workspace.candidates)
    maximum = std::max(maximum, candidate.logit);
  double total = 0.0;
  for (auto& candidate : workspace.candidates) {
    candidate.weight = std::exp((candidate.logit - maximum) / config.temperature);
    total += candidate.weight;
  }

  if (nucleus && total > 0.0 && std::isfinite(total)) {
    double retained_total = 0.0;
    size_t retained_count = 0;
    do {
      retained_total += workspace.candidates[retained_count++].weight;
    } while (retained_count < workspace.candidates.size() &&
             retained_total / total < config.top_p);
    workspace.candidates.resize(retained_count);
    total = retained_total;
  }
  return draw_candidates(workspace.candidates, total, rng);
}

int32_t sample(const Buffer& logits, const qhx_gen_cfg& config, LfmWorkspace& workspace,
               std::mt19937_64& rng) {
  if (config.temperature <= 0.0F) return greedy_sample(logits, config, workspace);
  const bool nucleus = config.top_p > 0.0F && config.top_p < 1.0F;
  if (config.top_k <= 0 && !nucleus) return draw_linear(logits, config, workspace, rng);
  return filtered_sample(logits, config, workspace, rng);
}

void copy_rope_row(const qhx_model& model, const LfmBindings& bindings, int position,
                   Buffers& decode_inputs) {
  const size_t row_bytes = static_cast<size_t>(model.head_dim) * 2U;
  const size_t offset = static_cast<size_t>(position) * row_bytes;
  std::memcpy(decode_inputs[bindings.decode_cos].data(), model.rope_cos.data() + offset,
              row_bytes);
  std::memcpy(decode_inputs[bindings.decode_sin].data(), model.rope_sin.data() + offset,
              row_bytes);
}

}  // namespace

bool prepare_lfm_model(qhx_model& model, std::string& error) {
  size_t x_index = 0;
  size_t cos_index = 0;
  size_t sin_index = 0;
  size_t mask_index = 0;
  if (!require_tensor(model.prefill->inputs, "x", x_index, error) ||
      !require_tensor(model.prefill->inputs, "cos", cos_index, error) ||
      !require_tensor(model.prefill->inputs, "sin", sin_index, error) ||
      !require_tensor(model.prefill->inputs, "cmask", mask_index, error)) {
    return false;
  }
  const auto& x = model.prefill->inputs[x_index];
  if (x.dimensions.size() != 2 || x.dimensions[0] == 0 ||
      x.dimensions[0] > static_cast<uint32_t>(model.max_ctx) ||
      x.dimensions[1] != static_cast<uint32_t>(model.hidden)) {
    error = "LFM prefill.x dimensions do not match the model contract";
    return false;
  }
  model.prefill_ctx = static_cast<int>(x.dimensions[0]);
  const size_t prefill_rope_bytes =
      static_cast<size_t>(model.prefill_ctx) * model.head_dim * 2U;
  const size_t mask_bytes =
      static_cast<size_t>(model.prefill_ctx) * model.prefill_ctx * 2U;
  if (!validate_bytes(model.prefill->inputs[cos_index], prefill_rope_bytes,
                      "prefill.cos", error) ||
      !validate_bytes(model.prefill->inputs[sin_index], prefill_rope_bytes,
                      "prefill.sin", error) ||
      !validate_bytes(model.prefill->inputs[mask_index], mask_bytes, "prefill.cmask", error)) {
    return false;
  }
  const size_t rope_bytes = static_cast<size_t>(model.max_ctx) * model.head_dim * 2U;
  model.rope_cos.resize(rope_bytes);
  model.rope_sin.resize(rope_bytes);
  model.prefill_mask.resize(mask_bytes);
  model.decode_mask_base.resize(static_cast<size_t>(model.max_ctx) * 2U);
  fill_rope_table(model.rope_cos, model.rope_sin, model.max_ctx, model.head_dim,
                  model.rope_theta);
  fill_prefill_mask(model.prefill_mask, model.prefill_ctx);
  fill_decode_mask_base(model.decode_mask_base, model.max_ctx);
  return true;
}

void reset_lfm_session(qhx_session& session) noexcept {
  if (session.workspace) session.workspace->reset_logical_state();
}

qhx_status run_lfm(qhx_session& session, const qhx_inputs& inputs, const qhx_gen_cfg& config,
                   const qhx_generate_options* options, qhx_token_cb callback,
                   void* callback_user, qhx_output& output, std::string& error) {
  qhx_model& model = *session.model;
  const bool profile = performance_logging_enabled();
  const auto initialize_workspace_start = profile ? Clock::now() : Clock::time_point{};
  if (!session.workspace) {
    auto workspace = std::make_unique<LfmWorkspace>();
    if (!workspace->initialize(model, error)) return QHX_ERROR_MODEL;
    session.workspace = std::move(workspace);
  }
  LfmWorkspace& workspace = *session.workspace;
  workspace.begin_generation(model);
  if (profile)
    workspace.timings.preparation_ms += elapsed_ms(initialize_workspace_start, Clock::now());
  workspace.tokens.push_back(model.bos_token_id);
  if (!model.tokenizer->encode(chat_prompt(inputs), workspace.tokens, error))
    return QHX_ERROR_MODEL;
  if (workspace.tokens.empty() ||
      static_cast<int>(workspace.tokens.size()) >= model.prefill_ctx) {
    error = "Prompt exceeds the LFM prefill window";
    return QHX_ERROR_INVALID_ARGUMENT;
  }
  for (int32_t token : workspace.tokens) workspace.mark_seen(token);

  const auto prefill_start = Clock::now();
  const auto prepare_prefill_start = profile ? Clock::now() : Clock::time_point{};
  for (size_t row = 0; row < workspace.tokens.size(); ++row) {
    if (!put_embedding(model, workspace.tokens[row],
                       workspace.prefill_inputs[workspace.bindings.prefill_x], row)) {
      error = "Embedding token is outside the mapped table";
      return QHX_ERROR_MODEL;
    }
  }
  if (profile)
    workspace.timings.preparation_ms += elapsed_ms(prepare_prefill_start, Clock::now());
  if (is_cancelled(session, options)) return QHX_ERROR_CANCELLED;
  if (!measure(profile, workspace.timings.prefill_graph_ms, [&] {
        return model.prefill->execute(workspace.prefill_qnn, workspace.prefill_inputs,
                                      workspace.prefill_outputs, error);
      })) {
    return QHX_ERROR_QNN_API;
  }

  const size_t hidden_bytes = static_cast<size_t>(model.hidden) * 2U;
  const auto prepare_lmhead_start = profile ? Clock::now() : Clock::time_point{};
  std::memcpy(workspace.lmhead_inputs[workspace.bindings.lmhead_input].data(),
              workspace.prefill_outputs[workspace.bindings.prefill_y].data() +
                  (workspace.tokens.size() - 1U) * hidden_bytes,
              hidden_bytes);
  if (profile)
    workspace.timings.preparation_ms += elapsed_ms(prepare_lmhead_start, Clock::now());
  if (!measure(profile, workspace.timings.lmhead_graph_ms, [&] {
        return model.lmhead->execute(workspace.lmhead_qnn, workspace.lmhead_inputs,
                                     workspace.lmhead_outputs, error);
      })) {
    return QHX_ERROR_QNN_API;
  }
  const auto prefill_end = Clock::now();

  const auto prepare_decode_start = profile ? Clock::now() : Clock::time_point{};
  KvStateLease kv_state(workspace.bindings, workspace.prefill_outputs,
                        workspace.decode_inputs);
  initialize_conv_state(model, workspace.bindings, workspace.prefill_outputs,
                        workspace.decode_inputs, workspace.tokens.size());
  for (size_t key = 0; key < workspace.tokens.size(); ++key)
    set_half(workspace.decode_inputs[workspace.bindings.decode_mask], key, 0.0F);
  if (profile)
    workspace.timings.preparation_ms += elapsed_ms(prepare_decode_start, Clock::now());

  std::mt19937_64 rng(config.seed ? config.seed : std::random_device{}());
  const int token_limit = std::max(0, config.max_new_tokens);
  int32_t next = model.eos_token_id;
  if (token_limit > 0) {
    next = measure(profile, workspace.timings.sampling_ms, [&] {
      return sample(workspace.lmhead_outputs[workspace.bindings.lmhead_output], config, workspace,
                    rng);
    });
  }
  const auto decode_start = Clock::now();
  const int prompt_count = static_cast<int>(workspace.tokens.size());
  int generated = 0;
  while (generated < token_limit && next != model.eos_token_id) {
    if (is_cancelled(session, options)) return QHX_ERROR_CANCELLED;
    workspace.tokens.push_back(next);
    workspace.mark_seen(next);
    const std::string piece = model.tokenizer->decode(next);
    session.output_text += piece;
    ++generated;
    if (callback) {
      const int keep_going = measure(profile, workspace.timings.callback_ms, [&] {
        return callback(callback_user, piece.data(), static_cast<int>(piece.size()), next, 0);
      });
      if (keep_going == 0) return QHX_ERROR_CANCELLED;
    }
    if (stopped(session.output_text, config)) break;
    if (generated >= token_limit) break;
    const int position = static_cast<int>(workspace.tokens.size()) - 1;
    if (position >= model.max_ctx) break;
    const auto prepare_token_start = profile ? Clock::now() : Clock::time_point{};
    if (!put_embedding(model, next, workspace.decode_inputs[workspace.bindings.decode_x], 0)) {
      error = "Embedding token is outside the mapped table";
      return QHX_ERROR_MODEL;
    }
    copy_rope_row(model, workspace.bindings, position, workspace.decode_inputs);
    set_half(workspace.decode_inputs[workspace.bindings.decode_mask],
             static_cast<size_t>(position), 0.0F);
    if (profile)
      workspace.timings.preparation_ms += elapsed_ms(prepare_token_start, Clock::now());
    if (!measure(profile, workspace.timings.decode_graph_ms, [&] {
          return model.decode->execute(workspace.decode_qnn, workspace.decode_inputs,
                                       workspace.decode_outputs, error);
        })) {
      return QHX_ERROR_QNN_API;
    }
    ++workspace.timings.decode_executions;
    const auto update_state_start = profile ? Clock::now() : Clock::time_point{};
    for (const auto& transfer : workspace.bindings.decode_kv_to_past) {
      std::memcpy(workspace.decode_inputs[transfer.destination].data() +
                      static_cast<size_t>(position) * model.kv_dim * 2U,
                  workspace.decode_outputs[transfer.source].data(),
                  static_cast<size_t>(model.kv_dim) * 2U);
    }
    for (const auto& transfer : workspace.bindings.decode_conv_to_past) {
      workspace.decode_inputs[transfer.destination].swap(
          workspace.decode_outputs[transfer.source]);
    }
    std::memcpy(workspace.lmhead_inputs[workspace.bindings.lmhead_input].data(),
                workspace.decode_outputs[workspace.bindings.decode_y].data(), hidden_bytes);
    if (profile)
      workspace.timings.preparation_ms += elapsed_ms(update_state_start, Clock::now());
    if (!measure(profile, workspace.timings.lmhead_graph_ms, [&] {
          return model.lmhead->execute(workspace.lmhead_qnn, workspace.lmhead_inputs,
                                       workspace.lmhead_outputs, error);
        })) {
      return QHX_ERROR_QNN_API;
    }
    next = measure(profile, workspace.timings.sampling_ms, [&] {
      return sample(workspace.lmhead_outputs[workspace.bindings.lmhead_output], config, workspace,
                    rng);
    });
  }
  const auto decode_end = Clock::now();
  if (callback) {
    (void)measure(profile, workspace.timings.callback_ms,
                  [&] { return callback(callback_user, nullptr, 0, -1, 1); });
  }
  output.text = session.output_text.c_str();
  output.n_prompt = prompt_count;
  output.n_generated = generated;
  output.prefill_ms = elapsed_ms(prefill_start, prefill_end);
  output.decode_ms = elapsed_ms(decode_start, decode_end);
  if (profile) log_performance(workspace.timings, prompt_count, generated);
  return QHX_OK;
}

}  // namespace qhx

qhx_session::qhx_session() = default;
qhx_session::~qhx_session() = default;
