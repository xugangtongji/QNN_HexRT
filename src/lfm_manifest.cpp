#include "lfm_manifest.hpp"

#include <cctype>
#include <optional>
#include <string_view>
#include <type_traits>

namespace qhx {
namespace {

std::optional<size_t> value_start(std::string_view json, std::string_view key) {
  const std::string quoted = "\"" + std::string(key) + "\"";
  const size_t key_position = json.find(quoted);
  if (key_position == std::string_view::npos) return std::nullopt;
  size_t position = json.find(':', key_position + quoted.size());
  if (position == std::string_view::npos) return std::nullopt;
  do {
    ++position;
  } while (position < json.size() &&
           std::isspace(static_cast<unsigned char>(json[position])));
  return position < json.size() ? std::optional<size_t>(position) : std::nullopt;
}

bool delimited_value(std::string_view json, std::string_view key, char open,
                     char close, std::string_view& value) {
  const auto start = value_start(json, key);
  if (!start || json[*start] != open) return false;
  size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t position = *start; position < json.size(); ++position) {
    const char current = json[position];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (current == '\\') {
        escaped = true;
      } else if (current == '"') {
        in_string = false;
      }
      continue;
    }
    if (current == '"') {
      in_string = true;
    } else if (current == open) {
      ++depth;
    } else if (current == close) {
      if (--depth == 0) {
        value = json.substr(*start + 1, position - *start - 1);
        return true;
      }
    }
  }
  return false;
}

bool object_value(std::string_view json, std::string_view key,
                  std::string_view& value) {
  return delimited_value(json, key, '{', '}', value);
}

bool array_value(std::string_view json, std::string_view key,
                 std::string_view& value) {
  return delimited_value(json, key, '[', ']', value);
}

bool string_value(std::string_view json, std::string_view key,
                  std::string& value) {
  const auto start = value_start(json, key);
  if (!start || json[*start] != '"') return false;
  value.clear();
  bool escaped = false;
  for (size_t position = *start + 1; position < json.size(); ++position) {
    const char current = json[position];
    if (escaped) {
      switch (current) {
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case '\\':
        case '"': value.push_back(current); break;
        default: return false;
      }
      escaped = false;
    } else if (current == '\\') {
      escaped = true;
    } else if (current == '"') {
      return true;
    } else {
      value.push_back(current);
    }
  }
  return false;
}

template <typename Number>
bool number_value(std::string_view json, std::string_view key, Number& value) {
  const auto start = value_start(json, key);
  if (!start) return false;
  size_t end = *start;
  while (end < json.size()) {
    const char current = json[end];
    if (!(std::isdigit(static_cast<unsigned char>(current)) || current == '-' ||
          current == '+' || current == '.' || current == 'e' || current == 'E')) {
      break;
    }
    ++end;
  }
  if (end == *start) return false;
  try {
    if constexpr (std::is_same_v<Number, int>) {
      value = std::stoi(std::string(json.substr(*start, end - *start)));
    } else {
      value = std::stod(std::string(json.substr(*start, end - *start)));
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool graph_binary(std::string_view contexts, std::string_view key,
                  std::string& path) {
  std::string_view graph;
  return object_value(contexts, key, graph) && string_value(graph, "bin", path);
}

bool parse_graphs(std::string_view root, LfmManifest& manifest) {
  std::string_view artifacts;
  std::string_view contexts;
  std::string_view plan;
  std::string_view plan_parameters;
  if (!object_value(root, "artifacts", artifacts) ||
      !object_value(artifacts, "contexts", contexts) ||
      !object_value(root, "plan", plan) ||
      !object_value(plan, "params", plan_parameters) ||
      !graph_binary(contexts, "lmhead", manifest.lmhead.binary_path) ||
      !string_value(plan_parameters, "prefill", manifest.prefill.graph_name) ||
      !string_value(plan_parameters, "decode", manifest.decode.graph_name) ||
      !string_value(plan_parameters, "lmhead", manifest.lmhead.graph_name) ||
      !string_value(artifacts, "embed", manifest.embedding_path) ||
      !string_value(artifacts, "tokenizer", manifest.tokenizer_path)) {
    return false;
  }

  std::string shared_binary;
  if (graph_binary(contexts, "shared", shared_binary)) {
    manifest.context_layout = LfmContextLayout::kSharedBody;
    manifest.prefill.binary_path = shared_binary;
    manifest.decode.binary_path = shared_binary;
    return true;
  }
  manifest.context_layout = LfmContextLayout::kSeparate;
  return graph_binary(contexts, "prefill", manifest.prefill.binary_path) &&
         graph_binary(contexts, "decode", manifest.decode.binary_path);
}

}  // namespace

bool parse_lfm_manifest_v1(const std::string& json, LfmManifest& manifest,
                           std::string& error) {
  manifest = {};
  std::string_view root(json);
  std::string_view model;
  std::string_view parameters;
  if (!number_value(root, "schema_version", manifest.schema_version) ||
      manifest.schema_version != 1 || !object_value(root, "model", model) ||
      !object_value(root, "params", parameters) ||
      !string_value(model, "name", manifest.name) ||
      !string_value(model, "family", manifest.family) ||
      !string_value(model, "dsp_arch", manifest.dsp_arch) ||
      !number_value(parameters, "hidden", manifest.hidden) ||
      !number_value(parameters, "vocab", manifest.vocab) ||
      !number_value(parameters, "n_layers", manifest.layers) ||
      !number_value(parameters, "max_ctx", manifest.max_context) ||
      !number_value(parameters, "kv_dim", manifest.kv_dimension) ||
      !number_value(parameters, "head_dim", manifest.head_dimension) ||
      !number_value(parameters, "rope_theta", manifest.rope_theta) ||
      !number_value(parameters, "eos_token_id", manifest.eos_token_id) ||
      !parse_graphs(root, manifest)) {
    error = "Invalid LFM schema-v1 manifest";
    return false;
  }

  std::string_view chat;
  std::string_view bos;
  if (!object_value(root, "chat", chat) || !array_value(chat, "bos", bos) ||
      !number_value(std::string("{\"value\":") + std::string(bos) + "}",
                    "value", manifest.bos_token_id)) {
    error = "LFM schema-v1 manifest is missing chat BOS metadata";
    return false;
  }
  if (manifest.hidden <= 0 || manifest.vocab <= 0 || manifest.layers <= 0 ||
      manifest.max_context <= 0 || manifest.kv_dimension <= 0 ||
      manifest.head_dimension <= 0 || manifest.bos_token_id < 0 ||
      manifest.eos_token_id < 0 || manifest.rope_theta <= 0.0) {
    error = "LFM schema-v1 manifest contains invalid model parameters";
    return false;
  }
  return true;
}

}  // namespace qhx
