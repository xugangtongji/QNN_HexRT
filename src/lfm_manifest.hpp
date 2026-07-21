#pragma once

#include <string>

namespace qhx {

enum class LfmContextLayout {
  kSeparate,
  kSharedBody,
};

struct LfmGraphSpec {
  std::string binary_path;
  std::string graph_name;
};

struct LfmManifest {
  int schema_version = 0;
  std::string name;
  std::string family;
  std::string dsp_arch;
  int hidden = 0;
  int vocab = 0;
  int layers = 0;
  int max_context = 0;
  int kv_dimension = 0;
  int head_dimension = 0;
  int eos_token_id = -1;
  int bos_token_id = -1;
  double rope_theta = 0.0;
  LfmContextLayout context_layout = LfmContextLayout::kSeparate;
  LfmGraphSpec prefill;
  LfmGraphSpec decode;
  LfmGraphSpec lmhead;
  std::string embedding_path;
  std::string tokenizer_path;
};

bool parse_lfm_manifest_v1(const std::string& json, LfmManifest& manifest,
                           std::string& error);

}  // namespace qhx
