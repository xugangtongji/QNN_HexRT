#pragma once

#include "qnn_runtime.hpp"
#include "read_only_mapping.hpp"
#include "tokenizer.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace qhx {
struct LfmWorkspace;
}

struct qhx_runtime {
  qhx::Runtime impl;
};

struct qhx_model {
  qhx_runtime* runtime = nullptr;
  std::string manifest_path;
  std::string artifacts_dir;
  std::string name;
  std::string dsp_arch;
  int hidden = 0;
  int vocab = 0;
  int max_ctx = 0;
  int kv_dim = 0;
  int head_dim = 0;
  int eos_token_id = -1;
  double rope_theta = 0.0;
  std::string tokenizer_path;
  std::string embedding_path;
  qhx::ReadOnlyMapping embedding;
  std::vector<uint8_t> rope_cos;
  std::vector<uint8_t> rope_sin;
  std::vector<uint8_t> prefill_mask;
  std::vector<uint8_t> decode_mask_base;
  std::unique_ptr<qhx::ContextGraph> prefill;
  std::unique_ptr<qhx::ContextGraph> decode;
  std::unique_ptr<qhx::ContextGraph> lmhead;
  std::unique_ptr<qhx::Tokenizer> tokenizer;
};

struct qhx_session {
  qhx_session();
  ~qhx_session();

  qhx_model* model = nullptr;
  std::atomic<bool> cancelled{false};
  std::string output_text;
  std::unique_ptr<qhx::LfmWorkspace> workspace;
};
