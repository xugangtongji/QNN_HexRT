#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qhx {

class Tokenizer {
 public:
  bool load(const std::string& path, std::string& error);
  bool encode(const std::string& text, std::vector<int32_t>& ids, std::string& error) const;
  std::string decode(int32_t id) const;

 private:
  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int32_t> merge_rank_;
  std::unordered_map<std::string, int32_t> special_;
  std::vector<std::string> byte_encoder_;
  std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};

}  // namespace qhx
