#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

namespace qhx {
namespace {

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) out.push_back(static_cast<char>(cp));
  else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

bool json_string(const std::string& json, size_t& pos, std::string& out) {
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (pos >= json.size() || json[pos++] != '"') return false;
  out.clear();
  while (pos < json.size()) {
    char c = json[pos++];
    if (c == '"') return true;
    if (c != '\\') { out.push_back(c); continue; }
    if (pos >= json.size()) return false;
    c = json[pos++];
    if (c == 'n') out.push_back('\n');
    else if (c == 'r') out.push_back('\r');
    else if (c == 't') out.push_back('\t');
    else if (c == 'b') out.push_back('\b');
    else if (c == 'f') out.push_back('\f');
    else if (c == 'u') {
      if (pos + 4 > json.size()) return false;
      uint32_t cp = 0;
      for (int i = 0; i < 4; ++i) {
        char h = json[pos++];
        cp = cp * 16 + (h >= '0' && h <= '9' ? h - '0' :
                        h >= 'a' && h <= 'f' ? h - 'a' + 10 : h - 'A' + 10);
      }
      append_utf8(out, cp);
    } else out.push_back(c);
  }
  return false;
}

uint32_t next_cp(const std::string& s, size_t& pos) {
  uint8_t c = static_cast<uint8_t>(s[pos++]);
  if (c < 0x80) return c;
  int n = (c & 0xe0) == 0xc0 ? 1 : 2;
  uint32_t cp = c & (n == 1 ? 0x1f : 0x0f);
  while (n-- && pos < s.size()) cp = (cp << 6) | (static_cast<uint8_t>(s[pos++]) & 0x3f);
  return cp;
}

bool ascii_letter(uint8_t c) { return std::isalpha(c) != 0; }
bool ascii_digit(uint8_t c) { return std::isdigit(c) != 0; }

std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> pieces;
  size_t i = 0;
  while (i < text.size()) {
    size_t start = i;
    const bool prefixed = text[i] == ' ' && i + 1 < text.size() &&
                          !std::isspace(static_cast<unsigned char>(text[i + 1]));
    if (text[i] == '\r' || text[i] == '\n') {
      while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
    } else if (!prefixed && std::isspace(static_cast<unsigned char>(text[i]))) {
      while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) &&
             text[i] != '\r' && text[i] != '\n') ++i;
      if (i < text.size() && i - start > 1) --i;
    } else if (ascii_digit(static_cast<uint8_t>(text[i]))) {
      while (i < text.size() && ascii_digit(static_cast<uint8_t>(text[i])) && i - start < 3) ++i;
    } else {
      if (prefixed) ++i;
      bool letter = i < text.size() && (ascii_letter(static_cast<uint8_t>(text[i])) ||
                                       static_cast<uint8_t>(text[i]) >= 0x80);
      if (letter) {
        while (i < text.size() && (ascii_letter(static_cast<uint8_t>(text[i])) ||
                                   static_cast<uint8_t>(text[i]) >= 0x80)) ++i;
      } else {
        if (i == start) ++i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i])) &&
               !ascii_letter(static_cast<uint8_t>(text[i])) &&
               !ascii_digit(static_cast<uint8_t>(text[i])) &&
               static_cast<uint8_t>(text[i]) < 0x80) ++i;
      }
    }
    pieces.push_back(text.substr(start, i - start));
  }
  return pieces;
}

}  // namespace

bool Tokenizer::load(const std::string& path, std::string& error) {
  std::ifstream stream(path, std::ios::binary);
  std::string json((std::istreambuf_iterator<char>(stream)), {});
  if (json.empty()) { error = "Cannot read tokenizer.json"; return false; }
  byte_encoder_.resize(256);
  std::vector<int> bs, cs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);
  for (int i = 161; i <= 172; ++i) bs.push_back(i);
  for (int i = 174; i <= 255; ++i) bs.push_back(i);
  cs = bs;
  int extra = 0;
  for (int b = 0; b < 256; ++b) if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
    bs.push_back(b); cs.push_back(256 + extra++);
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    append_utf8(byte_encoder_[bs[i]], cs[i]); byte_decoder_[cs[i]] = static_cast<uint8_t>(bs[i]);
  }

  size_t pos = json.find("\"vocab\""); pos = json.find('{', pos) + 1;
  while (pos && pos < json.size() && json[pos] != '}') {
    std::string token;
    if (!json_string(json, pos, token)) { error = "Invalid tokenizer vocab"; return false; }
    pos = json.find(':', pos) + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    size_t end = pos;
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) ++end;
    int32_t id = std::stoi(json.substr(pos, end - pos));
    vocab_[token] = id;
    if (tokens_.size() <= static_cast<size_t>(id)) tokens_.resize(id + 1);
    tokens_[id] = token;
    pos = json.find_first_of(",}", end);
    if (json[pos] == ',') ++pos;
  }
  pos = json.find("\"added_tokens\""); pos = json.find('[', pos) + 1;
  const size_t added_end = json.find("],", pos);
  while (pos < added_end) {
    const size_t object_start = json.find('{', pos);
    const size_t object_end = json.find('}', object_start);
    if (object_start >= added_end || object_end >= added_end) break;
    size_t content = json.find("\"content\"", object_start);
    if (content >= object_end) break;
    content = json.find(':', content) + 1; std::string token;
    if (!json_string(json, content, token)) break;
    size_t id_pos = json.find("\"id\"", object_start);
    if (id_pos >= object_end) break;
    id_pos = json.find(':', id_pos) + 1;
    while (id_pos < json.size() && std::isspace(static_cast<unsigned char>(json[id_pos]))) ++id_pos;
    int32_t id = std::stoi(json.substr(id_pos)); special_[token] = id; pos = object_end + 1;
  }
  pos = json.find("\"merges\""); pos = json.find('[', pos) + 1;
  int rank = 0;
  while (pos && pos < json.size()) {
    pos = json.find_first_of("[]", pos);
    if (pos == std::string::npos || json[pos] == ']') break;
    ++pos; std::string a, b;
    if (!json_string(json, pos, a)) break;
    pos = json.find(',', pos) + 1;
    if (!json_string(json, pos, b)) break;
    merge_rank_[a + '\0' + b] = rank++;
    pos = json.find(']', pos) + 1;
  }
  return !vocab_.empty() && !merge_rank_.empty();
}

bool Tokenizer::encode(const std::string& text, std::vector<int32_t>& ids, std::string& error) const {
  size_t cursor = 0;
  while (cursor < text.size()) {
    size_t special_at = std::string::npos; std::string special_token;
    for (const auto& item : special_) {
      size_t at = text.find(item.first, cursor);
      if (at < special_at) { special_at = at; special_token = item.first; }
    }
    const size_t plain_end = special_at == std::string::npos ? text.size() : special_at;
    for (const auto& piece : pretokenize(text.substr(cursor, plain_end - cursor))) {
      std::vector<std::string> symbols;
      for (uint8_t byte : piece) symbols.push_back(byte_encoder_[byte]);
      while (symbols.size() > 1) {
        int best = std::numeric_limits<int>::max(); size_t best_i = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
          auto it = merge_rank_.find(symbols[i] + '\0' + symbols[i + 1]);
          if (it != merge_rank_.end() && it->second < best) { best = it->second; best_i = i; }
        }
        if (best == std::numeric_limits<int>::max()) break;
        symbols[best_i] += symbols[best_i + 1]; symbols.erase(symbols.begin() + best_i + 1);
      }
      for (const auto& symbol : symbols) {
        auto it = vocab_.find(symbol);
        if (it == vocab_.end()) { error = "Tokenizer produced unknown BPE symbol"; return false; }
        ids.push_back(it->second);
      }
    }
    if (special_at == std::string::npos) break;
    ids.push_back(special_.at(special_token)); cursor = special_at + special_token.size();
  }
  return true;
}

std::string Tokenizer::decode(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return {};
  const std::string& token = tokens_[id];
  if (special_.find(token) != special_.end()) return token;
  std::string out; size_t pos = 0;
  while (pos < token.size()) {
    uint32_t cp = next_cp(token, pos); auto it = byte_decoder_.find(cp);
    if (it != byte_decoder_.end()) out.push_back(static_cast<char>(it->second));
  }
  return out;
}

}  // namespace qhx
