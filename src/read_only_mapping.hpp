#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace qhx {

enum class MappingAccess {
  kNormal,
  kSequential,
  kRandom,
};

class ReadOnlyMapping {
 public:
  ReadOnlyMapping() = default;
  ~ReadOnlyMapping();

  ReadOnlyMapping(const ReadOnlyMapping&) = delete;
  ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
  ReadOnlyMapping(ReadOnlyMapping&& other) noexcept;
  ReadOnlyMapping& operator=(ReadOnlyMapping&& other) noexcept;

  bool open(const std::string& path, MappingAccess access, std::string& error);
  void reset() noexcept;

  [[nodiscard]] const uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] std::span<const uint8_t> bytes() const noexcept { return {data_, size_}; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace qhx
