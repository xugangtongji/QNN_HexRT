#include "read_only_mapping.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qhx {
namespace {

std::string system_error(const char* operation, const std::string& path) {
  return std::string(operation) + "(" + path + "): " + std::strerror(errno);
}

}  // namespace

ReadOnlyMapping::~ReadOnlyMapping() { reset(); }

ReadOnlyMapping::ReadOnlyMapping(ReadOnlyMapping&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

ReadOnlyMapping& ReadOnlyMapping::operator=(ReadOnlyMapping&& other) noexcept {
  if (this == &other) return *this;
  reset();
  data_ = other.data_;
  size_ = other.size_;
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

bool ReadOnlyMapping::open(const std::string& path, MappingAccess access, std::string& error) {
  reset();
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    error = system_error("open", path);
    return false;
  }

  struct stat info {};
  if (fstat(descriptor, &info) != 0) {
    error = system_error("fstat", path);
    close(descriptor);
    return false;
  }
  if (info.st_size <= 0 || static_cast<uintmax_t>(info.st_size) >
                               static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
    error = "Invalid mapped file size: " + path;
    close(descriptor);
    return false;
  }

  const size_t mapped_size = static_cast<size_t>(info.st_size);
  void* mapped = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
  const int mapping_error = errno;
  close(descriptor);
  if (mapped == MAP_FAILED) {
    errno = mapping_error;
    error = system_error("mmap", path);
    return false;
  }

  data_ = static_cast<const uint8_t*>(mapped);
  size_ = mapped_size;
  if (access != MappingAccess::kNormal) {
    const int advice = access == MappingAccess::kRandom ? MADV_RANDOM : MADV_SEQUENTIAL;
    (void)madvise(const_cast<uint8_t*>(data_), size_, advice);
  }
  return true;
}

void ReadOnlyMapping::reset() noexcept {
  if (data_) munmap(const_cast<uint8_t*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

}  // namespace qhx
