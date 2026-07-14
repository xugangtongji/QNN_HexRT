#include "tokenizer.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  qhx::Tokenizer tokenizer;
  std::string error;
  if (!tokenizer.load(argv[1], error)) { std::cerr << error << '\n'; return 1; }
  std::vector<int32_t> ids;
  if (!tokenizer.encode(argv[2], ids, error)) { std::cerr << error << '\n'; return 1; }
  for (int32_t id : ids) std::cout << id << ':' << tokenizer.decode(id) << '\n';
}
