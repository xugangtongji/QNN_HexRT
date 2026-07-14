#pragma once

#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qhx {

struct DynamicLibrary {
  void* handle = nullptr;
  ~DynamicLibrary();
  bool open(const char* path, std::string& error);
  void* symbol(const char* name, std::string& error) const;
};

struct Runtime {
  DynamicLibrary htp_library;
  DynamicLibrary system_library;
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  std::string arch;
  int soc_model = 0;
  std::string error;

  ~Runtime();
  bool initialize(const char* htp_path, const char* system_path);
};

struct ContextGraph {
  struct TensorDef {
    uint32_t id = 0;
    std::string name;
    Qnn_TensorType_t type = QNN_TENSOR_TYPE_UNDEFINED;
    Qnn_DataType_t data_type = QNN_DATATYPE_UNDEFINED;
    std::vector<uint32_t> dimensions;
    size_t bytes = 0;
  };
  struct Bindings {
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
  };
  Runtime* runtime = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::string graph_name;
  std::vector<TensorDef> inputs;
  std::vector<TensorDef> outputs;
  ~ContextGraph();
  bool load(Runtime& runtime, const std::string& path, const std::string& expected_graph,
            std::string& error);
  [[nodiscard]] Bindings create_bindings() const;
  bool execute(Bindings& bindings, std::vector<std::vector<uint8_t>>& input_buffers,
               std::vector<std::vector<uint8_t>>& output_buffers, std::string& error);
};

}  // namespace qhx
