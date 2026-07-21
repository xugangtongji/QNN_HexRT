#include "qnn_runtime.hpp"

#include "read_only_mapping.hpp"

#include <android/log.h>
#include <dlfcn.h>
#include <sys/system_properties.h>

#include <HTP/QnnHtpDevice.h>

#include <cstring>
#include <limits>
#include <sstream>
#include <span>

namespace qhx {
namespace {

void qnn_log(const char* format, QnnLog_Level_t, uint64_t, va_list args) {
  __android_log_vprint(ANDROID_LOG_INFO, "QHexRT", format, args);
}

template <typename Provider>
bool compatible(const Provider* provider, uint32_t major, uint32_t minor) {
  const auto& v = provider->apiVersion;
  return v.coreApiVersion.major == major && v.coreApiVersion.minor >= minor;
}

std::string android_arch() {
  char value[PROP_VALUE_MAX]{};
  if (__system_property_get("ro.vendor.qti.soc_model", value) > 0) {
    std::string model(value);
    if (model.find("8735") != std::string::npos || model.find("8750") != std::string::npos)
      return "v81";
  }
  if (__system_property_get("ro.soc.model", value) > 0) {
    std::string model(value);
    if (model.find("SM8750") != std::string::npos) return "v79";
  }
  return "v79";
}

size_t element_size(Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_INT_16: return 2;
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_INT_32: return 4;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_BOOL_8: return 1;
    default: return 0;
  }
}

bool clone_tensor(const Qnn_Tensor_t& source, ContextGraph::TensorDef& result) {
  const Qnn_TensorV1_t* value = nullptr;
  Qnn_TensorV1_t converted = QNN_TENSOR_V1_INIT;
  if (source.version == QNN_TENSOR_VERSION_1) {
    value = &source.v1;
  } else if (source.version == QNN_TENSOR_VERSION_2) {
    converted.id = source.v2.id;
    converted.name = source.v2.name;
    converted.type = source.v2.type;
    converted.dataType = source.v2.dataType;
    converted.rank = source.v2.rank;
    converted.dimensions = source.v2.dimensions;
    value = &converted;
  } else {
    return false;
  }
  if (!value->name || !value->dimensions || value->rank == 0) return false;
  result.id = value->id;
  result.name = value->name;
  result.type = value->type;
  result.data_type = value->dataType;
  result.dimensions.assign(value->dimensions, value->dimensions + value->rank);
  size_t count = 1;
  for (uint32_t dimension : result.dimensions) {
    if (!dimension || count > std::numeric_limits<size_t>::max() / dimension) return false;
    count *= dimension;
  }
  const size_t width = element_size(result.data_type);
  if (!width || count > std::numeric_limits<size_t>::max() / width) return false;
  result.bytes = count * width;
  return true;
}

bool clone_graph_metadata(Runtime& runtime, std::span<const uint8_t> binary,
                          const std::string& expected_name,
                          std::vector<ContextGraph::TensorDef>& inputs,
                          std::vector<ContextGraph::TensorDef>& outputs, std::string& error) {
  QnnSystemContext_Handle_t handle = nullptr;
  if (!runtime.system.systemContextCreate ||
      runtime.system.systemContextCreate(&handle) != QNN_SUCCESS) {
    error = "QNN System context creation failed";
    return false;
  }
  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  const bool metadata_ok = runtime.system.systemContextGetMetaData &&
      runtime.system.systemContextGetMetaData(handle, binary.data(), binary.size(), &binary_info) ==
          QNN_SUCCESS;
  if (!metadata_ok || !binary_info) {
    runtime.system.systemContextFree(handle);
    error = "QNN System context metadata extraction failed";
    return false;
  }
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t graph_count = 0;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphs = binary_info->contextBinaryInfoV3.graphs;
    graph_count = binary_info->contextBinaryInfoV3.numGraphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphs = binary_info->contextBinaryInfoV2.graphs;
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphs = binary_info->contextBinaryInfoV1.graphs;
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
  }
  bool found = false;
  for (uint32_t i = 0; i < graph_count; ++i) {
    const char* name = nullptr;
    Qnn_Tensor_t* graph_inputs = nullptr;
    Qnn_Tensor_t* graph_outputs = nullptr;
    uint32_t input_count = 0, output_count = 0;
    if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      const auto& info = graphs[i].graphInfoV3;
      name = info.graphName; graph_inputs = info.graphInputs; input_count = info.numGraphInputs;
      graph_outputs = info.graphOutputs; output_count = info.numGraphOutputs;
    } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
      const auto& info = graphs[i].graphInfoV2;
      name = info.graphName; graph_inputs = info.graphInputs; input_count = info.numGraphInputs;
      graph_outputs = info.graphOutputs; output_count = info.numGraphOutputs;
    } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      const auto& info = graphs[i].graphInfoV1;
      name = info.graphName; graph_inputs = info.graphInputs; input_count = info.numGraphInputs;
      graph_outputs = info.graphOutputs; output_count = info.numGraphOutputs;
    }
    if (!name || expected_name != name) continue;
    inputs.resize(input_count);
    outputs.resize(output_count);
    found = true;
    for (uint32_t j = 0; j < input_count; ++j) found &= clone_tensor(graph_inputs[j], inputs[j]);
    for (uint32_t j = 0; j < output_count; ++j) found &= clone_tensor(graph_outputs[j], outputs[j]);
    break;
  }
  runtime.system.systemContextFree(handle);
  if (!found) error = "Expected graph metadata not found: " + expected_name;
  return found;
}

Qnn_Tensor_t tensor_descriptor(const ContextGraph::TensorDef& definition) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_1;
  tensor.v1.id = definition.id;
  tensor.v1.name = definition.name.c_str();
  tensor.v1.type = definition.type;
  tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v1.dataType = definition.data_type;
  tensor.v1.rank = static_cast<uint32_t>(definition.dimensions.size());
  tensor.v1.dimensions = const_cast<uint32_t*>(definition.dimensions.data());
  tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
  return tensor;
}

void bind_buffer(Qnn_Tensor_t& tensor, std::vector<uint8_t>& buffer) {
  tensor.v1.clientBuf.data = buffer.data();
  tensor.v1.clientBuf.dataSize = buffer.size();
}

}  // namespace

DynamicLibrary::~DynamicLibrary() {
  if (handle) dlclose(handle);
}

bool DynamicLibrary::open(const char* path, std::string& error) {
  handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    error = std::string("dlopen(") + path + "): " + dlerror();
    return false;
  }
  return true;
}

void* DynamicLibrary::symbol(const char* name, std::string& error) const {
  dlerror();
  void* result = dlsym(handle, name);
  if (const char* detail = dlerror()) error = std::string("dlsym(") + name + "): " + detail;
  return result;
}

Runtime::~Runtime() {
  if (device && qnn.deviceFree) qnn.deviceFree(device);
  if (backend && qnn.backendFree) qnn.backendFree(backend);
  if (log && qnn.logFree) qnn.logFree(log);
}

bool Runtime::initialize(const char* htp_path, const char* system_path) {
  const char* htp = htp_path && *htp_path ? htp_path : "libQnnHtp.so";
  const char* sys = system_path && *system_path ? system_path : "libQnnSystem.so";
  if (!system_library.open(sys, error) || !htp_library.open(htp, error)) return false;

  using GetQnn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  using GetSystem = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
  auto get_qnn = reinterpret_cast<GetQnn>(htp_library.symbol("QnnInterface_getProviders", error));
  auto get_system = reinterpret_cast<GetSystem>(
      system_library.symbol("QnnSystemInterface_getProviders", error));
  if (!get_qnn || !get_system) return false;

  const QnnInterface_t** providers = nullptr;
  uint32_t count = 0;
  if (get_qnn(&providers, &count) != QNN_SUCCESS) {
    error = "QnnInterface_getProviders failed";
    return false;
  }
  bool found = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (compatible(providers[i], QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR)) {
      qnn = providers[i]->QNN_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) {
    error = "No compatible QNN interface provider";
    return false;
  }

  const QnnSystemInterface_t** sys_providers = nullptr;
  count = 0;
  if (get_system(&sys_providers, &count) != QNN_SUCCESS) {
    error = "QnnSystemInterface_getProviders failed";
    return false;
  }
  found = false;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& v = sys_providers[i]->systemApiVersion;
    if (v.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      system = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) {
    error = "No compatible QNN System interface provider";
    return false;
  }

  if (!qnn.logCreate || qnn.logCreate(qnn_log, QNN_LOG_LEVEL_WARN, &log) != QNN_SUCCESS ||
      !qnn.backendCreate || qnn.backendCreate(log, nullptr, &backend) != QNN_SUCCESS) {
    error = "QNN log/backend creation failed";
    return false;
  }
  if (qnn.deviceCreate && qnn.deviceCreate(log, nullptr, &device) != QNN_SUCCESS) {
    error = "QNN HTP device creation failed";
    return false;
  }
  arch = android_arch();
  const QnnDevice_PlatformInfo_t* platform = nullptr;
  if (qnn.deviceGetPlatformInfo && qnn.deviceFreePlatformInfo &&
      qnn.deviceGetPlatformInfo(log, &platform) == QNN_SUCCESS && platform &&
      platform->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    for (uint32_t i = 0; i < platform->v1.numHwDevices; ++i) {
      const auto& hardware = platform->v1.hwDevices[i];
      if (hardware.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1 ||
          !hardware.v1.deviceInfoExtension) continue;
      const auto* extension = reinterpret_cast<const QnnHtpDevice_DeviceInfoExtension_t*>(
          hardware.v1.deviceInfoExtension);
      if (extension->devType != QNN_HTP_DEVICE_TYPE_ON_CHIP) continue;
      soc_model = static_cast<int>(extension->onChipDevice.socModel);
      const int detected_arch = static_cast<int>(extension->onChipDevice.arch);
      if (detected_arch >= 68 && detected_arch <= 99) arch = "v" + std::to_string(detected_arch);
      break;
    }
    qnn.deviceFreePlatformInfo(log, platform);
  }
  return true;
}

ContextBinary::~ContextBinary() {
  if (context && runtime && runtime->qnn.contextFree) {
    runtime->qnn.contextFree(context, nullptr);
  }
}

bool ContextBinary::load(Runtime& rt, const std::string& binary_path,
                         std::string& error) {
  if (context != nullptr) {
    error = "QNN context binary is already loaded";
    return false;
  }
  ReadOnlyMapping binary;
  if (!binary.open(binary_path, MappingAccess::kNormal, error)) return false;
  runtime = &rt;
  if (!rt.qnn.contextCreateFromBinary ||
      rt.qnn.contextCreateFromBinary(rt.backend, rt.device, nullptr, binary.data(), binary.size(),
                                     &context, nullptr) != QNN_SUCCESS) {
    error = "QNN contextCreateFromBinary failed: " + binary_path;
    return false;
  }
  path = binary_path;
  return true;
}

bool ContextGraph::load(Runtime& rt, const std::string& path,
                        const std::string& expected_graph, std::string& error) {
  auto binary = std::make_shared<ContextBinary>();
  if (!binary->load(rt, path, error)) return false;
  return load(binary, expected_graph, error);
}

bool ContextGraph::load(const std::shared_ptr<ContextBinary>& binary,
                        const std::string& expected_graph, std::string& error) {
  if (!binary || !binary->runtime || !binary->context || binary->path.empty()) {
    error = "Invalid QNN context binary";
    return false;
  }
  ReadOnlyMapping mapping;
  if (!mapping.open(binary->path, MappingAccess::kNormal, error)) return false;
  if (!clone_graph_metadata(*binary->runtime, mapping.bytes(), expected_graph,
                            inputs, outputs, error)) {
    return false;
  }
  context_binary = binary;
  runtime = binary->runtime;
  graph_name = expected_graph;
  if (!runtime->qnn.graphRetrieve ||
      runtime->qnn.graphRetrieve(binary->context, graph_name.c_str(), &graph) != QNN_SUCCESS) {
    error = "QNN graphRetrieve failed for: " + graph_name;
    return false;
  }
  return true;
}

ContextGraph::Bindings ContextGraph::create_bindings() const {
  Bindings bindings;
  bindings.inputs.reserve(inputs.size());
  bindings.outputs.reserve(outputs.size());
  for (const auto& input : inputs) bindings.inputs.push_back(tensor_descriptor(input));
  for (const auto& output : outputs) bindings.outputs.push_back(tensor_descriptor(output));
  return bindings;
}

bool ContextGraph::execute(Bindings& bindings,
                           std::vector<std::vector<uint8_t>>& input_buffers,
                           std::vector<std::vector<uint8_t>>& output_buffers,
                           std::string& error) {
  if (!runtime || !graph || input_buffers.size() != inputs.size() ||
      bindings.inputs.size() != inputs.size() || bindings.outputs.size() != outputs.size()) {
    error = "Invalid QNN graph execution request";
    return false;
  }
  output_buffers.resize(outputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (input_buffers[i].size() != inputs[i].bytes) {
      error = "Input tensor size mismatch for " + inputs[i].name;
      return false;
    }
    bind_buffer(bindings.inputs[i], input_buffers[i]);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_buffers[i].resize(outputs[i].bytes);
    bind_buffer(bindings.outputs[i], output_buffers[i]);
  }
  if (!runtime->qnn.graphExecute ||
      runtime->qnn.graphExecute(graph, bindings.inputs.data(), bindings.inputs.size(),
                                bindings.outputs.data(), bindings.outputs.size(), nullptr,
                                nullptr) != QNN_SUCCESS) {
    error = "QNN graphExecute failed for " + graph_name;
    return false;
  }
  return true;
}

}  // namespace qhx
