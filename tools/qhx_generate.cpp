#include <qhexrt/qhexrt_c.h>

#include <cstdlib>
#include <iostream>

namespace {
int emit(void*, const char* text, int length, int, int final) {
  if (!final && text && length > 0) {
    std::cout.write(text, length);
    std::cout.flush();
  }
  return 1;
}
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: qhx_generate MANIFEST libQnnHtp.so libQnnSystem.so PROMPT [MAX_TOKENS]\n";
    return 2;
  }
  qhx_runtime* runtime = qhx_runtime_create(argv[2], argv[3]);
  if (!runtime) { std::cerr << "runtime: " << qhx_status_str(QHX_ERROR_QNN_LOAD) << '\n'; return 1; }
  char arch[16]{}; int soc = 0;
  qhx_runtime_device(runtime, arch, sizeof(arch), &soc, nullptr);
  std::cerr << "QHexRT " << qhx_version() << " arch=" << arch << " soc=" << soc << '\n';
  qhx_model* model = qhx_model_load(runtime, argv[1], nullptr);
  if (!model) { std::cerr << "model: " << qhx_status_str(QHX_ERROR_MODEL) << '\n'; qhx_runtime_free(runtime); return 1; }
  qhx_session* session = qhx_session_create(model);
  qhx_inputs inputs{}; inputs.text = argv[4]; inputs.no_template = 1;
  qhx_gen_cfg config; qhx_gen_cfg_default(&config);
  config.temperature = 0.0F;
  if (argc == 6) config.max_new_tokens = std::atoi(argv[5]);
  qhx_output output{};
  const qhx_status status = qhx_generate(session, &inputs, &config, emit, nullptr, &output);
  std::cout << '\n';
  if (status != QHX_OK) std::cerr << "generate: " << qhx_status_str(status) << '\n';
  else std::cerr << "prompt=" << output.n_prompt << " generated=" << output.n_generated
                 << " prefill_ms=" << output.prefill_ms << " decode_ms=" << output.decode_ms << '\n';
  qhx_session_free(session); qhx_model_free(model); qhx_runtime_free(runtime);
  return status == QHX_OK ? 0 : 1;
}
