#ifndef QHEXRT_QHEXRT_C_H_
#define QHEXRT_QHEXRT_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QHX_ABI_VERSION_MAJOR 1
#define QHX_ABI_VERSION_MINOR 3

typedef struct qhx_runtime qhx_runtime;
typedef struct qhx_model qhx_model;
typedef struct qhx_session qhx_session;

typedef int32_t qhx_status;
enum {
  QHX_OK = 0,
  QHX_ERROR_INVALID_ARGUMENT = 1,
  QHX_ERROR_IO = 2,
  QHX_ERROR_QNN_LOAD = 3,
  QHX_ERROR_QNN_API = 4,
  QHX_ERROR_MODEL = 5,
  QHX_ERROR_UNSUPPORTED = 6,
  QHX_ERROR_CANCELLED = 7,
  QHX_ERROR_OUT_OF_MEMORY = 8,
  QHX_ERROR_INTERNAL = 9
};

typedef int32_t qhx_model_id;
enum {
  QHX_MODEL_UNKNOWN = 0,
  QHX_MODEL_QWEN3_5_0_8B = 1,
  QHX_MODEL_LFM2_5_350M = 2,
  QHX_MODEL_QWEN3_0_6B = 3,
  QHX_MODEL_LFM2_5_230M = 4
};

typedef int32_t qhx_model_family;
enum {
  QHX_MODEL_FAMILY_UNKNOWN = 0,
  QHX_MODEL_FAMILY_LLM = 1
};

typedef uint32_t qhx_model_operations;
enum {
  QHX_MODEL_OPERATION_NONE = 0,
  QHX_MODEL_OPERATION_GENERATE_TEXT = 1U << 0
};

typedef uint32_t qhx_hexagon_architectures;
enum {
  QHX_HEXAGON_ARCH_NONE = 0,
  QHX_HEXAGON_ARCH_V75 = 1U << 0,
  QHX_HEXAGON_ARCH_V79 = 1U << 1,
  QHX_HEXAGON_ARCH_V81 = 1U << 2
};

typedef int32_t qhx_model_support_state;
enum {
  QHX_MODEL_SUPPORT_UNKNOWN = 0,
  QHX_MODEL_SUPPORT_DECLARED = 1,
  QHX_MODEL_SUPPORT_EXECUTABLE = 2
};

typedef int32_t qhx_runner_kind;
enum {
  QHX_RUNNER_UNKNOWN = 0,
  QHX_RUNNER_LFM2 = 1,
  QHX_RUNNER_QWEN3 = 2,
  QHX_RUNNER_QWEN3_5 = 3
};

typedef int32_t qhx_graph_contract;
enum {
  QHX_GRAPH_CONTRACT_UNKNOWN = 0,
  QHX_GRAPH_CONTRACT_LFM2_SPLIT_V1 = 1,
  QHX_GRAPH_CONTRACT_LFM2_SHARED_V1 = 2
};

typedef struct qhx_model_capability {
  uint32_t struct_size;
  qhx_model_id model_id;
  qhx_model_family family;
  qhx_model_operations operations;
  qhx_hexagon_architectures architectures;
  qhx_model_support_state support_state;
  qhx_runner_kind runner;
  qhx_graph_contract graph_contract;
  const char* catalog_id;
  const char* manifest_name;
} qhx_model_capability;

typedef int (*qhx_token_cb)(void* user, const char* utf8, int len, int token_id, int is_final);
typedef int (*qhx_cancel_cb)(void* user);

typedef struct qhx_inputs {
  const char* text;
  const char* system_prompt;
  const char* const* history;
  int n_history;
  const char* image_path;
  const char* mask_path;
  const float* audio;
  int n_audio;
  int audio_sr;
  int no_template;
} qhx_inputs;

typedef struct qhx_gen_cfg {
  int max_new_tokens;
  float temperature;
  float top_p;
  int top_k;
  float repetition_penalty;
  float min_p;
  uint64_t seed;
  const char* const* stop_strings;
  int n_stop_strings;
  int grammar_kind;
  const char* grammar;
} qhx_gen_cfg;

typedef struct qhx_generate_options {
  uint32_t struct_size;
  int disable_thinking;
  qhx_cancel_cb should_cancel;
  void* should_cancel_user;
} qhx_generate_options;

typedef struct qhx_output {
  const char* text;
  int n_prompt;
  int n_generated;
  double prefill_ms;
  double decode_ms;
  const float* embedding;
  int n_embedding;
  const float* audio;
  int n_audio;
  uint32_t sample_rate;
  const uint8_t* image;
  int img_w;
  int img_h;
  int img_c;
} qhx_output;

/*
 * Returns the immutable supported-model registry for this QHexRT build.
 * The array, records, and strings are owned by QHexRT and remain valid for the
 * lifetime of the process. DECLARED entries identify planned model adapters;
 * only EXECUTABLE entries may be passed to qhx_model_load successfully.
 */
qhx_status qhx_model_capabilities(const qhx_model_capability** capabilities,
                                  size_t* count);
const qhx_model_capability* qhx_model_capability_find(qhx_model_id model_id);

qhx_runtime* qhx_runtime_create(const char* htp_library, const char* system_library);
void qhx_runtime_free(qhx_runtime* runtime);
qhx_status qhx_runtime_device(qhx_runtime* runtime, char* arch, size_t arch_size,
                              int* soc_model, int* htp_device);

qhx_model* qhx_model_load(qhx_runtime* runtime, const char* manifest_path,
                          const char* artifacts_dir);
void qhx_model_free(qhx_model* model);

qhx_session* qhx_session_create(qhx_model* model);
void qhx_session_free(qhx_session* session);
void qhx_session_reset(qhx_session* session);
void qhx_session_cancel(qhx_session* session);

void qhx_gen_cfg_default(qhx_gen_cfg* config);
void qhx_generate_options_default(qhx_generate_options* options);
qhx_status qhx_generate(qhx_session* session, const qhx_inputs* inputs,
                        const qhx_gen_cfg* config, qhx_token_cb callback,
                        void* callback_user, qhx_output* output);
qhx_status qhx_generate_ex(qhx_session* session, const qhx_inputs* inputs,
                           const qhx_gen_cfg* config,
                           const qhx_generate_options* options,
                           qhx_token_cb callback, void* callback_user,
                           qhx_output* output);

const char* qhx_status_str(qhx_status status);
const char* qhx_version(void);

#ifdef __cplusplus
}
#endif
#endif
