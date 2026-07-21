# API Reference (C-ABI)

The C API is defined in `include/qhexrt/qhexrt_c.h`.

## 1. Model capabilities

### `qhx_model_capabilities`
```c
qhx_status qhx_model_capabilities(const qhx_model_capability** capabilities,
                                  size_t* count);
```
Returns the immutable capability registry owned by the QHexRT process. The
array, records, and strings remain valid for the lifetime of the process.

Each size-versioned record uses typed fields for model identity, family,
operations, Hexagon architecture mask, support state, runner kind, and graph
contract. `QHX_MODEL_SUPPORT_DECLARED` does not permit loading;
`QHX_MODEL_SUPPORT_EXECUTABLE` identifies a model with an implemented runner
and verified graph-contract parser in this build.

ABI 1.3 defines two typed LFM graph contracts:

- `QHX_GRAPH_CONTRACT_LFM2_SPLIT_V1`: the executable LFM2.5-230M path with
  separate prefill, decode, and LM-head contexts.
- `QHX_GRAPH_CONTRACT_LFM2_SHARED_V1`: the V79 LFM2.5-350M/2048 adapter with a
  shared prefill/decode context, a 512-token prefill window, and a 2048-token
  decode state. The contract is implemented, but its model capability remains
  `DECLARED` until real-device validation succeeds.

### `qhx_model_capability_find`
```c
const qhx_model_capability* qhx_model_capability_find(qhx_model_id model_id);
```
Returns a runtime-owned record for a typed model ID, or `NULL` when the ID is
not declared by this build.

## 2. Lifecycle Management

### `qhx_runtime_create`
```c
qhx_runtime* qhx_runtime_create(const char* htp_library, const char* system_library);
```
Initializes the QNN environment. Requires absolute paths to `libQnnHtp.so` and `libQnnSystem.so`.

### `qhx_model_load`
```c
qhx_model* qhx_model_load(qhx_runtime* runtime, const char* manifest_path, const char* artifacts_dir);
```
Loads model graphs and weights into memory. `manifest_path` points to the JSON configuration.

### `qhx_session_create`
```c
qhx_session* qhx_session_create(qhx_model* model);
```
Creates an execution session. A session maintains the KV cache and conversation state.

## 3. Inference

### `qhx_generate`
```c
qhx_status qhx_generate(qhx_session* session, const qhx_inputs* inputs,
                        const qhx_gen_cfg* config, qhx_token_cb callback,
                        void* callback_user, qhx_output* output);
```
Performs inference.
- `inputs`: Prompt text, system prompt, history, etc.
- `config`: Generation parameters (temperature, top_p, etc.).
- `callback`: Invoked for every generated token (for streaming).
- `output`: Final statistics (tokens generated, latency).

## 4. Configuration Structures

### `qhx_gen_cfg`
Controls the sampling behavior:
- `temperature`: Randomness (0.0 = greedy).
- `top_p` / `top_k`: Nucleus and top-k sampling.
- `max_new_tokens`: Limit for generation.
- `repetition_penalty`: Penalize repeating tokens.

### `qhx_inputs`
- `text`: The user prompt.
- `system_prompt`: Optional system context.
- `history`: Optional conversation history array.
- `no_template`: If 1, bypasses internal chat templating.

## 5. Error Handling
Most functions return `qhx_status`. Use `qhx_status_str(status)` to get a human-readable error message.
- `QHX_OK`: Success.
- `QHX_ERROR_QNN_LOAD`: Failed to load `.so` files.
- `QHX_ERROR_MODEL`: Invalid model manifest or artifacts.
