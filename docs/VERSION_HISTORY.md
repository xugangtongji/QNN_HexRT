# Version History

## v0.1.0 (MVP)
*Initial release focusing on core functionality and Android HTP support.*

### Features
- Support for Large Foundation Models (LFM) via QNN HTP.
- Core C-API for integration.
- Streaming token generation.
- Basic sampling: Temperature, Top-P, Top-K, Repetition Penalty.
- Device-side smoke test CLI (`qhx_generate`).

### Build System
- CMake-based cross-compilation for `arm64-v8a`.
- Integration with QNN SDK 2.48.

### ABI Version
- **Major**: 1
- **Minor**: 3

### ABI 1.2 additions
- Runtime-owned, size-versioned `qhx_model_capability` records.
- Typed model, family, operation, architecture, support-state, runner, and
  graph-contract identities.
- Structured declarations for Qwen3.5-0.8B, LFM2.5-350M, Qwen3-0.6B, and
  LFM2.5-230M; only the verified LFM2.5-230M contract is executable.

### ABI 1.3 additions
- Typed `QHX_GRAPH_CONTRACT_LFM2_SHARED_V1` contract.
- Structured schema-v1 LFM manifest parsing for the canonical V79
  LFM2.5-350M/2048 bundle.
- Shared QNN context ownership for prefill/decode graphs.
- Independent prefill/decode capacities and safe 512-to-2048 KV-state
  initialization.
- The LFM2.5-350M capability is narrowed to V79 and remains `DECLARED` pending
  a real-device generation smoke run.

---

## Roadmap (Planned)
- **v0.2.0**:
    - Support for multi-modal inputs (Images/Audio).
    - Enhanced grammar-constrained generation.
    - Performance profiling and instrumentation improvements.
- **v0.3.0**:
    - INT4/INT8 Weight Quantization optimizations.
    - Multi-session support for concurrent requests.
