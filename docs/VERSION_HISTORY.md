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
- **Minor**: 1

---

## Roadmap (Planned)
- **v0.2.0**:
    - Support for multi-modal inputs (Images/Audio).
    - Enhanced grammar-constrained generation.
    - Performance profiling and instrumentation improvements.
- **v0.3.0**:
    - INT4/INT8 Weight Quantization optimizations.
    - Multi-session support for concurrent requests.
