# Dependency Specifications

QHexRT has specific requirements for both build-time and runtime environments to ensure optimal performance on Qualcomm hardware.

## 1. Build-Time Dependencies

*   **Qualcomm AI Stack (QNN) SDK**:
    *   **Version**: `2.48.0.260626` (Tested and recommended).
    *   **Required Components**: Includes for QNN API and libraries for linking/runtime.
*   **Android NDK**:
    *   **Version**: `r24` or newer (Tested with `r27`).
    *   **Toolchain**: LLVM/Clang with support for C++20.
*   **CMake**:
    *   **Version**: `3.22.1` or higher.
*   **Operating System**: Linux (Ubuntu 20.04+ recommended) for cross-compilation.

## 2. Runtime Dependencies (Android)

*   **Hardware**: Qualcomm Snapdragon SoC with Hexagon HTP (e.g., 8 Gen 1, 8 Gen 2, 8 Gen 3).
*   **System Libraries**:
    *   `libQnnHtp.so`: HTP Backend implementation.
    *   `libQnnSystem.so`: QNN System interface.
    *   `libQnnHtpV73Skel.so` / `libQnnHtpV75Skel.so` (and matching stubs): Architecture-specific HTP libraries.
    *   `libc++_shared.so`: Android C++ standard library.
*   **OS Version**: Android 12 (API level 31) or higher is recommended for the best HTP driver support.

## 3. Model Artifacts

QHexRT requires models converted to the QNN format, typically consisting of:
- **Manifest**: A `.json` file describing the model metadata and graph locations.
- **Contexts**: Serialized QNN graphs (often `.bin` or `.serialized` files) for:
    - `prefill`: Token processing and KV cache population.
    - `decode`: Single-token recurrent generation.
    - `lm_head`: Logit generation from hidden states.
- **Tokenizer**: A `tokenizer.json` or similar file compatible with the internal BPE/SentencePiece implementation.
