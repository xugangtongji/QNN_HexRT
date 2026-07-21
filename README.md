# QHexRT

[中文版本](./README_zh.md)

QHexRT is a high-performance inference runtime designed for running Large Foundation Models (LFM) on Qualcomm Hexagon DSPs. It leverages the Qualcomm AI Stack (QNN) SDK to provide low-latency, energy-efficient AI capabilities on Android mobile devices.

### Key Features

*   **Hardware Accelerated**: Optimized specifically for Qualcomm Hexagon HTP (Hexagon Tensor Processor).
*   **Simple C API**: Easy to integrate into Android applications via JNI or native C++.
*   **Streaming Support**: Real-time token generation with callback support.
*   **Highly Configurable**: Control over temperature, top-p, top-k, repetition penalties, and more.
*   **Android Native**: Built for `arm64-v8a` Android environments.

### Model Capabilities

QHexRT ABI 1.3 exposes a typed, runtime-owned model-capability registry. A
`DECLARED` entry identifies a model adapter boundary; only an `EXECUTABLE`
entry has a graph contract and runner implemented by the current build.

| Catalog ID | Runner | Architectures | State |
|---|---|---|---|
| `qwen3_5_0_8b` | Qwen3.5 | v75, v79, v81 | Declared |
| `lfm2_5_350m` | LFM2 | v79 | Declared |
| `qwen3_0_6b` | Qwen3 | v75, v81 | Declared |
| `lfm2_5_230m` | LFM2 | v75, v79, v81 | Executable |

Use `qhx_model_capabilities()` to enumerate the records and
`qhx_model_capability_find()` for typed lookup. A catalog row must not be
treated as loadable unless its state is `QHX_MODEL_SUPPORT_EXECUTABLE` and its
architecture mask contains the current device architecture.

The V79 `lfm2-5-350m-2048` shared-context loader and runner contract are
implemented in ABI 1.3. Its capability intentionally remains `DECLARED` until
the existing `qhx_generate` CLI produces valid output on a real V79 device.
Validation APK builds may opt in with
`QHX_ENABLE_LFM2_5_350M_CANDIDATE=ON`; the option defaults to `OFF`.

### Architecture

QHexRT consists of:
- `qhexrt_core`: The core inference logic and C-API implementation.
- `model_capabilities`: Typed model identity, runner, graph-contract, and
  architecture policy owned by the core runtime.
- `qhexrt_host`: Host-side runtime management, including tokenizer and QNN environment setup.
- `qhx_generate`: A smoke-test CLI tool for device-side testing.

### Getting Started

#### Prerequisites

*   **Qualcomm AI Stack (QNN) SDK**: Recommended version `2.48.0.260626`.
*   **Android NDK**: To cross-compile for Android.
*   **CMake**: Version 3.22 or higher.

#### Building

1. Set the `QNN_SDK_ROOT` environment variable to your QNN SDK path.
2. Use CMake to configure and build for Android:

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31
make
```

#### Running the CLI

Push the built binaries and QNN libraries to your Android device:

```bash
adb push bin/arm64-v8a/qhx_generate /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/
# Push your model manifest and artifacts...

adb shell
cd /data/local/tmp/
./qhx_generate model_manifest.json libQnnHtp.so libQnnSystem.so "Hello, who are you?" 128
```
