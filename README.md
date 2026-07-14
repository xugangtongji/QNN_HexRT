# QHexRT

[中文版本](./README_zh.md)

QHexRT is a high-performance inference runtime designed for running Large Foundation Models (LFM) on Qualcomm Hexagon DSPs. It leverages the Qualcomm AI Stack (QNN) SDK to provide low-latency, energy-efficient AI capabilities on Android mobile devices.

### Key Features

*   **Hardware Accelerated**: Optimized specifically for Qualcomm Hexagon HTP (Hexagon Tensor Processor).
*   **Simple C API**: Easy to integrate into Android applications via JNI or native C++.
*   **Streaming Support**: Real-time token generation with callback support.
*   **Highly Configurable**: Control over temperature, top-p, top-k, repetition penalties, and more.
*   **Android Native**: Built for `arm64-v8a` Android environments.

### Architecture

QHexRT consists of:
- `qhexrt_core`: The core inference logic and C-API implementation.
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
