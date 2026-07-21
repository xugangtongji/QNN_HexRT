# QHexRT

[English Version](./README.md)

QHexRT 是一个高性能推理运行时，专为在 Qualcomm Hexagon DSP 上运行大基座模型（LFM）而设计。它利用高通 AI 栈（QNN）SDK，在 Android 移动设备上提供低延迟、高能效的 AI 能力。

### 核心特性

*   **硬件加速**：专门针对高通 Hexagon HTP（Hexagon 张量处理器）进行优化。
*   **简洁的 C API**：通过 JNI 或原生 C++ 轻松集成到 Android 应用中。
*   **流式输出**：支持通过回调函数进行实时的 Token 生成。
*   **高度可配置**：支持调节 Temperature、Top-p、Top-k、重复惩罚等生成参数。
*   **Android 原生**：专为 `arm64-v8a` Android 环境打造。

### 模型能力

QHexRT ABI 1.3 提供由运行库持有的结构化模型能力注册表。`DECLARED`
表示已经建立模型适配边界；只有 `EXECUTABLE` 才表示当前构建已经包含可用的
图契约和 Runner。

| Catalog ID | Runner | Hexagon 架构 | 状态 |
|---|---|---|---|
| `qwen3_5_0_8b` | Qwen3.5 | v75、v79、v81 | Declared |
| `lfm2_5_350m` | LFM2 | v79 | Declared |
| `qwen3_0_6b` | Qwen3 | v75、v81 | Declared |
| `lfm2_5_230m` | LFM2 | v75、v79、v81 | Executable |

使用 `qhx_model_capabilities()` 枚举能力，通过
`qhx_model_capability_find()` 使用类型化模型 ID 查询。只有状态为
`QHX_MODEL_SUPPORT_EXECUTABLE` 且架构掩码包含当前设备架构的模型才能加载。

ABI 1.3 已实现 V79 `lfm2-5-350m-2048` 的共享 Context 加载与 Runner
契约。由于当前没有 V79 真机完成 `qhx_generate` 验证，其能力状态仍保持
`DECLARED`；真机生成成功后再提升为 `EXECUTABLE`。
验证 APK 可以显式设置 `QHX_ENABLE_LFM2_5_350M_CANDIDATE=ON`；该选项
默认关闭，不改变普通构建的 `DECLARED` 状态。

### 项目架构

QHexRT 包含以下部分：
- `qhexrt_core`：核心推理逻辑 and C-API 实现。
- `model_capabilities`：由核心运行库统一维护模型 ID、Runner、图契约和架构策略。
- `qhexrt_host`：宿主侧运行时管理，包括 Tokenizer 和 QNN 环境配置。
- `qhx_generate`：用于设备端测试的命令行工具。

### 快速上手

#### 环境准备

*   **Qualcomm AI Stack (QNN) SDK**：建议版本为 `2.48.0.260626`。
*   **Android NDK**：用于为 Android 进行交叉编译。
*   **CMake**：3.22 或更高版本。

#### 编译构建

1. 将环境变量 `QNN_SDK_ROOT` 设置为您的 QNN SDK 路径。
2. 使用 CMake 配置并构建：

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31
make
```

#### 运行命令行工具

将编译生成的二进制文件和相关 QNN 库推送到 Android 设备：

```bash
adb push bin/arm64-v8a/qhx_generate /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/
# 推送您的模型清单和工件文件...

adb shell
cd /data/local/tmp/
./qhx_generate model_manifest.json libQnnHtp.so libQnnSystem.so "你好，请问你是谁？" 128
```
