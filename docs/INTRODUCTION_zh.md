# QHexRT 详细介绍

QHexRT 是专为高通 Hexagon DSP 优化的高性能大基座模型（LFM）推理运行时。本文档详细介绍了其架构、依赖、接口及优化方法。

## 1. 架构说明

QHexRT 采用分层设计，旨在高效衔接高层模型需求与底层硬件能力。详细架构请参考 [详细架构说明](./ARCHITECTURE_zh.md)。

## 2. 依赖说明

### 编译时依赖
*   **QNN SDK**: 建议版本 `2.48.0.260626`。
*   **Android NDK**: `r24` 或更高版本，需支持 C++20。
*   **CMake**: `3.22.1` 或更高版本。

### 运行时依赖
*   **硬件**：搭载 Hexagon HTP 的高通骁龙芯片（如 8 Gen 1/2/3）。
*   **系统库**：`libQnnHtp.so`, `libQnnSystem.so` 以及对应的 HTP Skel 库。
*   **操作系统**：建议 Android 12 (API 31) 或更高版本。

## 3. 接口说明 (C-API)

主要接口定义在 `qhexrt_c.h` 中：
*   `qhx_runtime_create`：初始化环境并加载 QNN 后端库。
*   `qhx_model_load`：加载模型清单（JSON）及图形工件。
*   `qhx_session_create`：创建推理会话，维护 KV 缓存。
*   `qhx_generate`：执行生成任务，支持通过回调函数实现流式输出。

## 4. 优化方法

*   **硬件加速 (HTP)**：将核心矩阵运算下放至 Hexagon HTP 处理器，大幅提升能效比。
*   **图切分 (Graph Splitting)**：将模型分为 Prefill（预填充）和 Decode（解码）两个子图，分别针对吞吐和延迟进行优化。
*   **零拷贝张量绑定**：减少 CPU 与 DSP 之间的数据搬运开销。
*   **内存映射**：通过 `mmap` 加载权重文件，优化系统内存占用。
*   **高效采样**：在 CPU 侧实现优化的 Temperature/Top-P/Top-K 采样逻辑。

## 5. 版本说明 (v0.1.0 MVP)

*   支持通过 QNN HTP 运行大基座模型。
*   提供流式 Token 生成能力。
*   包含基础采样参数配置。
*   提供设备端测试工具 `qhx_generate`。
