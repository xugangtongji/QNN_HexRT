# Optimization Methods

QHexRT employs several optimization techniques to achieve low-latency and high-efficiency inference on mobile devices.

## 1. Hardware Acceleration (HTP)
- **Hexagon Tensor Processor**: Offloads heavy matrix multiplications (GEMM) and convolutions to the dedicated HTP on Snapdragon SoCs.
- **Quantization**: Supports 16-bit (FP16) and 8-bit (INT8) activations/weights via QNN, significantly reducing memory bandwidth and power consumption.

## 2. Graph Splitting (Prefill vs. Decode)
- **Prefill Graph**: Optimized for high throughput to process the entire input prompt at once.
- **Decode Graph**: Optimized for low latency to generate tokens one by one, minimizing the overhead of the recurrent loop.
- **Shared KV Cache**: Efficiently transfers Key-Value tensors between the prefill and decode stages to avoid redundant computations.

## 3. Memory Optimizations
- **Zero-Copy Tensor Binding**: Tensors are bound directly to QNN memory buffers to avoid unnecessary data copying between the CPU host and the DSP.
- **Memory Mapping (mmap)**: Large model weights are mapped into the process's address space as read-only, allowing the OS to manage memory demand and sharing.

## 4. Host-Side Efficiency
- **C++20 Implementation**: Utilizes modern C++ features for efficient resource management (RAII) and performance.
- **Optimized Sampling**: The logit sampling (Temperature, Top-P, Top-K) is implemented in optimized C++ on the CPU, running in parallel with or immediately after the DSP inference.

## 5. Streaming and Concurrency
- **Streaming Callbacks**: Reduces perceived latency by delivering tokens to the UI as soon as they are generated.
- **Asynchronous Execution**: While the DSP is busy with inference, the CPU can prepare the next set of inputs or handle tokenization.
