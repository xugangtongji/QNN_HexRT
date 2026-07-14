# QHexRT Android build

This source tree implements QHexRT ABI 1.1 for Android arm64. The current MVP
loads the three serialized LFM2-230M QNN contexts and retrieves their real HTP
graphs. Generation remains fail-closed until tensor binding and the recurrent
host loop are complete; it never returns synthetic output.

```bash
export QNN_SDK_ROOT=/home/xugang/qairt-SDK/2.48.0.260626
export ANDROID_NDK_HOME=/home/xugang/Android/Sdk/ndk/27.3.13750724

cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

The build also produces `build/qhx_generate`, a device-only real inference
smoke CLI. Push it together with the selected model bundle, `libQnnHtp.so`,
`libQnnSystem.so`, the matching HTP stub/skel, and `libc++_shared.so`, then run:

```bash
export ADSP_LIBRARY_PATH=/data/local/tmp/qhexrt/lib
export LD_LIBRARY_PATH=/data/local/tmp/qhexrt/lib
./qhx_generate model/lfm2-5-230m.json \
  lib/libQnnHtp.so lib/libQnnSystem.so 'The capital of France is' 16
```

Optional stage profiling is disabled by default. Enable it for a smoke run with
`QHEXRT_PROFILE=1`, or on a debuggable Android device with
`setprop debug.qhexrt.profile 1`. Successful generations then emit one
`QHexRTPerf` log line covering preparation, prefill graph, decode graph,
lmhead, sampling, callbacks, and decode execution count. The profiler does not
change the public C ABI.

Stage the archives into the public SDK's immutable prebuilt store:

```bash
tools/scripts/stage_prebuilt_for_sdk.sh \
  --sdk-root /home/xugang/runanywhere-sdks
```

The first model bundle is
`/media/xugang/disk_180G/lfm2_5_230m_HNPU`. Its `v79` or `v81` manifest is
selected by the public RunAnywhere adapter after `qhx_runtime_device()` reports
the device architecture.
