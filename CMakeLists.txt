cmake_minimum_required(VERSION 3.22)
project(QHexRT VERSION 0.1.0 LANGUAGES C CXX)

if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "QHexRT MVP supports Android arm64-v8a only")
endif()
set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "QAIRT/QNN SDK root")
if(QNN_SDK_ROOT STREQUAL "")
  message(FATAL_ERROR "Set QNN_SDK_ROOT to QAIRT 2.48.0.260626")
endif()
set(QAIRT_ROOT "${QNN_SDK_ROOT}")
if(NOT EXISTS "${QAIRT_ROOT}/include/QNN/QnnInterface.h")
  message(FATAL_ERROR "Invalid QAIRT SDK root: ${QAIRT_ROOT}")
endif()

add_library(qhexrt_core STATIC
  src/lfm_manifest.cpp
  src/model_capabilities.cpp
  src/qhexrt_c.cpp)
add_library(qhexrt_host STATIC
  src/read_only_mapping.cpp
  src/qnn_runtime.cpp
  src/tokenizer.cpp
  src/lfm_runner.cpp)

option(QHX_ENABLE_LFM2_5_350M_CANDIDATE
  "Enable the device-validation-only LFM2.5-350M execution candidate" OFF)
if(QHX_ENABLE_LFM2_5_350M_CANDIDATE)
  target_compile_definitions(qhexrt_core PRIVATE
    QHX_ENABLE_LFM2_5_350M_CANDIDATE=1)
endif()

foreach(target qhexrt_core qhexrt_host)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror
    -fvisibility=hidden -fvisibility-inlines-hidden)
  target_include_directories(${target} PUBLIC include PRIVATE src "${QAIRT_ROOT}/include/QNN")
endforeach()
target_link_libraries(qhexrt_host PRIVATE dl log)
set_target_properties(qhexrt_core PROPERTIES OUTPUT_NAME qhexrt_core)
set_target_properties(qhexrt_host PROPERTIES OUTPUT_NAME qhexrt_host)

option(QHX_BUILD_SMOKE_CLI "Build the Android device smoke CLI" ON)
if(QHX_BUILD_SMOKE_CLI)
  add_executable(qhx_generate tools/qhx_generate.cpp)
  target_compile_features(qhx_generate PRIVATE cxx_std_20)
  target_include_directories(qhx_generate PRIVATE include)
  target_link_libraries(qhx_generate PRIVATE
    -Wl,--start-group qhexrt_core qhexrt_host -Wl,--end-group dl log)
endif()

install(TARGETS qhexrt_core qhexrt_host ARCHIVE DESTINATION lib/arm64-v8a)
install(DIRECTORY include/ DESTINATION include)
if(QHX_BUILD_SMOKE_CLI)
  install(TARGETS qhx_generate RUNTIME DESTINATION bin/arm64-v8a)
endif()
