cmake_minimum_required(VERSION 3.18)
project(saguaro_core VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Edition Configuration (SAGUARO default: ENTERPRISE for no scale limits)
# =============================================================================
# HN_EDITION: 2 = ENTERPRISE (No caps)
add_compile_definitions(HN_EDITION=2)
message(STATUS "SAGUARO Build: Enforcing HN_EDITION=2 (Enterprise/No Caps)")

# =============================================================================
# Find TensorFlow
# =============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Get TensorFlow paths - filter cv2 pollution by extracting last line only
execute_process(
    COMMAND ${Python3_EXECUTABLE} -I -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE RAW_TF_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
# Extract last line only (filters any stdout pollution from cv2/other packages)
string(REGEX REPLACE ".*\n" "" TF_INCLUDE_DIR "${RAW_TF_INCLUDE_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -I -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE RAW_TF_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX REPLACE ".*\n" "" TF_CFLAGS "${RAW_TF_CFLAGS}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -I -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE RAW_TF_LFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX REPLACE ".*\n" "" TF_LFLAGS "${RAW_TF_LFLAGS}")

message(STATUS "TF Include: ${TF_INCLUDE_DIR}")

# =============================================================================
# Sources
# =============================================================================
set(SAGUARO_SOURCES
    src/ops/fused_qwt_tokenizer_op.cc
    src/ops/fused_text_tokenizer_op.cc
    src/ops/quantum_embedding_op.cc
    src/ops/fused_coconut_bfs_op.cc
    src/ops/time_crystal_step_op.cc
    src/ops/hd_timecrystal_op.cc
    src/ops/fused_fft_projector_op.cc
    src/ops/circular_conv_op.cc
    src/ops/hyperdimensional_embedding_op.cc
    src/ops/common/tensor_stream_pool.cc
    src/ops/saguaro_memory_ops.cc
    # Native C API for Python ctypes (NO TensorFlow dependency in callers)
    src/ops/native_indexer_api.cc
)

# =============================================================================
# Build Library
# =============================================================================
add_library(saguaro_core SHARED ${SAGUARO_SOURCES})

target_include_directories(saguaro_core PRIVATE
    ${TF_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro  # For "ops/foo.h" resolution
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro/ops/common
)

# Explicitly set ABI flag to match TF 2.x defaults on Linux
target_compile_definitions(saguaro_core PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# TensorFlow flags
# Separate TF compile flags
separate_arguments(TF_CFLAGS_LIST UNIX_COMMAND "${TF_CFLAGS}")
target_compile_options(saguaro_core PRIVATE 
    -DEIGEN_USE_THREADS 
    ${TF_CFLAGS_LIST} 
    -Wall 
    -O3 
    -march=native 
    -fPIC
    -fno-rtti
)

# Separate TF link flags
separate_arguments(TF_LFLAGS_LIST UNIX_COMMAND "${TF_LFLAGS}")
target_link_libraries(saguaro_core PRIVATE ${TF_LFLAGS_LIST})

# Allow undefined symbols (resolved by Python process at runtime)
if(UNIX AND NOT APPLE)
    target_link_options(saguaro_core PRIVATE "-Wl,--allow-shlib-undefined")
    target_link_options(saguaro_core PRIVATE "-Wl,--unresolved-symbols=ignore-all")
elseif(APPLE)
    target_link_options(saguaro_core PRIVATE "-undefined" "dynamic_lookup")
endif()

target_link_options(saguaro_core PRIVATE -Wl,--no-as-needed)

# RPATH Logic (Disabled as we rely on process symbols)
# string(REGEX MATCH "-L([^ ]+)" TF_LIB_DIR_MATCH "${TF_LFLAGS}")
# ...

set_target_properties(saguaro_core PROPERTIES
    OUTPUT_NAME "_saguaro_core"
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
)

# =============================================================================
# Build Native Library (NO TensorFlow Dependency)
# =============================================================================
add_library(saguaro_native SHARED 
    src/ops/native_indexer_api.cc
)

target_include_directories(saguaro_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/include/saguaro/ops/common
)

# Native compiler flags (AVX2/SIMD support)
target_compile_options(saguaro_native PRIVATE 
    -Wall 
    -O3 
    -march=native 
    -fPIC
    -fno-rtti
)

# No TensorFlow linking here!
if(UNIX AND NOT APPLE)
    target_link_options(saguaro_native PRIVATE "-Wl,--no-undefined")
elseif(APPLE)
    target_link_options(saguaro_native PRIVATE "-undefined" "error")
endif()

set_target_properties(saguaro_native PROPERTIES
    OUTPUT_NAME "_saguaro_native"
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
)


