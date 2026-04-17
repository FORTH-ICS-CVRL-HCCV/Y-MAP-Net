cmake_minimum_required(VERSION 3.18)
project(ymapnet_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# CUDA (optional – auto-detected, falls back to CPU-only)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    # Set arch BEFORE enable_language(CUDA) and BEFORE FetchContent so ggml picks it up.
    # sm_86 (Ampere) instead of sm_89 (Ada): ggml's convert.cu incorrectly emits FP8
    # instructions for sm_89 which require sm_90+ in CUDA 12.2's ptxas.  sm_86 kernels
    # run unmodified on Ada Lovelace via CUDA backward compatibility.
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures" FORCE)
    enable_language(CUDA)
    set(GGML_CUDA       ON  CACHE BOOL "Enable ggml CUDA backend" FORCE)
    set(GGML_CUDA_F16   ON  CACHE BOOL "Enable F16 CUDA ops"      FORCE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found – building with GPU backend (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    message(STATUS "CUDA not found – CPU-only build")
endif()

# ---------------------------------------------------------------------------
# ggml (standalone – contains gguf support)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)

# Disable everything we don't need to keep build times short
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ggml)

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# ymapnet executable
# ---------------------------------------------------------------------------
add_executable(ymapnet
    src/main.cpp
    src/model.cpp
)

target_include_directories(ymapnet PRIVATE
    src
    # Handle both flat (include/ggml.h) and namespaced (include/ggml/ggml.h) layouts
    ${ggml_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ymapnet PRIVATE
    ggml
    ${OpenCV_LIBS}
)

if(CUDAToolkit_FOUND)
    target_compile_definitions(ymapnet PRIVATE GGML_USE_CUDA)
    target_link_libraries(ymapnet PRIVATE CUDA::cudart)
endif()

target_compile_options(ymapnet PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)

message(STATUS "Targets: ymapnet")
message(STATUS "  ggml CUDA backend : ${GGML_CUDA}")
message(STATUS "  OpenCV libs       : ${OpenCV_LIBS}")
