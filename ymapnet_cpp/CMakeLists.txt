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
# Vulkan (optional – for GPUs without CUDA; needs libvulkan-dev + glslc)
# ---------------------------------------------------------------------------
option(YMAPNET_VULKAN "Build the ggml Vulkan backend (select at runtime with --vulkan)" OFF)
set(GGML_VULKAN ${YMAPNET_VULKAN} CACHE BOOL "Enable ggml Vulkan backend" FORCE)

# ---------------------------------------------------------------------------
# ggml (standalone – contains gguf support)
# ---------------------------------------------------------------------------
include(FetchContent)
# Local ggml patches (see patches/), applied in order to a fresh clone via PATCH_COMMAND, or directly to
# an existing checkout given by FETCHCONTENT_SOURCE_DIR_GGML:
#   ggml-vulkan-bias-prelu.patch   fused per-channel bias + PReLU Vulkan shader
#   ggml-vulkan-pipeline-log.patch log each shader name before the driver compiles it (Adreno crash triage)
set(YMAPNET_GGML_PATCHES ggml-vulkan-bias-prelu.patch ggml-vulkan-pipeline-log.patch)
set(YMAPNET_GGML_PATCH_CMD)
foreach(patch ${YMAPNET_GGML_PATCHES})
    if(YMAPNET_GGML_PATCH_CMD)
        list(APPEND YMAPNET_GGML_PATCH_CMD COMMAND)
    endif()
    list(APPEND YMAPNET_GGML_PATCH_CMD ${CMAKE_COMMAND} -DSRC=<SOURCE_DIR>
         -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patch.cmake)
    if(FETCHCONTENT_SOURCE_DIR_GGML)
        execute_process(COMMAND ${CMAKE_COMMAND} -DSRC=${FETCHCONTENT_SOURCE_DIR_GGML}
            -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/${patch}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patch.cmake COMMAND_ERROR_IS_FATAL ANY)
    endif()
endforeach()
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    # Pinned: validated against Keras (max|diff| 0.23 F32/CUDA). Bump deliberately, then re-run validate_cpp.
    GIT_TAG        353b63b439f27ab2cc19dac97ab1681ba6d2d084   # ggml 0.25.3
    GIT_SHALLOW    FALSE
    GIT_PROGRESS   TRUE
    PATCH_COMMAND  ${YMAPNET_GGML_PATCH_CMD}
)

# Disable everything we don't need to keep build times short
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ggml)

# ---------------------------------------------------------------------------
# OpenCV  (the library needs core + imgproc only; the CLI adds videoio + highgui)
# ---------------------------------------------------------------------------
option(YMAPNET_BUILD_CLI "Build the ymapnet command-line / viewer executable" ON)

set(YMAPNET_OPENCV_COMPONENTS core imgproc)
if(YMAPNET_BUILD_CLI)
    list(APPEND YMAPNET_OPENCV_COMPONENTS videoio highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${YMAPNET_OPENCV_COMPONENTS})
message(STATUS "OpenCV ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")

# -march=native only makes sense when building for the machine we compile on
set(YMAPNET_COMPILE_OPTIONS -Wall -Wextra -Wno-unused-parameter -O3)
if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND YMAPNET_COMPILE_OPTIONS -march=native)
endif()

# ---------------------------------------------------------------------------
# ymapnet_core library (no UI): model.hpp + preprocess/skeleton.hpp
# ---------------------------------------------------------------------------
add_library(ymapnet_core STATIC
    src/model.cpp
)

target_include_directories(ymapnet_core PUBLIC
    src
    # Handle both flat (include/ggml.h) and namespaced (include/ggml/ggml.h) layouts
    ${ggml_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ymapnet_core PUBLIC
    ggml
    opencv_core
    opencv_imgproc
)

if(YMAPNET_VULKAN)
    target_compile_definitions(ymapnet_core PRIVATE GGML_USE_VULKAN)
    find_package(Vulkan REQUIRED)   # model.cpp checks the device API version itself
    target_link_libraries(ymapnet_core PRIVATE Vulkan::Vulkan)
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(ymapnet_core PRIVATE GGML_USE_CUDA)
    target_link_libraries(ymapnet_core PRIVATE CUDA::cudart)
endif()

target_compile_options(ymapnet_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:${YMAPNET_COMPILE_OPTIONS}>
)

# ---------------------------------------------------------------------------
# ymapnet executable (CLI, video loop, OpenCV windows)
# ---------------------------------------------------------------------------
if(YMAPNET_BUILD_CLI)
    add_executable(ymapnet
        src/main.cpp
    )

    target_link_libraries(ymapnet PRIVATE
        ymapnet_core
        ${OpenCV_LIBS}
    )

    # main.cpp only checks GGML_USE_CUDA for the --cpu sentinel
    if(CUDAToolkit_FOUND)
        target_compile_definitions(ymapnet PRIVATE GGML_USE_CUDA)
    endif()

    target_compile_options(ymapnet PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:${YMAPNET_COMPILE_OPTIONS}>
    )

    # skeleton-resolver parity check against ymapnet/skeletons (scripts/testSkeletonParity.py)
    add_executable(test_skeleton src/test_skeleton.cpp)
    target_include_directories(test_skeleton PRIVATE src ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_skeleton PRIVATE opencv_core opencv_imgproc)
endif()

message(STATUS "Targets: ymapnet_core")
message(STATUS "  CLI               : ${YMAPNET_BUILD_CLI}")
message(STATUS "  ggml CUDA backend : ${GGML_CUDA}")
message(STATUS "  ggml Vulkan backend: ${GGML_VULKAN}")
message(STATUS "  OpenCV libs       : ${OpenCV_LIBS}")
