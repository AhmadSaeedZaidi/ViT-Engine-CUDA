cmake_minimum_required(VERSION 3.18)
project(vit_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Auto-discover PyTorch CMake configuration from Python environment
if(NOT Torch_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RESULT
    )
    if(TORCH_PREFIX_RESULT EQUAL 0 AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    endif()
    find_package(Torch REQUIRED)
endif()

# Apply PyTorch CXX flags (e.g. CXX11 ABI compatibility)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -----------------------------------------------------------------------------
# 1. CUDA Kernels Target (Pure CUDA implementations)
# -----------------------------------------------------------------------------
file(GLOB KERNEL_SRCS CONFIGURE_DEPENDS "csrc/kernels/*.cu")

add_library(vit_cuda_kernels STATIC ${KERNEL_SRCS})

set_target_properties(vit_cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES "75;86"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(vit_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --allow-unsupported-compiler
        -use_fast_math
        --expt-relaxed-constexpr
    >
)

target_include_directories(vit_cuda_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/include>
)

target_link_libraries(vit_cuda_kernels PUBLIC
    CUDA::cudart
)

# -----------------------------------------------------------------------------
# 2. PyTorch C++ Ops Wrapper Target (Bridges PyTorch tensors to CUDA kernels)
# -----------------------------------------------------------------------------
file(GLOB OPS_SRCS CONFIGURE_DEPENDS "csrc/ops/*.cpp")

add_library(vit_cuda_ops STATIC ${OPS_SRCS})

set_target_properties(vit_cuda_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(vit_cuda_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_include_directories(vit_cuda_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/include>
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(vit_cuda_ops PUBLIC
    vit_cuda_kernels
    ${TORCH_LIBRARIES}
)

# -----------------------------------------------------------------------------
# 3. Python Extension Module Target (_C / Pybind11 layer)
# -----------------------------------------------------------------------------
# Explicitly specify the binding source file
Python_add_library(_C MODULE "csrc/bindings/binding.cpp" WITH_SOABI)

set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(_C PRIVATE
    TORCH_EXTENSION_NAME=_C
)

target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_include_directories(_C PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/include>
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE
    vit_cuda_ops
    vit_cuda_kernels
    ${TORCH_LIBRARIES}
)

# Copy _C extension into vit_cuda/ source package for immediate in-place imports
add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_C>
        "${CMAKE_CURRENT_SOURCE_DIR}/vit_cuda/$<TARGET_FILE_NAME:_C>"
    COMMENT "Copying _C extension into vit_cuda source package"
)

# Install rule for scikit-build-core packaging
install(TARGETS _C LIBRARY DESTINATION vit_cuda)
