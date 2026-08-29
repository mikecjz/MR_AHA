# ============================================================================
#  MT_CUDA - GPU forward normal operator for non-Cartesian MRI with low-rank
#            (subspace / MR-Multitasking) modelling.
#
#  Layout produced by this file:
#    mt_cuda           - the standalone C/C++/CUDA library (the "C++ base")
#    AhA_Toeplitz_mex  - MATLAB MEX gateway (optional, MT_CUDA_BUILD_MATLAB)
#    calc_toep_mex     - MATLAB MEX gateway (optional, MT_CUDA_BUILD_MATLAB)
#    libmt_cuda_c.so   - shared library consumed by the Python ctypes binding
#                        (optional, MT_CUDA_BUILD_PYTHON)
#
#  The library only depends on the CUDA toolkit (cudart, cuFFT) and on
#  cufinufft, which is pulled in from the bundled `finufft/` submodule.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# Must be set before project() so that it applies to the CUDA language check.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89;90 CACHE STRING
        "CUDA architectures to generate code for")
endif()

project(MT_CUDA VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Position-independent code everywhere: the MEX gateways and the Python shared
# library both link the core as a static archive into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MT_CUDA_BUILD_MATLAB "Build the MATLAB MEX interface"        OFF)
option(MT_CUDA_BUILD_PYTHON "Build the shared library used by Python" ON)
option(MT_CUDA_BUILD_EXAMPLES "Build the standalone C++ examples"   OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

# cufinufft, from the bundled submodule. Only the CUDA half is needed.
if(NOT TARGET cufinufft)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/finufft/CMakeLists.txt")
        message(FATAL_ERROR
            "The finufft submodule is missing. Run:\n"
            "  git submodule update --init --recursive")
    endif()
    set(FINUFFT_USE_CPU        OFF CACHE BOOL "" FORCE)
    set(FINUFFT_USE_CUDA       ON  CACHE BOOL "" FORCE)
    set(FINUFFT_STATIC_LINKING ON  CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_MATLAB   OFF CACHE BOOL "" FORCE)
    set(FINUFFT_BUILD_PYTHON   OFF CACHE BOOL "" FORCE)
    set(FINUFFT_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(finufft EXCLUDE_FROM_ALL)
endif()

# ----------------------------------------------------------------------------
# The core library
# ----------------------------------------------------------------------------

add_library(mt_cuda STATIC
    src/core/mt_config.cu
    src/core/mt_kernels.cu
    src/core/mt_aha.cu
    src/core/mt_calc_toep.cu
)

target_include_directories(mt_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(mt_cuda
    PUBLIC  CUDA::cudart CUDA::cufft
    PRIVATE cufinufft
)

set_target_properties(mt_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ----------------------------------------------------------------------------
# Shared library for the Python ctypes binding
# ----------------------------------------------------------------------------

if(MT_CUDA_BUILD_PYTHON)
    add_library(mt_cuda_c SHARED src/core/mt_shared_stub.cu)
    target_link_libraries(mt_cuda_c PRIVATE
        # Whole-archive so that the extern "C" entry points, which nothing
        # inside the shared object references, survive the link.
        $<LINK_LIBRARY:WHOLE_ARCHIVE,mt_cuda>
    )
    target_link_libraries(mt_cuda_c PUBLIC CUDA::cudart CUDA::cufft)
    set_target_properties(mt_cuda_c PROPERTIES
        OUTPUT_NAME mt_cuda_c
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# ----------------------------------------------------------------------------
# MATLAB MEX interface
# ----------------------------------------------------------------------------

if(MT_CUDA_BUILD_MATLAB)
    if(DEFINED ENV{MATLAB_ROOT} AND NOT DEFINED Matlab_ROOT_DIR)
        set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})
    endif()
    find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY MEX_COMPILER)

    # The MathWorks GPU Computing SDK (mxGPUArray) lives outside the default
    # MATLAB include path.
    set(MT_MATLAB_GPU_INCLUDE
        "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include")
    if(NOT EXISTS "${MT_MATLAB_GPU_INCLUDE}/gpu/mxGPUArray.h")
        message(FATAL_ERROR
            "mxGPUArray.h not found under ${MT_MATLAB_GPU_INCLUDE}. "
            "Parallel Computing Toolbox is required for the MATLAB interface.")
    endif()
    find_library(MT_MWGPU_LIBRARY mwgpu
        HINTS "${Matlab_ROOT_DIR}/bin/glnxa64"
              "${Matlab_ROOT_DIR}/bin/maci64"
              "${Matlab_ROOT_DIR}/bin/win64"
        REQUIRED)

    foreach(MT_MEX_NAME AhA_Toeplitz_mex calc_toep_mex)
        matlab_add_mex(
            NAME ${MT_MEX_NAME}
            SRC matlab/${MT_MEX_NAME}.cu
            R2018a
        )
        target_include_directories(${MT_MEX_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MT_MATLAB_GPU_INCLUDE})
        # FindMatlab's matlab_add_mex uses the plain target_link_libraries
        # signature, so this call must stay plain too.
        target_link_libraries(${MT_MEX_NAME}
            mt_cuda ${MT_MWGPU_LIBRARY} CUDA::cudart)
        set_target_properties(${MT_MEX_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab)
    endforeach()
endif()

# ----------------------------------------------------------------------------
# Standalone C++ examples
# ----------------------------------------------------------------------------

if(MT_CUDA_BUILD_EXAMPLES)
    add_executable(mt_example_aha examples/example_aha.cu)
    target_link_libraries(mt_example_aha PRIVATE mt_cuda)
endif()

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS mt_cuda ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MT_CUDA_BUILD_PYTHON)
    install(TARGETS mt_cuda_c LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
