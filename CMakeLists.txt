# ============================================================================
#  MR_AHA - GPU forward normal operator for non-Cartesian MRI with low-rank
#            (subspace / MR-Multitasking) modelling.
#
#  Layout produced by this file:
#    mr_aha           - the standalone C/C++/CUDA library (the "C++ base")
#    AhA_Toeplitz_mex  - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    calc_toep_mex     - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    calc_mask_mex     - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    cufinufft (MEX)   - finufft's own MATLAB GPU interface, used by the
#                        NUFFT reference AhA_cufinufft.m (optional,
#                        MR_AHA_BUILD_CUFINUFFT_MATLAB)
#    libmr_aha_c.so   - shared library consumed by the Python ctypes binding
#                        (optional, MR_AHA_BUILD_PYTHON)
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

project(MR_AHA VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Position-independent code everywhere: the MEX gateways and the Python shared
# library both link the core as a static archive into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MR_AHA_BUILD_MATLAB "Build the MATLAB MEX interface"        OFF)
option(MR_AHA_BUILD_PYTHON "Build the shared library used by Python" ON)
option(MR_AHA_BUILD_EXAMPLES "Build the standalone C++ examples"   OFF)
option(MR_AHA_BUILD_CUFINUFFT_MATLAB
       "Also build finufft's cufinufft MATLAB MEX (needed by AhA_cufinufft.m)" OFF)

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
    if(MR_AHA_BUILD_CUFINUFFT_MATLAB)
        # finufft's own GPU MEX + .m wrappers, built from the same submodule
        # so that the NUFFT reference uses the very same cufinufft version.
        if(DEFINED ENV{MATLAB_ROOT} AND NOT DEFINED Matlab_ROOT_DIR)
            set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})
        endif()
        set(FINUFFT_BUILD_MATLAB ON  CACHE BOOL "" FORCE)
    else()
        set(FINUFFT_BUILD_MATLAB OFF CACHE BOOL "" FORCE)
    endif()
    set(FINUFFT_BUILD_PYTHON   OFF CACHE BOOL "" FORCE)
    set(FINUFFT_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(finufft EXCLUDE_FROM_ALL)
endif()

# ----------------------------------------------------------------------------
# The core library
# ----------------------------------------------------------------------------

add_library(mr_aha STATIC
    src/core/mr_config.cu
    src/core/mr_kernels.cu
    src/core/mr_aha.cu
    src/core/mr_calc_toep.cu
    src/core/mr_calc_mask.cu
)

target_include_directories(mr_aha
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(mr_aha
    PUBLIC  CUDA::cudart CUDA::cufft
    PRIVATE cufinufft
)

set_target_properties(mr_aha PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ----------------------------------------------------------------------------
# Shared library for the Python ctypes binding
# ----------------------------------------------------------------------------

if(MR_AHA_BUILD_PYTHON)
    add_library(mr_aha_c SHARED src/core/mr_shared_stub.cu)
    target_link_libraries(mr_aha_c PRIVATE
        # Whole-archive so that the extern "C" entry points, which nothing
        # inside the shared object references, survive the link.
        $<LINK_LIBRARY:WHOLE_ARCHIVE,mr_aha>
    )
    target_link_libraries(mr_aha_c PUBLIC CUDA::cudart CUDA::cufft)
    set_target_properties(mr_aha_c PROPERTIES
        OUTPUT_NAME mr_aha_c
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# ----------------------------------------------------------------------------
# MATLAB MEX interface
# ----------------------------------------------------------------------------

if(MR_AHA_BUILD_MATLAB)
    if(DEFINED ENV{MATLAB_ROOT} AND NOT DEFINED Matlab_ROOT_DIR)
        set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})
    endif()
    find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY MEX_COMPILER)

    # The MathWorks GPU Computing SDK (mxGPUArray) lives outside the default
    # MATLAB include path.
    set(MR_MATLAB_GPU_INCLUDE
        "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include")
    if(NOT EXISTS "${MR_MATLAB_GPU_INCLUDE}/gpu/mxGPUArray.h")
        message(FATAL_ERROR
            "mxGPUArray.h not found under ${MR_MATLAB_GPU_INCLUDE}. "
            "Parallel Computing Toolbox is required for the MATLAB interface.")
    endif()
    find_library(MR_MWGPU_LIBRARY mwgpu
        HINTS "${Matlab_ROOT_DIR}/bin/glnxa64"
              "${Matlab_ROOT_DIR}/bin/maci64"
              "${Matlab_ROOT_DIR}/bin/win64"
        REQUIRED)

    foreach(MR_MEX_NAME AhA_Toeplitz_mex calc_toep_mex calc_mask_mex)
        matlab_add_mex(
            NAME ${MR_MEX_NAME}
            SRC matlab/${MR_MEX_NAME}.cu
            R2018a
        )
        target_include_directories(${MR_MEX_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MR_MATLAB_GPU_INCLUDE})
        # FindMatlab's matlab_add_mex uses the plain target_link_libraries
        # signature, so this call must stay plain too.
        target_link_libraries(${MR_MEX_NAME}
            mr_aha ${MR_MWGPU_LIBRARY} CUDA::cudart)
        set_target_properties(${MR_MEX_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab)
    endforeach()
endif()

# finufft is added EXCLUDE_FROM_ALL, so pull its MEX into the default build.
if(MR_AHA_BUILD_CUFINUFFT_MATLAB AND TARGET cufinufft_mex)
    add_custom_target(mr_cufinufft_matlab ALL DEPENDS cufinufft_mex)
endif()

# ----------------------------------------------------------------------------
# Standalone C++ examples
# ----------------------------------------------------------------------------

if(MR_AHA_BUILD_EXAMPLES)
    add_executable(mr_example_aha examples/example_aha.cu)
    target_link_libraries(mr_example_aha PRIVATE mr_aha)
endif()

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS mr_aha ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MR_AHA_BUILD_PYTHON)
    install(TARGETS mr_aha_c LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
