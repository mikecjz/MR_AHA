# ============================================================================
#  MR_AHA - GPU forward normal operator for non-Cartesian MRI with low-rank
#            (subspace / MR-Multitasking) modelling.
#
#  The library is built for exactly one GPU backend, chosen with
#  MR_AHA_BACKEND:
#    CUDA   - NVIDIA GPUs: CUDA kernels, cuFFT, cufinufft (default off macOS)
#    METAL  - Apple silicon: Metal kernels, VkFFT-Metal (default on macOS)
#
#  Layout produced by this file:
#    mr_aha           - the standalone C/C++ library (the "C++ base")
#    AhA_Toeplitz_mex  - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    calc_toep_mex     - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    calc_mask_mex     - MATLAB MEX gateway (optional, MR_AHA_BUILD_MATLAB)
#    mr_aha_op_mex     - MATLAB MEX gateway of the multi-GPU operator class
#                        mr_aha_op (optional, MR_AHA_BUILD_MATLAB, CUDA only)
#    cufinufft (MEX)   - finufft's own MATLAB GPU interface, used by the
#                        NUFFT reference AhA_cufinufft.m (optional,
#                        MR_AHA_BUILD_CUFINUFFT_MATLAB, CUDA only)
#    libmr_aha_c       - shared library consumed by the Python ctypes binding
#                        (optional, MR_AHA_BUILD_PYTHON; .so or .dylib)
#
#  Sources:
#    src/common/  backend-independent configuration and error handling
#    src/cuda/    CUDA backend (depends on the CUDA toolkit and on cufinufft
#                 from the bundled `third_party/finufft` submodule)
#    src/metal/   Metal backend (depends on the Metal framework and on VkFFT /
#                 metal-cpp from the `third_party/VkFFT` submodule)
# ============================================================================

cmake_minimum_required(VERSION 3.24)

# ----------------------------------------------------------------------------
# Backend selection (before project(), so that CUDA is only enabled when used)
# ----------------------------------------------------------------------------

if(APPLE)
    set(MR_AHA_DEFAULT_BACKEND METAL)
else()
    set(MR_AHA_DEFAULT_BACKEND CUDA)
endif()
set(MR_AHA_BACKEND "${MR_AHA_DEFAULT_BACKEND}" CACHE STRING
    "GPU backend: CUDA (NVIDIA GPUs) or METAL (Apple silicon)")
set_property(CACHE MR_AHA_BACKEND PROPERTY STRINGS CUDA METAL)
string(TOUPPER "${MR_AHA_BACKEND}" MR_AHA_BACKEND)
if(NOT MR_AHA_BACKEND MATCHES "^(CUDA|METAL)$")
    message(FATAL_ERROR "MR_AHA_BACKEND must be CUDA or METAL, got '${MR_AHA_BACKEND}'")
endif()

# Must be set before the CUDA language is enabled.
if(MR_AHA_BACKEND STREQUAL "CUDA" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89;90 CACHE STRING
        "CUDA architectures to generate code for")
endif()

project(MR_AHA VERSION 1.0.0 LANGUAGES C CXX)
message(STATUS "MR_AHA backend: ${MR_AHA_BACKEND}")

if(MR_AHA_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
elseif(NOT APPLE)
    message(FATAL_ERROR "MR_AHA_BACKEND=METAL requires macOS on Apple silicon")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Position-independent code everywhere: the MEX gateways and the Python shared
# library both link the core as a static archive into a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MR_AHA_BUILD_MATLAB "Build the MATLAB MEX interface"        OFF)
option(MR_AHA_BUILD_PYTHON "Build the shared library used by Python" ON)
option(MR_AHA_BUILD_EXAMPLES "Build the standalone C++ examples"   OFF)
option(MR_AHA_BUILD_CUFINUFFT_MATLAB
       "Also build finufft's cufinufft MATLAB MEX (needed by AhA_cufinufft.m)" OFF)

if(MR_AHA_BACKEND STREQUAL "METAL")
    foreach(MR_OPT MR_AHA_BUILD_EXAMPLES MR_AHA_BUILD_CUFINUFFT_MATLAB)
        if(${MR_OPT})
            message(FATAL_ERROR "${MR_OPT} is not available for the Metal "
                    "backend yet (CUDA only).")
        endif()
    endforeach()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Sources shared by every backend.
set(MR_COMMON_SOURCES
    src/common/mr_config.cpp
)

if(MR_AHA_BACKEND STREQUAL "CUDA")

# ============================================================================
# CUDA backend
# ============================================================================

find_package(CUDAToolkit REQUIRED)

# cufinufft, from the bundled `third_party/finufft` submodule. Only the CUDA
# half is needed. Its build tree stays at `<build>/finufft` (explicit binary
# directory below), so the MATLAB wrappers remain in `<build>/finufft/matlab`.
if(NOT TARGET cufinufft)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/finufft/CMakeLists.txt")
        message(FATAL_ERROR
            "The finufft submodule is missing. Run:\n"
            "  git submodule update --init third_party/finufft")
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
    add_subdirectory(third_party/finufft finufft EXCLUDE_FROM_ALL)
endif()

add_library(mr_aha STATIC
    ${MR_COMMON_SOURCES}
    src/cuda/mr_backend_cuda.cu
    src/cuda/mr_kernels.cu
    src/cuda/mr_kernels_bd.cu
    src/cuda/mr_aha.cu
    src/cuda/mr_op.cu
    src/cuda/mr_nufft_op.cu
    src/cuda/mr_calc_toep.cu
    src/cuda/mr_toep_mgpu.cu
    src/cuda/mr_calc_mask.cu
)

target_include_directories(mr_aha
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
)

# Every translation unit that sees mr_aha.h, nvcc-compiled or not, must agree
# that mr_complex is cuFloatComplex; see the typedef in the header.
target_compile_definitions(mr_aha PUBLIC MR_AHA_CUDA_COMPLEX)

# CUDA::cuda_driver: the operator (src/cuda/mr_op.cu) checks whether a
# device's primary context still exists (cuDevicePrimaryCtxGetState) before
# releasing it.
# CUDA::cublas: the Gram multiplication of the NUFFT method is a batched GEMM
# (src/cuda/mr_nufft_op.cu).
# Threads::Threads: the multi-GPU kernel computation drives each GPU from its
# own host thread (std::thread, src/cuda/mr_toep_mgpu.cu).
find_package(Threads REQUIRED)
target_link_libraries(mr_aha
    PUBLIC  CUDA::cudart CUDA::cufft CUDA::cublas CUDA::cuda_driver Threads::Threads
    PRIVATE cufinufft
)

# mr_kernels_bd.cu instantiates the block-diagonal kernels for every rank up
# to 32 (about 1.5 min of ptxas per architecture); let nvcc compile the
# architectures of CMAKE_CUDA_ARCHITECTURES in parallel.
target_compile_options(mr_aha PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>)

set_target_properties(mr_aha PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set(MR_BACKEND_LINK_LIBRARIES CUDA::cudart CUDA::cufft CUDA::cublas CUDA::cuda_driver)

else()

# ============================================================================
# Metal backend
# ============================================================================

# VkFFT (header-only) and metal-cpp (header-only, bundled with VkFFT), from the
# `third_party/VkFFT` submodule. It points to the mikecjz/VkFFT fork, which
# carries the Metal autorelease fix (see
# https://mikecjz.github.io/MR_AHA_docs/benchmarks/vkfft.html).
set(MR_VKFFT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/VkFFT")
if(NOT EXISTS "${MR_VKFFT_DIR}/vkFFT/vkFFT.h")
    message(FATAL_ERROR
        "The VkFFT submodule is missing. Run:\n"
        "  git submodule update --init third_party/VkFFT")
endif()

find_library(MR_METAL_FRAMEWORK Metal REQUIRED)
find_library(MR_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(MR_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
set(MR_BACKEND_LINK_LIBRARIES
    ${MR_METAL_FRAMEWORK} ${MR_FOUNDATION_FRAMEWORK} ${MR_QUARTZCORE_FRAMEWORK})

# Metal kernels: always embedded as source (compiled by the system's Metal
# compiler on first use); additionally precompiled to a .metallib and embedded
# when the Metal toolchain is installed (`xcrun metal`, part of Xcode's Metal
# Toolchain component), which moves shader errors to build time and skips the
# first-use compile. See src/metal/mr_metal_runtime.cpp.
option(MR_AHA_METAL_PRECOMPILE
       "Precompile the Metal kernels when the Metal toolchain is available" ON)
set(MR_METAL_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/mr_kernels.metal")
set(MR_METAL_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(MR_EMBED "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mr_embed.cmake")
file(MAKE_DIRECTORY "${MR_METAL_GEN}")

add_custom_command(
    OUTPUT  "${MR_METAL_GEN}/mr_metal_source.cpp"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${MR_METAL_KERNELS}
            -DOUTPUT=${MR_METAL_GEN}/mr_metal_source.cpp
            -DSYMBOL=mr_metal_source -P ${MR_EMBED}
    DEPENDS "${MR_METAL_KERNELS}" "${MR_EMBED}"
    COMMENT "Embedding the Metal kernel source")

set(MR_METAL_TOOLCHAIN OFF)
if(MR_AHA_METAL_PRECOMPILE)
    execute_process(COMMAND xcrun -sdk macosx metal --version
                    RESULT_VARIABLE MR_METAL_RC OUTPUT_QUIET ERROR_QUIET)
    if(MR_METAL_RC EQUAL 0)
        set(MR_METAL_TOOLCHAIN ON)
    else()
        message(STATUS "Metal toolchain not found (xcrun metal); the kernels "
                       "will be compiled from source at run time")
    endif()
endif()

if(MR_METAL_TOOLCHAIN)
    # Metal 3.0 (macOS 13+) with fast math off, matching the run-time path.
    add_custom_command(
        OUTPUT  "${MR_METAL_GEN}/mr_kernels.air"
        COMMAND xcrun -sdk macosx metal -std=metal3.0 -fno-fast-math
                -mmacosx-version-min=13.0 -c ${MR_METAL_KERNELS}
                -o ${MR_METAL_GEN}/mr_kernels.air
        DEPENDS "${MR_METAL_KERNELS}"
        COMMENT "Compiling the Metal kernels")
    add_custom_command(
        OUTPUT  "${MR_METAL_GEN}/mr_kernels.metallib"
        COMMAND xcrun -sdk macosx metallib ${MR_METAL_GEN}/mr_kernels.air
                -o ${MR_METAL_GEN}/mr_kernels.metallib
        DEPENDS "${MR_METAL_GEN}/mr_kernels.air"
        COMMENT "Linking the Metal kernel library")
    set(MR_METALLIB_INPUT "${MR_METAL_GEN}/mr_kernels.metallib")
    set(MR_METALLIB_DEPENDS "${MR_METAL_GEN}/mr_kernels.metallib")
    message(STATUS "Metal kernels: precompiled metallib (+ source fallback)")
else()
    set(MR_METALLIB_INPUT "")
    set(MR_METALLIB_DEPENDS "")
    message(STATUS "Metal kernels: compiled from embedded source at run time")
endif()
add_custom_command(
    OUTPUT  "${MR_METAL_GEN}/mr_metal_metallib.cpp"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${MR_METALLIB_INPUT}
            -DOUTPUT=${MR_METAL_GEN}/mr_metal_metallib.cpp
            -DSYMBOL=mr_metal_metallib -P ${MR_EMBED}
    DEPENDS ${MR_METALLIB_DEPENDS} "${MR_EMBED}"
    COMMENT "Embedding the Metal kernel library")

add_library(mr_aha STATIC
    ${MR_COMMON_SOURCES}
    src/metal/mr_backend_metal.cpp
    src/metal/mr_metal_runtime.cpp
    src/metal/mr_metal_fft.cpp
    src/metal/mr_aha_metal.cpp
    src/metal/mr_calc_toep_metal.cpp
    src/metal/mr_calc_mask_metal.cpp
    "${MR_METAL_GEN}/mr_metal_source.cpp"
    "${MR_METAL_GEN}/mr_metal_metallib.cpp"
)

target_include_directories(mr_aha
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common
        ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
)
# SYSTEM: metal-cpp and VkFFT emit hundreds of deprecation warnings that are
# not ours to fix.
target_include_directories(mr_aha SYSTEM PRIVATE
    "${MR_VKFFT_DIR}/metal-cpp"
    "${MR_VKFFT_DIR}/vkFFT"
)
target_compile_definitions(mr_aha PRIVATE VKFFT_BACKEND=5)
target_link_libraries(mr_aha PUBLIC ${MR_BACKEND_LINK_LIBRARIES})

endif()

# ----------------------------------------------------------------------------
# Shared library for the Python ctypes binding
# ----------------------------------------------------------------------------

if(MR_AHA_BUILD_PYTHON)
    add_library(mr_aha_c SHARED src/common/mr_shared_stub.cpp)
    target_link_libraries(mr_aha_c PRIVATE
        # Whole-archive so that the extern "C" entry points, which nothing
        # inside the shared object references, survive the link.
        $<LINK_LIBRARY:WHOLE_ARCHIVE,mr_aha>
    )
    target_link_libraries(mr_aha_c PUBLIC ${MR_BACKEND_LINK_LIBRARIES})
    set_target_properties(mr_aha_c PROPERTIES
        OUTPUT_NAME mr_aha_c
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# ----------------------------------------------------------------------------
# MATLAB MEX interface
#   CUDA : gpuArray gateways (matlab/*.cu), need the Parallel Computing Toolbox
#   METAL: host-array gateways (matlab/metal/*.cpp); calc_toep and the
#          threshold mask go through MATLAB's Python interface (pyenv)
#   both : mr_backend_mex, which the .m wrappers use to pick the data path
# ----------------------------------------------------------------------------

if(MR_AHA_BUILD_MATLAB)
    if(DEFINED ENV{MATLAB_ROOT} AND NOT DEFINED Matlab_ROOT_DIR)
        set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})
    endif()
    find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY MEX_COMPILER)

    set(MR_MEX_TARGETS mr_backend_mex)
    matlab_add_mex(NAME mr_backend_mex SRC matlab/mr_backend_mex.c R2018a)

    if(MR_AHA_BACKEND STREQUAL "CUDA")
        # The MathWorks GPU Computing SDK (mxGPUArray) lives outside the
        # default MATLAB include path.
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

        # mr_aha_op_mex: plan-based multi-GPU A^H A behind the mr_aha_op
        # handle class (matlab/mr_aha_op.m); CUDA only.
        foreach(MR_MEX_NAME AhA_Toeplitz_mex calc_toep_mex calc_mask_mex
                            mr_aha_op_mex)
            matlab_add_mex(
                NAME ${MR_MEX_NAME}
                SRC matlab/${MR_MEX_NAME}.cu
                R2018a
            )
            target_include_directories(${MR_MEX_NAME} PRIVATE
                ${MR_MATLAB_GPU_INCLUDE})
            list(APPEND MR_MEX_TARGETS ${MR_MEX_NAME})
        endforeach()
        set(MR_MEX_EXTRA_LIBS ${MR_MWGPU_LIBRARY} CUDA::cudart)
    else()
        foreach(MR_MEX_NAME AhA_Toeplitz_mex calc_mask_mex)
            matlab_add_mex(
                NAME ${MR_MEX_NAME}
                SRC matlab/metal/${MR_MEX_NAME}.cpp
                R2018a
            )
            list(APPEND MR_MEX_TARGETS ${MR_MEX_NAME})
        endforeach()
        set(MR_MEX_EXTRA_LIBS "")
    endif()

    foreach(MR_MEX_NAME ${MR_MEX_TARGETS})
        target_include_directories(${MR_MEX_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        # FindMatlab's matlab_add_mex uses the plain target_link_libraries
        # signature, so this call must stay plain too.
        target_link_libraries(${MR_MEX_NAME} mr_aha ${MR_MEX_EXTRA_LIBS})
        set_target_properties(${MR_MEX_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab)
        if(APPLE)
            # FindMatlab exports the C++ MEX API's entry points from every MEX
            # file on macOS (cppMexFunction.map + -Wl,-U). The linker of
            # Xcode 26+ rejects exported symbols that are undefined, and these
            # gateways use the C API, so export exactly what `mex` exports for
            # a C MEX file.
            set_target_properties(${MR_MEX_NAME} PROPERTIES LINK_FLAGS
                "-Wl,-exported_symbols_list,${Matlab_EXTERN_LIBRARY_DIR}/c_exportsmexfileversion.map")
        endif()
    endforeach()
endif()

# finufft is added EXCLUDE_FROM_ALL, so pull its MEX into the default build.
if(MR_AHA_BUILD_CUFINUFFT_MATLAB AND TARGET cufinufft_mex)
    add_custom_target(mr_cufinufft_matlab ALL DEPENDS cufinufft_mex)
endif()

# ----------------------------------------------------------------------------
# Standalone C++ examples (CUDA backend)
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
