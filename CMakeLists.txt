cmake_minimum_required(VERSION 3.18)  # Increased minimum version

# Add this line near the top of your CMakeLists.txt, after cmake_minimum_required
include(cmake/CPM.cmake)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)  # Adjust these values as needed

project(MT_CUDA CUDA C)



# Find CUDA
find_package(CUDA REQUIRED)

# Set MATLAB root directory, supplied by user
if(NOT DEFINED ENV{MATLAB_ROOT})
    message(FATAL_ERROR "MATLAB_ROOT environment variable is not set. Please set MATLAB_ROOT to the root directory of your MATLAB installation.")
endif()
set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})


# Find MATLAB
find_package(Matlab REQUIRED)

# Add MATLAB GPU Computing SDK include directory
include_directories(${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include)

# Set compiler flags for debugging
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Include directories
include_directories(${MATLAB_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Add FINUFFT include directories
include_directories(${CMAKE_BINARY_DIR}/_deps/finufft-src/include)

# Set custom FINUFFT with explicit build step
CPMAddPackage(
    NAME finufft
    GITHUB_REPOSITORY mikecjz/finufft
    GIT_TAG toeplitz_v2.0.3
    CMAKE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
    OPTIONS 
        "FINUFFT_STATIC_LINKING ON"
        "FINUFFT_USE_CPU OFF"
        "FINUFFT_USE_CUDA ON"
        "FINUFFT_BUILD_MATLAB OFF"
)

# Ensure cufinufft target exists
if(NOT TARGET cufinufft)
    message(FATAL_ERROR "cufinufft target not created")
endif()

# Create MEX targets
set(MEX_SOURCES
    src/AhA_ps_cufinufft.cu
    src/AhA_cufinufft_toep.cu
    src/cufinufftf2d1.cu
    src/cufinufftfspread2d1.cu
)

foreach(SOURCE ${MEX_SOURCES})
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    matlab_add_mex(
        NAME ${TARGET_NAME}
        SRC ${SOURCE}
        LINK_TO ${CUDA_LIBRARIES} 
               ${CUDA_cufft_LIBRARY}
               ${CUDA_cublas_LIBRARY}
        R2018a
    )
    
    # Add dependency to ensure FINUFFT is built first
    add_dependencies(${TARGET_NAME} cufinufft)
    
    # Link FINUFFT CUDA library using the correct target name
    target_link_libraries(${TARGET_NAME}
        cufinufft
    )
endforeach()
