cmake_minimum_required(VERSION 3.10)
project(CuFinufftMex)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# MATLAB setup
set(Matlab_ROOT_DIR "/mnt/gpussd1/jc_350/matlab/matlab2023b")
find_package(Matlab REQUIRED MX_LIBRARY ENG_LIBRARY)

# Include directories for MATLAB and CUDA
include_directories(${Matlab_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Assuming cufinufft is installed in a standard location, otherwise specify its path
find_path(CUFINUFFT_INCLUDE_DIR NAMES cufinufft.h)
find_library(CUFINUFFT_LIBRARY NAMES cufinufft)

# Include the cufinufft header directory
include_directories(${CUFINUFFT_INCLUDE_DIR})

# Define the MEX file
set(MEX_FILE_NAME mexFunctionName) # Change mexFunctionName to your actual MEX function name

# Define source files
set(SOURCE_FILES your_mex_file.c) # Change your_mex_file.c to your actual source file name

# Enable CUDA support in MEX file
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-R2018a)

# Add the MEX target
cuda_add_library(${MEX_FILE_NAME} SHARED ${SOURCE_FILES})

# Link with MATLAB, CUDA, and cufinufft libraries
target_link_libraries(${MEX_FILE_NAME} ${Matlab_LIBRARIES} ${CUDA_LIBRARIES} ${CUFINUFFT_LIBRARY})

# Set the output name of the MEX file to match MATLAB's expectations
set_target_properties(${MEX_FILE_NAME} PROPERTIES PREFIX "" SUFFIX .${Matlab_MEX_EXTENSION})

# Set output directory
set_target_properties(${MEX_FILE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
