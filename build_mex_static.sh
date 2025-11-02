#! /bin/bash

if [ -z "$MATLAB_ROOT" ]; then
    echo "Error: MATLAB_ROOT environment variable is not set"
    echo "Please set MATLAB_ROOT to the root directory of your MATLAB installation: \n export MATLAB_ROOT=</path/to/matlab>"
    echo "You can this path by running the command 'matlabroot' in MATLAB."
    exit 1
fi

# Parse command line arguments
CMAKE_BUILD_TYPE="Release"
CUFINUFFT_DEBUG="OFF"
REBUILD_FLAG=false

# Function to display help message
show_help() {
    echo "Usage: ./build_mex_static.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  --debug              Build MEX files in debug mode"
    echo "  --rebuild            Clean up the build directory before building"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --debug)
            CMAKE_BUILD_TYPE="Debug"
            shift
            ;;
        --rebuild)
            REBUILD_FLAG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help to see available options"
            exit 1
            ;;
    esac
done

# Clean up the build directory only if --rebuild flag is set
if [ "$REBUILD_FLAG" = true ]; then
    rm -rf build/
fi

mkdir -p build
cd build

# Actual build process with debug flags
cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
cmake --build . -j 8

# Move the built MEX files to the parent directory
mv *.mexa64 ../


cd ..
