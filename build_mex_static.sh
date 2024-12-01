#! /bin/bash

if [ -z "$MATLAB_ROOT" ]; then
    echo "Error: MATLAB_ROOT environment variable is not set"
    echo "Please set MATLAB_ROOT to the root directory of your MATLAB installation: \n export MATLAB_ROOT=</path/to/matlab>"
    echo "You can this path by running the command 'matlabroot' in MATLAB."
    exit 1
fi


mkdir -p build
cd build

# Actual build process
cmake ..
cmake --build . -j 8

# Move the built MEX files to the parent directory
mv *.mexa64 ../


cd ..
# Clean up the build directory
rm -rf build/