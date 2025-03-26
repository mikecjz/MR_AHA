# MT_CUDA 

This repository holds custom CUDA code that can be used for low-rank MR reconstruction for
stack-of-stars sampling trajectories. It contains:

1. [cuFINUFFT-based AhA Operator](#cufinufft-based-aha-operator-less-memory-intensive)
2. [Toeplitz-based AhA Operator](#toeplitz-based-aha-operator-very-memory-intensive)
3. [MATLAB-cuFINUFFT interface for type-1 NUFFT](.src/cufinufftf2d1.cu)

## cuFINUFFT-based AhA Operator (Less Memory Intensive)

An implementation in c and cuda-based source codes to perform efficient GPU-based AhA operation for
stack-of-stars trajectory in MR-Multitasking reconconstruction. Under the hood, it makes API calls
to [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) and
[cufinufft](https://finufft.readthedocs.io/enlatest/c_gpu.html) for both uniform and non-uniform
FFTs. This repo also contains self-defined CUDA kernels to perform operations such as
coil-sensitivity maps multiplication (broadcast) and block-diagnal low-rank kspace update with
$\Phi\Phi^H$. The main function is [AhA_ps_cufinufft.cu](AhA_ps_cufinufft.cu). Additionally, this
repo contains example simple nufft API calls to [cufinufftf2d1(GPU)](cufinufftf2d1.cu) and
[finufftf2d1(CPU)](finufftf2d1.c).

>ⓘ Use [cu_finufft_test.m](./cu_finufft_test.m) to test the Toeplitz-based operator. Test data is
>provided and can be downloaded here --> ([cuFINUFFT Test
>Data](https://junzhou.chen.engineer/MT_TEST/AhAx_test.mat)).

At the moment, the AhA operation takes around 5.8 seconds for a image matrix size of $320 \times 320
\times 56$ with $L = 16$ and $Ncoils = 12$.

>Hardware environments:
> * NVIDIA GeForce RTX 3090 (24G)
> * NVIDIA-SMI 535.171.04
> * Driver Version: 535.171.04
> * CUDA Version: 12.2

## Toeplitz-based AhA Operator (VERY Memory Intensive)

### The Fastest Stack-of-Stars MR-Multitasking Reconstruction West of Mississippi ([Why Toeplitz?](#why-toeplitz-and-how-does-it-work))

An implementation c and cuda-based source codes to perform efficient GPU-and-Toeplitz-based AhA
operation for stack-of-stars trajectory in MR-Multitasking reconstruction. Under the hood, it makes
API calls to [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) for uniform FFTs. We use a
[modified cufinufft](https://github.com/mikecjz/finufft/tree/toeplitz) to
[spread](./src/cufinufftfspread2d1.cu) the non-Cartesian $\Phi\Phi^H$ operator to Cartesian grid.

>ⓘ Use [toeplitz_test.m](./toeplitz_test.m) to test the Toeplitz-based operator. Test data is
>provided and can be downloaded here --> ([Toeplitz Test
>Data](https://junzhou.chen.engineer/MT_TEST/AhAx_test_toeplitz.mat)).


At the moment, the Toeplitz AhA operation takes around 1.2 seconds for a image matrix size of $320
\times 320 \times 56$ with $L = 12$ and $Ncoils = 12$.

>Hardware environments:
>
> * NVIDIA GeForce RTX 3090 (24G)
> * NVIDIA-SMI 535.171.04
> * Driver Version: 535.171.04
> * CUDA Version: 12.2
>
> ⚠️ The above image dimensions are about the maximum this hardware can support due to memory
> limitations.

## Compilation and Usage
>
>CAUTION: This project is in its alpha phase. Please exercise patience when setting up necessary
>environments for compilation and runtime. The author has undertaken efforts to reduce the chance of
>unexpected errors but they still might occur and lead to matlab crashing.

#### Dependencies

1. You will need to have [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
2. You will need to download and compile [modified
   finufft](https://github.com/mikecjz/finufft/tree/toeplitz) with [CUDA
   enabled](https://finufft.readthedocs.io/en/latest/install_gpu.html).
3. You will need matlab version R2018a or later.

#### Compilation

1. Set the paths in [compile_mex.m](./compile_mex.m).
2. run [compile_mex.m](./compile_mex.m) in matlab.

>To compile with debug flags, uncomment the debug section.

#### Runtime Environments
Because the mex is compiled with references to compiled dynamic libraries from finufft and cuFFT, it
is necessary to let matlab know where to find these libraries. This is achieved by setting the
environment variable `LD_LIBRARY_PATH` before launching matlab. Additionally, in a multi-GPU server,
we want to handle the GPU selection by setting the encironment variable `CUDA_VISIBLE_DEVICES`
before launching matlab since we are making low-level API calls to CUDA codes. 

A shell script [matlab_with_env.sh](./scripts/matlab_with_env.sh) is provided to launch matlab after
selecting the GPU with the largest available memory and setting the `LD_LIBRARY_PATH` variable.

1. Set the `FINUFFT_LIB` and `CUDA_LIB` variables inside the script.
2. Make sure the script is executable. If not, run in shell:

    ```bash
    chmod +x ./matlab_with_env.sh
    ```

3. Launch matlab using the script

    ```bash
    ./matlab_with_env.sh
    ```

    >Note: Copy the script in one of your `PATH` folders (e.g. `$HOME/.local/bin`) to launch without
    >the relative path indicator `./`

### Linking cuFINUFFT statically (CMake)

---
A script ([build_mex_static.sh](./build_mex_static.sh)) is provided to compile the mex files using
CMake. The only requirement is you set the env variable `MATLAB_ROOT` before launching.

To figure out your matlabroot, run in matlab

```matlab
matlabroot
```

Set environment and run compile script

```bash
export MATLAB_ROOT=</path/to/matlab>

./build_mex_static.sh
```

This will pull the [modified finufft](https://github.com/mikecjz/finufft/tree/toeplitz)
automatically, compile it, and link its static `libcufinufft.a` library to the mex files.

The `FINUFFT_LIB` environment is then unnecessary. `CUDA_LIB` is still required.
>#### :warning: Known Issue 
>If you use CMake to compile. [Manually initializing
>GPU](https://www.mathworks.com/help/parallel-computing/mxinitgpu.html) in matlab before running the
>mex functions.
> * Option 1: Run gpuDevice(gpuIndex).
> * Option 2: Create any gpuArray.
>
>Otherwise you will get an error. Something like:
>
>```
>Invalid MEX-file 'MT_CUDA/AhA_ps_cufinufft.mexa64': MT_CUDA/>AhA_ps_cufinufft.mexa64: undefined symbol: mxInitGPU
>```

## Development and Debugging in Visual Studio Code

This code was developed in Visual Studio Code remotely with ssh. Set the `includePath` in
[c_cpp_properties.json](/.vscode/c_cpp_properties.json) for IntelliSense and auto code completion.

To launch debug session:

1. Compile mex with debug flags enabled.
2. Change `program` in [launch.json](/.vscode/launch.json) to the matlab executable file in your
   server.
3. Make sure `environment` for `LD_LIBRARY_PATH` in [launch.json](/.vscode/launch.json) is set to
   the specification described above.
4. Set breakpoints in [AhA_cufinufft_toep.cu](./src/AhA_cufinufft_toep.cu) and launch directly in VS
   Code.

Use the [MATLAB GPU CUDA and MEX Programming
Manual](https://www.mathworks.com/help/parallel-computing/gpu-cuda-and-mex-programming.html) as a
reference for development.

## Useful troubleshooting commands

```bash
nm -D <FINUFFT_DIR>/build/libcufinufft.so | grep cufinufft2d1_exec
```

## Why Toeplitz and How Does It Work?

The AhA operator for MRI is [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) under
arbitrary non-Cartesian trajectories ([Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)). A Toeplitz matrix can be padded to 2X
on its original dimensions to form a [Circulant
matrix](https://en.wikipedia.org/wiki/Circulant_matrix), and be [calculated via
FFTs](https://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf), requiring only
$O(N \log N)$ operations instead of $O(N^2)$ for general matrix-vector multiplications. In other
words, the Toeplitz operator can be diagonalized by FFTs.

$$Tx \implies F^H\Lambda F\hat{x}, $$

>* $T \in \mathbb{C}^{N \times N}$ is a symmetric Toeplitz operator
>* $x \in \mathbb{C}^{N}$ is a vector
>* $F \in \mathbb{C}^{2N \times 2N}$ is the 2D DFT operator
>* $\Lambda \in \mathbb{C}^{2N \times 2N}$ is a diagonal matrix
>* $\hat{x} \in \mathbb{C}^{2N}$ is the zero-padded vector of $x$

The diagonal elements of $\Lambda$ can be derived based on the non-Cartesian coordinates but we
usually use NUFFT packages to estimate them (see [Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)).

Under MR-Multitasking, the AhA operator is block-Toeplitz, which can be diagonalized similarly, with
the "diagonal" aspect being block-diagonal, a.k.a. the gridded $\Phi\Phi^H$ operator.

The challenge for 3D stack-of-stars AhA is that there is usually not enough GPU memory to hold the
gridded $\Phi\Phi^H$ operator, which has a size of $2Nx \times 2Ny \times Nz \times L \times L$. For
the image parameters mentioned above, the gridded $\Phi\Phi^H$ operator is about 26GB under single
precision and a 24G GPU cannot hold it.

Luckily for us, each individual matrix ($L \times L$) of the gridded $\Phi\Phi^H$ operator is
[Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix). This means that the lower triangular
portion of the matrix is an exact copy of the complex conjugate of the upper portion and vice versa.
If we only store the lower triangular portion of the gridded $\Phi\Phi^H$ operator, which will only
require about 13GB memory, the 3D AhA operation becomes feasible on the GPU, albeit still memory
demanding. See [AhA_cufinufft_toep.cu](./src/AhA_cufinufft_toep.cu#L328) for details on how to
handle if the operator is only stored in the lower triangular portion.

>"But I saw non-Hermitian $\Phi\Phi^H$ after gridding when I was doing 2D Toeplitz and we had this whole thing to enforce symmetric positive semi-definitness via eigen decomposition?", [Zihao](https://mrrl.ucla.edu/pages/Zihao_Chen) might ask.

Good question! non-Hermitian matrices after gridding is also what I saw if I follow the exact
formula detailed in [Baron et al., MRM 2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928).
For non-low-rank applications, this formula no doubt works well. However, for the low-rank case like
MR-Multitasking, let's take a closer look why the Hermitian symmetry might be disrupted.

A standard type 1 NUFFT (non-uniform to uniform) involves the following steps:

1. Spread the non-Cartesian coordinates to oversampled Cartesian grid via convolution with a kernel
   (e.g. Kaiser-Bessel, exponential of semicircle).
2. Perform FFT.
3. Crop and deconvolve by dividing the output with the transfer function of the kernel.
4. (Only for recovering $\Lambda$ for Toeplitz operations) Peform inverse FFT.

If we spread the Hermitian $\Phi\Phi^H$ to Cartesian grid via a real and non-negative kernel, there
is `NO REASON` to believe the elements in the gridded operator is `NOT` also Hermitian. My
hypothesis is that the Hermitian symmetry is disrupted by the "decolvultion" in step 3. This is why
we have [modified finufft](https://github.com/mikecjz/finufft/tree/toeplitz) such that we hijack the
nufft process and expose the
[spreading](https://github.com/mikecjz/finufft/blob/5d438feb35925e114ad0cc4535258f92041b0e14/src/cuda/2d/cufinufft2d.cu#L81C5-L81C25)
and the
[deconvolution](https://github.com/mikecjz/finufft/blob/5d438feb35925e114ad0cc4535258f92041b0e14/src/cuda/2d/cufinufft2d.cu#L147C5-L147C28)
API.
