/*
 * MATLAB to CUDA interface to run AhA for MR-Multitasking using cufinufft with float precision.
 * Vectorized nufft call possible. This Function assumes all major data variables to be in single (float)
 * precision and on CPU at input. All multidimensional arrays are assumed to have column-major order 
 * (https://en.wikipedia.org/wiki/Row-_and_column-major_order), same as MATLAB.
 * Author: Junzhou Chen (junzhou@chen.engineer)
 *
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cufinufft.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <time.h>
#include <string.h>

typedef struct {
  int Nx; // Number of first image dimension
  int Ny; // Number of second image dimension
  int Nz; // Number of thrid image dimension
  int L;  // rank
  int Ncoils; // Number of coils
  int Ntrajs; // Number of unique radial spokes
  int Nkx; // Number of k-space samples per spoke
  int M; // Total number of non-uniform samples in one plane
  int Ntrans; // Number of stacked transformations
} dataParamsStruct;

__constant__ dataParamsStruct d_dataParams;





/**
 * @brief Multiplies image data by sensitivity encoding maps element-wise.
 *
 * This CUDA kernel is designed to handle the multiplication of complex-valued 3D (plus rank) image data with
 * sensitivity encoding map (single coil). The multiplication is broadcasted and performed element-wise and the 
 * results are stored in the output array which should be padded to the size of 2*Nx, 2*Ny and initialized to zero before function call.
 * The phase is negated in the result if the coordinate is odd. This ensured that the following FFTs produce centered k-space.
 *
 * @param d_im Pointer to the input device array containing the image data. The array is a flattened
 *             4D array with dimensions [Nx, Ny, Nz, L], where L is the number of image sets or time points.
 * @param d_SE Pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param d_im_SE Pointer to the output device array where the result of the multiplication is stored.
 *                Dimensions [2*Nx, 2*Ny, Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void Im_SE_Pad_AddPhase( cuFloatComplex *d_im,  cuFloatComplex *d_SE,
                             cuFloatComplex *d_im_SE)
{

    // Assign data parameters from constant memory
    int Nx = d_dataParams.Nx;
    int Ny = d_dataParams.Ny;
    int Nz = d_dataParams.Nz;
    int L = d_dataParams.L;

    // Calculate global thread idx
    int threadIdx_global = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int x = threadIdx_global % Nx;
    int y = (threadIdx_global / Nx) % Ny;
    int z = threadIdx_global / (Nx * Ny);

    int x_hat = x + Nx/2; //x hat is the x-coordinate in the padded grid
    int y_hat = y + Ny/2; //y hat is the y-coordinate in the padded grid

    if (x < Nx && y < Ny && z < Nz)
    {
        //Multiply sensitivity map with image.
        //Phase is negated in the result if the coordinate is odd
        for (int l = 0; l < L; l++)
        {
            int im_index =      ((l * Nz + z) * Ny + y) * Nx + x;
            int im_hat_index =  ((l * Nz + z) * Ny*2 + y_hat) * Nx*2 + x_hat;
            int se_index = ((z * Ny + y) * Nx + x);

            float phase = 1-2*((x+y)&1); //1 if even, -1 if odd

            d_im_SE[im_hat_index] =  cuCmulf(d_SE[se_index], d_im[im_index]);

            d_im_SE[im_hat_index].x *= phase;
            d_im_SE[im_hat_index].y *= phase;
        }

    }
}

/**
 * @brief Performs adjoint operation of sensitivity encoding by multiplying image data with conjugate of sensitivity maps.
 *
 * This CUDA kernel performs the adjoint operation of sensitivity encoding by multiplying complex-valued 3D (plus rank) image data 
 * with the complex conjugate of the sensitivity encoding map (single coil). The multiplication is performed element-wise and the
 * results are stored in the output array. This is the adjoint/transpose operation of the forward sensitivity encoding.
 *
 * @param[out] d_im Pointer to the output device array that will store the result. The array is a flattened
 *             4D array with dimensions [Nx, Ny, Nz, L], where L is the number of image sets or time points.
 * @param[in] d_SE Pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param[in] d_im_SE Pointer to the input device array containing the sensitivity-encoded image data.
 *                Dimensions [Nx, Ny, Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void Im_SE_adj( cuFloatComplex *d_im,  cuFloatComplex *d_SE,
                             cuFloatComplex *d_im_SE)
{

    // Assign data parameters from constant memory
    int Nx = d_dataParams.Nx;
    int Ny = d_dataParams.Ny;
    int Nz = d_dataParams.Nz;
    int L = d_dataParams.L;

    // Calculate global thread idx
    int threadIdx_global = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int x = threadIdx_global % Nx;
    int y = (threadIdx_global / Nx) % Ny;
    int z = threadIdx_global / (Nx * Ny);


    if (x < Nx && y < Ny && z < Nz)
    {
        //Multiply sensitivity map with image.
        //Phase is negated in the result if the coordinate is odd
        for (int l = 0; l < L; l++)
        {
            int im_index = ((l * Nz + z) * Ny + y) * Nx + x;
            int se_index = ((z * Ny + y) * Nx + x);

            float phase = 1-2*((x+y)&1); //1 if even, -1 if odd

            d_im[im_index] =  cuCmulf(cuConjf(d_SE[se_index]), d_im_SE[im_index]);

        }

    }
}




/**
 * @brief Crops a padded 4D image array back to its original size.
 *
 * This CUDA kernel extracts the central region from a zero-padded 4D complex-valued image array.
 * The input padded array has dimensions [2*Nx, 2*Ny, Nz, L] and is cropped to [Nx, Ny, Nz, L].
 * The kernel assumes column-major memory layout.
 *
 * @param[in] d_im_padded Pointer to the input device array containing the padded image data.
 *                        Array dimensions are [2*Nx, 2*Ny, Nz, L].
 * @param[out] d_im Pointer to the output device array that will store the cropped image.
 *                  Array dimensions are [Nx, Ny, Nz, L].
 *
 * @note The kernel uses constant memory parameter d_dataParams to access array dimensions:
 *       - Nx: Width of the output image
 *       - Ny: Height of the output image  
 *       - Nz: Depth of both input and output
 *       - L: Number of image sets/time points
 *
 * @note The kernel assumes that all input and output arrays are properly allocated on the GPU
 *       and that the grid size covers the entire volume of the output image (Nx, Ny, Nz).
 */
__global__ void cropImage(cuFloatComplex *d_im_padded, cuFloatComplex *d_im)
{
    // Assign data parameters from constant memory
    int Nx = d_dataParams.Nx;
    int Ny = d_dataParams.Ny;
    int Nz = d_dataParams.Nz;
    int L = d_dataParams.L;

    // Calculate global thread idx
    int threadIdx_global = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int x = threadIdx_global % Nx;
    int y = (threadIdx_global / Nx) % Ny;
    int z = threadIdx_global / (Nx * Ny);

    int x_hat = x + Nx/2; //x hat is the x-coordinate in the padded grid
    int y_hat = y + Ny/2; //y hat is the y-coordinate in the padded grid

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            int im_index =      ((l * Nz + z) * Ny + y) * Nx + x;
            int im_hat_index =  ((l * Nz + z) * Ny*2 + y_hat) * Nx*2 + x_hat;
            d_im[im_index] = d_im_padded[im_hat_index];
        }
    }

}




/**
 * @brief Updates each k-space [1 x L] line with its corresponding (Φ * ΦH) [L x L]. Overall, this amounts to a block-diagnal matrix 
 *        multiplication.
 *
 * @param[in,out] d_k     Pointer to the k-space data being updated. Output will also be stored in this memory. In GPU memory. 
 *                        This is a complex-valued array representing the k-space data. Dimension [M, Ntrans], equivalent to [Ntrajs, Nz, Nkx, L], 
 *                        with Ntrajs being the number of unique radial spokes. This layout is for better memory coalescing.
 * @param[in]     d_Phi2  Pointer to the overall (Φ * ΦH) transformation matrix. In GPU memory. 
 *                        This is a complex-valued array representing the transformation to be applied.
 *                        Dimension [Ntraj, Nz, L, L]. This layout is for better memory coalescing.
 * @param[in]     Ntrajs  The number of trajectories or unique radial spokes in the k-space data.
 * @param[in]     Nkx     The size of the k-space data in the x-dimension.
 * @param[in]     Nz      The size of the k-space data in the z-dimension.
 * @param[in]     L       The number of ranks.
 *
 * 
 * 
 * @note The kernel assumes that the input data (d_k and d_Phi2) are stored in column-major order.
 * 
 * d_k [Ntrajs, Nz, Nkx, L]
 *                       ^innerL
 * 
 * d_Phi2 [Ntraj, Nz, L,       L]
 *                    ^innerL  ^l
 */
__global__ void kspaceUpdateNew(cuFloatComplex *d_k, cuFloatComplex *d_Phi2)
{
    //Assign data parameters from constant memory
    int Ntrajs = d_dataParams.Ntrajs;
    int Nkx = d_dataParams.Nkx;
    int Nz = d_dataParams.Nz;
    int L = d_dataParams.L;

    // Calculate global thread coordinates, blocks will be launched in 1D grid
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int traj = global_thread_idx % Ntrajs;
    int z = (global_thread_idx / Ntrajs) % Nz;
    int kx = global_thread_idx / (Ntrajs * Nz);

    // Declare shared memory pointer at the start of the kernel
    extern __shared__ cuFloatComplex sharedMem[];

    cuFloatComplex *sharedInput = sharedMem + threadIdx.x * L;

    if (traj < Ntrajs && kx < Nkx && z < Nz)
    {
        size_t input_index;
        size_t Phi2_index;
        size_t output_index;
        //Load input data into shared memory
        for (int innerL = 0; innerL < L; innerL++)
        {
            //calculate input index based on column-major order
            input_index = traj + Ntrajs * (z + Nz * (kx + Nkx * innerL));
            sharedInput[innerL] = d_k[input_index];
        }

        //Perform matrix multiplication and assign to d_k
        for (int l = 0; l < L; l++)
        {
            cuFloatComplex tempValue = {0.0f, 0.0f};
            for (int innerL = 0; innerL < L; innerL++)
            {
                Phi2_index = ((l * L + innerL) * Nz + z) * Ntrajs + traj;
                tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], d_Phi2[Phi2_index]));
            }
            output_index = traj + Ntrajs * (z + Nz * (kx + Nkx * l));
            d_k[output_index] = tempValue;
        }


    }
        
}

/**
 * @brief Updates each k-space [1 x L] line with its corresponding (Φ * ΦH) [L x L] (Hermitian, stored in LOWER TRIANGLE including the diagonal). 
 *        Overall, this amounts to a block-diagnal matrix multiplication.
 *
 * @param[in,out] d_k     Pointer to the k-space data being updated. Output will also be stored in this memory. In GPU memory. 
 *                        This is a complex-valued array representing the k-space data from zero-padded image. [2*Nx, 2*Ny, Nz, L], 
 *                        with Ntrajs being the number of unique radial spokes. This layout is for better memory coalescing.
 * @param[in]     d_Phi2  Pointer to the overall (Φ * ΦH) transformation matrix (Toeplitz form), with Hermitian symmetry stored in 
 *                        LOWER TRIANGLE including the diagonal. In GPU memory. 
 *                        This is a complex-valued array representing the transformation to be applied.
 *                        Dimension [2*Nx, 2*Ny, Nz, (L-1)*L/2 +L]. This layout is for better memory coalescing.
 *
 * 
 * 
 * @note The kernel assumes that the input data (d_k and d_Phi2) are stored in column-major order.
 * 
 * d_k [2*Nx, 2*Ny, Nz, L]
 *                      ^innerL
 * 
 * d_Phi2 [2*Nx, 2*Ny, Nz, L,       L] (after explansion from lower triangle)
 *                         ^innerL  ^l
 */
__global__ void kspaceUpdateToeplitzHermitian(cuFloatComplex *d_k, const cuFloatComplex *d_Phi2Cartesian)
{
    //Assign data parameters from constant memory
    int Nx_hat = d_dataParams.Nx * 2; //padded size
    int Ny_hat = d_dataParams.Ny * 2; //padded size
    int Nz = d_dataParams.Nz;
    int L = d_dataParams.L;

    // Calculate global thread coordinates, blocks will be launched in 1D grid
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int x_hat = global_thread_idx % Nx_hat;             //padded x-coordinate
    int y_hat = (global_thread_idx / Nx_hat) % Ny_hat;  //padded y-coordinate
    int z = global_thread_idx / (Nx_hat * Ny_hat);      //z-coordinate

    // Declare shared memory pointer at the start of the kernel
    extern __shared__ cuFloatComplex sharedMem[];

    cuFloatComplex *sharedInput = sharedMem + threadIdx.x * L;

    if (x_hat < Nx_hat && z < Nz && y_hat < Ny_hat)
    {
        size_t input_index;
        size_t Phi2_index;
        size_t output_index;
        //Load input data into shared memory
        for (int innerL = 0; innerL < L; innerL++)
        {
            //calculate input index based on column-major order
            input_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * innerL));
            sharedInput[innerL] = d_k[input_index];
        }

        int matrix_linear_index;

        //Perform matrix multiplication and assign to d_k
        for (int l = 0; l < L; l++)
        {
            cuFloatComplex tempValue = {0.0f, 0.0f};
            for (int innerL = 0; innerL < L; innerL++)
            {
                if (innerL >= l) //lower triangle
                {
                    matrix_linear_index = (2 * L -l + 1)*l/2 + innerL - l;
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], d_Phi2Cartesian[Phi2_index]));
                    
                }
                else //upper triangle
                {
                    matrix_linear_index = (2 * L -innerL + 1)*innerL/2 + l - innerL;
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], cuConjf(d_Phi2Cartesian[Phi2_index])));
                }
                
            }
            output_index = x_hat + Nx_hat * (y_hat + Nz * (z + Ny_hat * l));
            d_k[output_index] = tempValue;
        }


    }
        
}

/**
 * @brief CUDA kernel to permute a 4D array in arbitrary order, assuming column-major input/output.
 * 
 * This kernel efficiently permutes a 4D array according to a specified permutation order.
 * For example, if the input dimensions are [d1, d2, d3, d4] and permutation is [3,1,0,2],
 * the output dimensions will be [d4, d2, d1, d3].
 * 
 * @param input Pointer to the input array (column-major)
 * @param output Pointer to the output array (column-major)
 * @param dims Original dimensions [d1, d2, d3, d4]
 * @param perm Permutation order [p1, p2, p3, p4] where each p is 0-3
 * @param size Total size of the array (product of dimensions)
 */
__global__ void permute4DCUDAKernel(cuFloatComplex* input, cuFloatComplex* output,
                         const int* dims, const int* perm, const int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Calculate input coordinates (column-major)
        int coords[4];
        int temp = idx;
        coords[0] = temp % dims[0];
        temp /= dims[0];
        coords[1] = temp % dims[1];
        temp /= dims[1];
        coords[2] = temp % dims[2];
        coords[3] = temp / dims[2];
        
        // Calculate output index based on permutation (column-major)
        int outIdx = coords[perm[0]];
        int stride = dims[perm[0]];
        
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            outIdx += coords[perm[i]] * stride;
            stride *= dims[perm[i]];
        }
        
        output[outIdx] = input[idx];
    }
}



/**
 * @brief Host function to handle 4D permutation setup and execution of arbitrary permutation order.
 * 
 * This function handles the allocation of device memory, copying of permutation parameters,
 * kernel execution, and cleanup for 4D array permutation operations.
 * 
 * @param d_input Input array on device to be permuted, output will also be stored in this memory
 * @param dims Array of 4 dimensions [d1, d2, d3, d4]
 * @param perm Array of 4 permutation indices [p1, p2, p3, p4]
 */
void permute4D(cuFloatComplex** d_input, const int* dims, const int* perm) {
    const int imageSize = dims[0] * dims[1] * dims[2] * dims[3];

    cuFloatComplex *temp;
    cudaMalloc(&temp, imageSize * sizeof(cuFloatComplex));

    // Allocate and copy dims array to device
    int* d_dims;
    cudaMalloc(&d_dims, 4 * sizeof(int));
    cudaMemcpy(d_dims, dims, 4 * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate and copy perm array to device
    int* d_perm;
    cudaMalloc(&d_perm, 4 * sizeof(int));
    cudaMemcpy(d_perm, perm, 4 * sizeof(int), cudaMemcpyHostToDevice);

    // Setup and execute kernel
    const int permutationBlockSize = 256;
    const int permutationGridSize = (imageSize + permutationBlockSize - 1) / permutationBlockSize;
    // clock_t start = clock();
    permute4DCUDAKernel<<<permutationGridSize, permutationBlockSize>>>(*d_input, temp, d_dims, d_perm, imageSize);
    cudaDeviceSynchronize();
    // clock_t end = clock();
    // double time_spent = (double)(end - start)/CLOCKS_PER_SEC;
    // mexPrintf("permute4D execution time: %f seconds\n", time_spent);

    cudaFree(*d_input);
    *d_input = temp; // Now this properly updates the pointer in the calling function
    temp = nullptr;

    // Cleanup
    cudaFree(d_dims);
    cudaFree(d_perm);

    return;
}


/**
 * @brief CUDA kernel that performs element-wise addition of two complex arrays.
 *
 * This kernel adds corresponding elements from array b into array a, storing the result in array a.
 * AKA  a = a + b
 * The operation is performed in-place on array a. The kernel uses CUDA's complex arithmetic functions
 * to handle complex number addition.
 *
 * @param[in,out] a Pointer to the first complex array, which will also store the result
 * @param[in] b Pointer to the second complex array to be added
 * @param[in] arraySize The total number of elements in each array
 *
 * @note The kernel assumes both input arrays have the same size and are properly allocated in GPU memory.
 *       The arrays should contain complex numbers in CUDA's cuFloatComplex format.
 *
 * @see cuCaddf
 */
__global__ void addArrays(cuFloatComplex *a, cuFloatComplex *b, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize) {
        a[index] = cuCaddf(a[index], b[index]);
    }
}

/**
 * @brief Performs the AhA operation for MR-Multitasking subspace reconstruction with stack-of-stars trajectory
 * 
 * This function implements the core AhA operation for a single coil, which consists of:
 * 1. Multiplying input image with sensitivity map (SEs * im)
 * 2. Forward FFT in Z dimension 
 * 3. Type 1 NUFFT to transform from uniform to non-uniform k-space
 * 4. Block-diagonal k-space update with Phi2
 * 5. Type 2 NUFFT to transform back to uniform space
 * 6. Inverse FFT in Z dimension
 * 7. Conjugate multiplication with sensitivity map
 *
 * @param[in] d_im Input image on GPU device memory [Nx, Ny, Nz, L]
 * @param[in] d_SE Single coil sensitivity map on GPU [Nx, Ny, Nz]
 * @param[in] d_Phi2 Block-diagonal update matrix on GPU [Ntrajs, Nz, L, L]
 * @param[in] d_k Temporary k-space storage on GPU [M, Ntrans]
 * @param[out] d_im_out Output image on GPU [Nx, Ny, Nz, L]
 * @param[in] cufft_plan cuFFT plan for Z dimension transforms
 * @param[in] cufinufft_type1_plan Plan for Type 1 NUFFT (uniform to non-uniform)
 * @param[in] cufinufft_type2_plan Plan for Type 2 NUFFT (non-uniform to uniform)
 * @param[in] dataParams Struct containing data dimensions and parameters
 * @param[in] verbose Whether to print execution times
 * @note All arrays are assumed to be in column-major order and complexsingle precision
 * @note The function uses constant memory for data parameters via d_dataParams
 */

void AhA_by_coil(cuFloatComplex *d_im,//input image
                cuFloatComplex *d_SE, //single coil map
                const cuFloatComplex *d_Phi2Cartesian, //(Φ * ΦH)
                cuFloatComplex *d_im_out, //output image
                cufftHandle cufft_plan,
                cufinufftf_plan cufinufft_type1_plan,
                cufinufftf_plan cufinufft_type2_plan,
                dataParamsStruct dataParams,
                bool verbose)
{
    // ==========Broadcast SE to image input on GPU. (SEs * im)========== //

    clock_t start, AhA_start;
    clock_t end, AhA_end;
    double time_spent;


    AhA_start = clock();

    // Assign data parameters from dataParams struct
    int Nx = dataParams.Nx;
    int Ny = dataParams.Ny;
    int Nz = dataParams.Nz;
    int L = dataParams.L;
    int Ntrajs = dataParams.Ntrajs;
    int Nkx = dataParams.Nkx;
    int M = dataParams.M;
    int Ntrans = dataParams.Ntrans;

    const int imageSize = Nx * Ny * Nz * L;

    // Intialze Blocks and Grids and Perform Broadcasting
    int blockSize = 256;
    int gridSize = (imageSize*4 + 255) / 256;

    cuFloatComplex *d_im_SE;
    cudaMalloc(&d_im_SE, imageSize * 4 * sizeof(cuFloatComplex));
    cudaMemset(d_im_SE, 0, imageSize * 4 * sizeof(cuFloatComplex));

    start = clock();
    Im_SE_Pad_AddPhase<<<gridSize, blockSize>>>(d_im, d_SE, d_im_SE);
    cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("Im_SE_Pad_AddPhase execution time: %f seconds\n", time_spent);

    // ========== Batched Forward 3D cuFFT ========== /


    //Execute forward cuFFT
    start = clock();
    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_FORWARD);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("cufftExecC2C execution time: %f seconds\n", time_spent);



    // =========Block-Diagonal kspace update ========/


    const int blockSizeBlkdiag = 256;
    const int gridSizeBlkdiag = (4*imageSize + 255) / 256;
    start = clock();
    //d_im_SE is in kspace due to precious cuFFT.
    kspaceUpdateToeplitzHermitian<<<gridSizeBlkdiag, blockSizeBlkdiag, blockSizeBlkdiag * L * sizeof(cuFloatComplex)>>>(d_im_SE, d_Phi2Cartesian);
    cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("kspaceUpdateToeplitzHermitian execution time: %f seconds\n", time_spent);

    // ========== Inverse cuFFT in the 3rd (Z) dimension ========== /

    //Execute
    start = clock();

    //d_im_SE is now in image space.
    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_INVERSE);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("cufftExecC2C execution time: %f seconds\n", time_spent);


    // ========= FINUFFT Type 1 deconvolution ONLY========/

    cuFloatComplex *d_im_SE_deconv;
    cudaMalloc(&d_im_SE_deconv, imageSize * sizeof(cuFloatComplex));


    start = clock();
    // cufinufftf_deconvolve(cufinufft_type1_plan, d_im_SE_deconv, d_im_SE);
    blockSize = 256;
    gridSize = (imageSize + 255) / 256;
    cropImage<<<gridSize, blockSize>>>(d_im_SE, d_im_SE_deconv);
    cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("type1 cufinufftf_deconvolve execution time: %f seconds\n", time_spent);

    cudaFree(d_im_SE); //no longer used


    // ========= coil_combination ========/
    start = clock();

    //d_im_SE_deconv is in image domain due to precious inverse cuFFT.
    blockSize = 256;
    gridSize = (imageSize + 255) / 256;
    Im_SE_adj<<<gridSize, blockSize>>>(d_im_out, d_SE, d_im_SE_deconv);
    cudaDeviceSynchronize();    
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("SEs_times_Im_adj execution time: %f seconds\n", time_spent);

    cudaFree(d_im_SE_deconv); //Clean up


    AhA_end = clock();
    time_spent = (double)(AhA_end - AhA_start) / CLOCKS_PER_SEC;
    if (verbose) mexPrintf("AhA_by_coil execution time: %f seconds\n", time_spent);
}


/**
 * @brief Main functioin for the MR-Multitasking subspace recon AhA operations for stack-of-stars trajectory
 * 
 * Overall, it does the following:
 * 1. Initialize necessary parameters, data array and does rudimentray data checking.
 * 2. copy data from host (CPU) to GPU device.
 * 3. Perform AhA calculations.
 * 4. Transfer calculated data from GPU device back to host. Then return to matlab workspace.
 * 
 * All major data arrays are assumed to be in single precision and stored in column-major order on CPU.
 * 
 * @param x prhs[0]. Spatial basis function image input of size [Nx, Ny, Nz, L]. 
 * @param st prhs[1]. Matlab structure containing necessary data parameters incluing in-plane matrix size N, Number of NU points M (Ntrajs * Nro)
 *           and actual trajectoris points om [M, 2] for kx and ky.
 * @param SEs prhs[2]. Sensitivity maps of dimension [Nx, Ny, Nz, Ncoils].
 * @param Phi2 prhs[3]. The overall (Φ * ΦH) matrix. Dimension [L, L, Ntraj, Nz].
 * 
 * @param AhAx plhs[0]. The flatenned output of the operation AhA(x). Dimension [(Nx * Ny * Nz * L), 1]
 * 
 * @note AhAx = AhA_ps_cufinufft(x, st, SEs, Phi2)
 *
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    clock_t start, end;
    double time_spent;

    /* Declare all variables.*/
    float *x;              // om(1)
    float *y;              // om(2)
    mxComplexSingle *im;   // Pointer to complex image-domain data (x), on host [Nx, Ny, Nz, L]
    mxComplexSingle *SEs;  // Pointer to complex Complex sensitivity maps (SEs), on host [Nx, Ny, Nz, Ncoils]
    const mxGPUArray *Phi2Cartesian;       // Pointer to complex Phi2 data (Phi2, aka, Om), on host [2*Nx, 2*Ny, Nz, num_lower_coordinates]
    int isSign = 1;        // Whether to use + sign in complex exponential. Default to true
    float eps = 1e-6;      // NUFFT precision, default 1e-6;
    int64_t Nx;
    int64_t Ny;
    int64_t Nz;
    int64_t Ncoils;
    int64_t L;          // rank
    int64_t Nkx;        // length of one k-space line // LeeHL
    size_t M;           // number of NU points
    size_t Ntrans;      // Number of stacked transforms !!!!for each coil!!!!
    size_t Ntrajs;      // Number of in-plane tranjectories. e.g. 280
    dataParamsStruct dataParams;
    //int GPUDeviceIndex; // Which GPU to use
    // ==========Initialize the MathWorks GPU API. ========== //
    mxInitGPU();

    char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const *const errMsg = "Invalid input to MEX file.";
    char const *const instruction = "Usage:\n AhAx = AhA_ps_cufinufft(x, st, SEs, Phi2).";

    char errorMessage[1024]; // remember to ensure this buffer is large enough for your message
    // Check input number
    if (nrhs != 4)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Exactly 4 inputs are required. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    // Check input type
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(mxGetField(prhs[1], 0, "om")) || !mxIsSingle(prhs[2]))
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), NU trajectories (st.om), Sensitivity maps (SEs) and Phi2 must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    // ==========Assign Pointers and values ========== //

    // Input 1: image domain data
    im = mxGetComplexSingles(prhs[0]); // Get input image domain data



    // Input 2: Trajectories and Image Dimensions
    // Separate om(:,1) and om(:,2)
    size_t numRows = mxGetM(mxGetField(prhs[1], 0, "om"));
    float *om = mxGetSingles(mxGetField(prhs[1], 0, "om"));
    x = (float *)malloc(numRows * sizeof(float));
    y = (float *)malloc(numRows * sizeof(float));
    for (size_t i = 0; i < numRows; i++)
    {
        x[i] = om[i];
        y[i] = om[numRows + i];
    }

    Nx = (int64_t)mxGetDoubles(mxGetField(prhs[1], 0, "Nd"))[0];
    Ny = (int64_t)mxGetDoubles(mxGetField(prhs[1], 0, "Nd"))[1];
    Nz = (int64_t)mxGetScalar(mxGetField(prhs[1], 0, "Nz"));
    M = (size_t)mxGetScalar(mxGetField(prhs[1], 0, "M"));
    L = (int64_t)mxGetScalar(mxGetField(prhs[1], 0, "L"));
    // Get k-space size from Phi2 and st.traj_bart  
    mxArray *traj_bart = mxGetField(prhs[1], 0, "traj_bart");
    const mwSize *traj_bart_dims = mxGetDimensions(traj_bart);
    Ntrajs = (size_t)traj_bart_dims[1];
    Nkx = (int64_t)traj_bart_dims[2];

    Ntrans = Nz * L; //Number of stacked transforms for each coil
    

    // Input 3: Sensitivity Maps
    const mwSize *coilDims = mxGetDimensions(prhs[2]);
    Ncoils = (int64_t)coilDims[3];
    SEs = mxGetComplexSingles(prhs[2]);

    // Input 4: Phi2 (aka, Om)
    // Handle Phi2 after MathWorks GPU API initialization
    Phi2Cartesian = mxGPUCreateFromMxArray(prhs[3]);
    //Phi2 will be handled after MathWorks GPU API initialization
    const mwSize *phi2Dims = mxGPUGetDimensions(Phi2Cartesian);

    

    // Assign data parameters to struct
    dataParams.Nx = Nx;
    dataParams.Ny = Ny;
    dataParams.Nz = Nz;
    dataParams.L = L;
    dataParams.Ncoils = Ncoils;
    dataParams.Ntrajs = Ntrajs;
    dataParams.Nkx = Nkx;
    dataParams.M = M;
    dataParams.Ntrans = Ntrans;


    // Create output gpuArray

    mxGPUArray *AhAx;
    mwSize dims[4] = {Nx, Ny, Nz, L};
    AhAx = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

    // Assign data parameters to device constant memory
    cudaMemcpyToSymbol(d_dataParams, &dataParams, sizeof(dataParamsStruct));

    
    cuFloatComplex *d_im, *d_SEs; // input image and SEs on device
    cuFloatComplex *d_im_out;    // output image from AhA_by_coil on devices
    cuFloatComplex *d_AhAx;     // output image from AhA from all coils on device

    float *d_x, *d_y;               // NU points on device
    const cuFloatComplex *d_Phi2Cartesian;   //Phi2 on device

    // Allocate device memories

    //images related
    cudaMalloc(&d_im, Nx * Ny * Nz * L * sizeof(cuFloatComplex));
    cudaMalloc(&d_SEs, Nx * Ny * Nz * Ncoils * sizeof(cuFloatComplex));
    cudaMalloc(&d_im_out, Nx * Ny * Nz * L * sizeof(cuFloatComplex));
    cudaMalloc(&d_AhAx, Nx * Ny * Nz * L * sizeof(cuFloatComplex));

    //k-space related
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));



    // Copy data to device
    //images related
    cudaMemcpy(d_im, im, Nx * Ny * Nz * L * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SEs, SEs, Nx * Ny * Nz * Ncoils * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemset(d_AhAx, 0, Nx * Ny * Nz * L * sizeof(cuFloatComplex));

    //kspace related
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    d_Phi2Cartesian = (cuFloatComplex *)mxGPUGetDataReadOnly(Phi2Cartesian);


    cuFloatComplex *d_SE; //Pointer to current coil map's starting address (SINGLE COIL)

    //make cuFFT and cufinufft plans
    cufftHandle cufft_plan;
    int n[3] = {Nz, Nx*2, Ny*2};
    cufftPlanMany(&cufft_plan, 3, n, NULL, NULL, 4*Nx*Ny*Nz, NULL, NULL, 4*Nx*Ny*L, CUFFT_C2C, L);

    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    int64_t modes[2] = {Nx, Ny};

    cufinufftf_plan cufinufft_type1_plan;
    cufinufftf_makeplan(1, 2, modes, 1, Ntrans, 1e-6, &cufinufft_type1_plan, &opts);// make type 1 plan, deafault options
    cufinufftf_setpts(cufinufft_type1_plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    cufinufftf_plan cufinufft_type2_plan;
    cufinufftf_makeplan(2, 2, modes, -1, Ntrans, 1e-6, &cufinufft_type2_plan, &opts);// make type 2 plan, deafault options
    cufinufftf_setpts(cufinufft_type2_plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    start = clock();
    //loop through each coil

    bool verbose = true; 
    for (int coil = 0; coil < Ncoils; coil++)
    {
        //Calculate pointer to current coil map's starting address
        d_SE = d_SEs + coil * Nx * Ny * Nz;

        AhA_by_coil(d_im, d_SE, d_Phi2Cartesian, d_im_out, cufft_plan, cufinufft_type1_plan, cufinufft_type2_plan, dataParams, verbose);
          

        //Add the output from each coil to d_AhAx
        int blockSize = 256;
        int gridSize = (Nx * Ny * Nz * L + 255) / 256;
        clock_t start_add = clock();
        addArrays<<<gridSize, blockSize>>>(d_AhAx, d_im_out, Nx * Ny * Nz * L);
        cudaDeviceSynchronize();
        clock_t end_add = clock();
        time_spent = (double)(end_add - start_add) / CLOCKS_PER_SEC;
        if (verbose) mexPrintf("addArrays execution time: %f seconds\n", time_spent);

        verbose = false;  
    }
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    mexPrintf("AhA actual execution time: %f seconds\n", time_spent);

    
    // ========= Post-operation clean-up========/

    cuFloatComplex *d_AhAx_out = (cuFloatComplex *)mxGPUGetData(AhAx);
    cudaMemcpy(d_AhAx_out, d_AhAx, Nx * Ny * Nz * L * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);


    plhs[0] = mxGPUCreateMxArrayOnGPU(AhAx);
    
    // Free device memories
    cudaFree(d_im);
    cudaFree(d_AhAx);
    cudaFree(d_SEs);
    cudaFree(d_im_out);

    cudaFree(d_x);
    cudaFree(d_y);

    //destroy cufft and cufinufft plans
    cufftDestroy(cufft_plan);
    cufinufftf_destroy(cufinufft_type1_plan);
    cufinufftf_destroy(cufinufft_type2_plan);

   
    mxGPUDestroyGPUArray(AhAx);
    mxGPUDestroyGPUArray(Phi2Cartesian);

    // Free host allocated memory
    free(x);
    free(y);

    return;
}

