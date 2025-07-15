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
#include <cublas_v2.h>
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
 * @param[in] d_im Readonly pointer to the input device array containing the image data. The array is a flattened
 *             4D array with dimensions [Nx, Ny, Nz, L], where L is the number of image sets or time points.
 * @param[in] d_SE Readonly pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param[out] d_im_SE Pointer to the output device array where the result of the multiplication is stored.
 *                Dimensions [2*Nx, 2*Ny, Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void Im_SE_Pad_shift_AddPhase( const cuFloatComplex *d_im,  const cuFloatComplex *d_SE,
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

    x_hat = (x_hat + Nx) % (2 * Nx); //fftshift-ed coordinate
    y_hat = (y_hat + Ny) % (2 * Ny); //fftshift-ed coordinate

    if (x < Nx && y < Ny && z < Nz)
    {

        int se_index = ((z * Ny + y) * Nx + x);
        cuFloatComplex se_val = d_SE[se_index];

        //Multiply sensitivity map with image.
        //Phase is negated in the result if the coordinate is odd
        for (int l = 0; l < L; l++)
        {
            int im_index =      ((l * Nz + z) * Ny + y) * Nx + x;
            int im_hat_index =  ((l * Nz + z) * Ny*2 + y_hat) * Nx*2 + x_hat;

            float phase = 1-2*((x+y)&1); //1 if even, -1 if odd

            d_im_SE[im_hat_index] =  cuCmulf(se_val, d_im[im_index]);

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
 * @param[in] d_SE Readonly pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param[in] d_im_SE Pointer to the input device array containing the sensitivity-encoded image data.
 *                Dimensions [Nx, Ny, Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void Im_SE_adj( cuFloatComplex *d_im,  const cuFloatComplex *d_SE,
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

        int se_index = ((z * Ny + y) * Nx + x);
        cuFloatComplex se_val = d_SE[se_index];
        
        //Multiply sensitivity map with image.
        //Phase is negated in the result if the coordinate is odd
        for (int l = 0; l < L; l++)
        {
            int im_index = ((l * Nz + z) * Ny + y) * Nx + x;
            

            float phase = 1-2*((x+y)&1); //1 if even, -1 if odd

            d_im[im_index] =  cuCmulf(cuConjf(se_val), d_im_SE[im_index]);

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
                    matrix_linear_index = (2 * L -l + 1)*l/2 + innerL - l; //Sum of arithmetic series
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], d_Phi2Cartesian[Phi2_index]));
                    
                }
                else //upper triangle
                {
                    matrix_linear_index = (2 * L -innerL + 1)*innerL/2 + l - innerL; //Sum of arithmetic series, indices swapped
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], cuConjf(d_Phi2Cartesian[Phi2_index])));
                }
                
            }
            output_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z + Nz * l));
            d_k[output_index] = tempValue;
        }


    }
        
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
 * @brief Validates input arguments for the AhA_ps_cufinufft MEX function
 *
 * This function performs validation checks on the input arguments passed to the MEX function:
 * 1. Verifies that exactly 4 input arguments are provided
 * 2. Checks that inputs x (image), SEs (sensitivity maps), and Phi2 are GPU arrays
 * 3. Validates that these GPU arrays contain valid GPU data
 * 4. Ensures that the non-uniform trajectories (st.om) are in single precision
 *
 * @param[in] nrhs Number of right-hand side (input) arguments
 * @param[in] prhs Array of pointers to the input arguments:
 *                 - prhs[0]: x - Image data (must be gpuArray)
 *                 - prhs[1]: st - Structure containing trajectories (st.om must be single)
 *                 - prhs[2]: SEs - Sensitivity maps (must be gpuArray)
 *                 - prhs[3]: Phi2 - Transformation matrix (must be gpuArray)
 *
 * @throws AhA_ps_cufinufft:InputError if any validation check fails, with a descriptive error message
 *
 * @note Uses mexErrMsgIdAndTxt() to throw errors, which will terminate MEX function execution
 * @note Error messages include usage instructions for proper function calling
 * @note Data types and complexities for the input gpuArrays are checked in checkGPUArrays()
 */
void checkMexInputs(int nrhs, mxArray const *prhs[])
{
    char const *const instruction = "Usage:\n AhAx = AhA_ps_cufinufft(x, st, SEs, Phi2).";

    char errorMessage[1024]; // remember to ensure this buffer is large enough for your message
    // Check input number
    if (nrhs != 4)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Exactly 4 inputs are required. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    //check if certain inputs are gpuArray
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[2]) || !mxIsGPUArray(prhs[3]))
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be gpuArray and are complex singles. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    //check if certain inputs are valid gpuArrays
    if (!mxGPUIsValidGPUData(prhs[0]) || !mxGPUIsValidGPUData(prhs[2]) || !mxGPUIsValidGPUData(prhs[3]))
    {
        snprintf(errorMessage, sizeof(errorMessage), 
                "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be valid gpuArrays. \n"
                "If you see this message, gpuArray might be cleared by gpuDevice(). \n"
                "%s", 
                instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    // Check NU trajectories type
    if (!mxIsSingle(mxGetField(prhs[1], 0, "om")))
    {
        snprintf(errorMessage, sizeof(errorMessage), "NU trajectories (st.om) must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }
}


/**
 * @brief Validates data types and complexity of GPU arrays used in AhA_ps_cufinufft
 *
 * This function performs validation checks on three GPU arrays to ensure they meet the
 * requirements for the AhA operation:
 * 1. Verifies that all arrays are in single precision (float)
 * 2. Verifies that all arrays contain complex values
 *
 * @param[in] Phi2Cartesian GPU array containing the transformation matrix
 * @param[in] im GPU array containing the input image data
 * @param[in] SEs GPU array containing the sensitivity maps
 *
 * @throws AhA_ps_cufinufft:InputError if any array has incorrect precision or is not complex,
 *         with a descriptive error message including usage instructions
 *
 * @note Uses mexErrMsgIdAndTxt() to throw errors, which will terminate MEX function execution
 * @note Error messages include proper usage instructions for the main function
 * @note The error message buffer is sized to 1024 characters to accommodate detailed messages
 */
void checkGPUArrays(const mxGPUArray *Phi2Cartesian, const mxGPUArray *im, const mxGPUArray *SEs)
{

    char const *const instruction = "Usage:\n AhAx = AhA_ps_cufinufft(x, st, SEs, Phi2).";

    char errorMessage[1024]; // remember to ensure this buffer is large enough for your message
    // Check if gpuArrays are singles
    if (mxGPUGetClassID(Phi2Cartesian) != mxSINGLE_CLASS || mxGPUGetClassID(im) != mxSINGLE_CLASS || mxGPUGetClassID(SEs) != mxSINGLE_CLASS)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be singles (and complex too). \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    // Check if gpuArrays are complex
    if (mxGPUGetComplexity(Phi2Cartesian) != mxCOMPLEX || mxGPUGetComplexity(im) != mxCOMPLEX || mxGPUGetComplexity(SEs) != mxCOMPLEX)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be complex (and singles too). \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }
}

/**
 * @brief Performs the AhA operation for MR-Multitasking subspace reconstruction with stack-of-stars trajectory
 * 
 * This function implements the core AhA operation for a single coil, which consists of:
 * 1. Multiplying input image with sensitivity map (SEs * im)
 * 2. Forward 3D cuFFT
 * 3. Type 1 NUFFT to transform from uniform to non-uniform k-space
 * 4. Block-diagonal k-space update with Phi2
 * 5. Type 2 NUFFT to transform back to uniform space
 * 6. Inverse 3D cuFFT
 * 7. Conjugate multiplication with sensitivity map
 *
 * @param[in] d_im Input image on GPU device memory [Nx, Ny, Nz, L]
 * @param[in] d_SE Single coil sensitivity map on GPU [Nx, Ny, Nz]
 * @param[in] d_Phi2 Block-diagonal update matrix on GPU [Ntrajs, Nz, L, L]
 * @param[in] d_k Temporary k-space storage on GPU [M, Ntrans]
 * @param[out] d_im_out Output image on GPU [Nx, Ny, Nz, L]
 * @param[in] cufft_plan cuFFT plan for Z dimension transforms
 * @param[in] cufinufft_type1_plan Plan for Type 1 NUFFT (non-uniform to uniform)
 * @param[in] cufinufft_type2_plan Plan for Type 2 NUFFT (uniform to non-uniform)
 * @param[in] dataParams Struct containing data dimensions and parameters
 * @param[in] verbose Whether to print execution times
 * @note All arrays are assumed to be in column-major order and complexsingle precision
 * @note The function uses constant memory for data parameters via d_dataParams
 */

void AhA_by_coil(const cuFloatComplex *d_im,//input image
                const cuFloatComplex *d_SE, //single coil map
                const cuFloatComplex *d_Phi2Cartesian, //(Φ * ΦH)
                cuFloatComplex *d_im_out, //output image
                cufftHandle cufft_plan,
                cufinufftf_plan cufinufft_type1_plan,
                cufinufftf_plan cufinufft_type2_plan,
                dataParamsStruct dataParams,
                bool is_verbose)
{
    // Create CUDA events
    cudaEvent_t function_start, function_stop, start, stop;
    cudaEventCreate(&function_start);
    cudaEventCreate(&function_stop);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float seconds = 0;

    // Overall timing
    cudaEventRecord(function_start);

    // ==========Broadcast SE to image input on GPU. (SEs * im)========== //

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
    int gridSize = (imageSize + 255) / 256;

    cuFloatComplex *d_im_SE;
    cudaMalloc(&d_im_SE, imageSize * 4 * sizeof(cuFloatComplex));
    cudaMemset(d_im_SE, 0, imageSize * 4 * sizeof(cuFloatComplex));

    // Example of timing a specific kernel:
    cudaEventRecord(start);
    Im_SE_Pad_shift_AddPhase<<<gridSize, blockSize>>>(d_im, d_SE, d_im_SE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("Im_SE_Pad_shift_AddPhase execution time: %.6f seconds\n", seconds);

    // ========== Batched Forward 3D cuFFT ========== /


    //Execute forward cuFFT
    cudaEventRecord(start);
    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("cufftExecC2C execution time: %.6f seconds\n", seconds);



    // =========Block-Diagonal kspace update ========/

    int blockSizeBlkdiag;

    // To make sure shared memory is within limits
    if (L < 24 ){
        blockSizeBlkdiag = 256;
    }else if (L < 48){
        blockSizeBlkdiag = 128;
    }else if (L < 96){
        blockSizeBlkdiag = 64;
    }

    const int gridSizeBlkdiag = (4*imageSize + blockSizeBlkdiag -1 ) / blockSizeBlkdiag;
    const int sharedMemSizeBlkdiag = blockSizeBlkdiag * L * sizeof(cuFloatComplex);
    cudaEventRecord(start);
    //d_im_SE is in kspace due to previous cuFFT.
    kspaceUpdateToeplitzHermitian<<<gridSizeBlkdiag, blockSizeBlkdiag, sharedMemSizeBlkdiag>>>(d_im_SE, d_Phi2Cartesian);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("kspaceUpdateToeplitzHermitian execution time: %.6f seconds\n", seconds);

    // ========== Batched Inverse 3D cuFFT========== /

    //Execute
    cudaEventRecord(start);

    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_INVERSE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("cufftExecC2C execution time: %.6f seconds\n", seconds);


    // ========= FINUFFT Type 1 deconvolution ONLY========/

    cuFloatComplex *d_im_SE_deconv;
    cudaMalloc(&d_im_SE_deconv, imageSize * sizeof(cuFloatComplex));

    //d_im_SE is now in image space.
    cudaEventRecord(start);
    cufinufftf_deconvolve(cufinufft_type1_plan, d_im_SE_deconv, d_im_SE);
    
    // blockSize = 256;
    // gridSize = (imageSize + 255) / 256;
    // shiftAndCropImage<<<gridSize, blockSize>>>(d_im_SE, d_im_SE_deconv);
    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("type1 cufinufftf_deconvolve execution time: %.6f seconds\n", seconds);

    cudaFree(d_im_SE); //no longer used


    // ========= coil_combination ========/
    cudaEventRecord(start);

    //d_im_SE_deconv is in image domain due to precious inverse cuFFT.
    blockSize = 256;
    gridSize = (imageSize + 255) / 256;
    Im_SE_adj<<<gridSize, blockSize>>>(d_im_out, d_SE, d_im_SE_deconv);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("SEs_times_Im_adj execution time: %.6f seconds\n", seconds);

    cudaFree(d_im_SE_deconv); //Clean up


    // Overall timing end
    cudaEventRecord(function_stop);
    cudaEventSynchronize(function_stop);
    cudaEventElapsedTime(&milliseconds, function_start, function_stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("AhA_by_coil total execution time: %.6f seconds\n", seconds);

    cudaEventDestroy(function_start);
    cudaEventDestroy(function_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
    float *x;             // om(1)
    float *y;             // om(2)
    const mxGPUArray *mxGPUArray_im;               // Pointer to mxGPUArray object that holds complex image-domain data (x), on device (GPU) [Nx, Ny, Nz, L]
    const mxGPUArray *mxGPUArray_SEs;              // Pointer to mxGPUArray object that holds complex Complex sensitivity maps (SEs), on device (GPU) [Nx, Ny, Nz, Ncoils]
    const mxGPUArray *mxGPUArray_Phi2Cartesian;    // Pointer to mxGPUArray object that holds complex Phi2 data (Phi2, aka, Om), on device (GPU) [2*Nx, 2*Ny, Nz, num_lower_coordinates]

    int isSign = 1;      // Whether to use + sign in complex exponential. Default to true
    float eps = 1e-6;    // NUFFT precision, default 1e-6;
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

    // Check inputs
    checkMexInputs(nrhs, prhs);

    // ==========Assign Pointers and values ========== //

    // Input 1: image domain data
    mxGPUArray_im = mxGPUCreateFromMxArray(prhs[0]); // Get input image domain data



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

    // Make sure L is not too large
    if (L >= 96){
        mexErrMsgIdAndTxt("AhA_cufinufft_toep:InputError", "L must be less than 96 (< 96)");
    }

    // Get k-space size from Phi2 and st.traj_bart  
    mxArray *traj_bart = mxGetField(prhs[1], 0, "traj_bart");
    const mwSize *traj_bart_dims = mxGetDimensions(traj_bart);
    Ntrajs = (size_t)traj_bart_dims[1];
    Nkx = (int64_t)traj_bart_dims[2];

    Ntrans = Nz * L; //Number of stacked transforms for each coil
    

    // Input 3: Sensitivity Maps
    mxGPUArray_SEs = mxGPUCreateFromMxArray(prhs[2]);
    const mwSize *coilDims = mxGPUGetDimensions(mxGPUArray_SEs);
    Ncoils = (int64_t)coilDims[3];

    // Input 4: Phi2 (aka, Om)
    // Handle Phi2 after MathWorks GPU API initialization
    mxGPUArray_Phi2Cartesian = mxGPUCreateFromMxArray(prhs[3]);
    //Phi2 will be handled after MathWorks GPU API initialization
    const mwSize *phi2Dims = mxGPUGetDimensions(mxGPUArray_Phi2Cartesian);


    // Check gpuArrays
    checkGPUArrays(mxGPUArray_Phi2Cartesian, mxGPUArray_im, mxGPUArray_SEs);

    
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

    mxGPUArray *mxGPUArray_AhAx;
    cuFloatComplex *d_AhAx;     // output image from AhA from all coils on device
    mwSize dims[1] = {(mwSize)(Nx * Ny * Nz * L)};
    mxGPUArray_AhAx = mxGPUCreateGPUArray(1, dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    d_AhAx = (cuFloatComplex *)mxGPUGetData(mxGPUArray_AhAx);

    // Assign data parameters to device constant memory
    cudaMemcpyToSymbol(d_dataParams, &dataParams, sizeof(dataParamsStruct));

    
    // declare device pointers
    const cuFloatComplex *d_im, *d_SEs; // input image and SEs on device
    cuFloatComplex *d_im_out;    // output image from AhA_by_coil on devices
    float *d_x, *d_y;               // NU points on device
    const cuFloatComplex *d_Phi2Cartesian;   //Phi2 on device


    // Allocate device memories
    // images related
    cudaMalloc(&d_im_out, Nx * Ny * Nz * L * sizeof(cuFloatComplex));
    // k-space related
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));



    // process input data
    // images related
    d_im = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_im);
    d_SEs = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_SEs);

    //kspace related
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    d_Phi2Cartesian = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_Phi2Cartesian);

    

    // make cuFFT and cufinufft plans
    cufftHandle cufft_plan;
    int n[3] = {(int)Nz, (int)(Nx*2), (int)(Ny*2)};
    cufftPlanMany(&cufft_plan, 3, n, NULL, NULL, 4*Nx*Ny*Nz, NULL, NULL, 4*Nx*Ny*Nz, CUFFT_C2C, L);

    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac = 2.0;
    int64_t modes[2] = {Nx, Ny};

    cufinufftf_plan cufinufft_type1_plan;
    cufinufftf_makeplan(1, 2, modes, 1, Ntrans, 1e-6, &cufinufft_type1_plan, &opts);// make type 1 plan, deafault options
    cufinufftf_setpts(cufinufft_type1_plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    cufinufftf_plan cufinufft_type2_plan;
    cufinufftf_makeplan(2, 2, modes, -1, Ntrans, 1e-6, &cufinufft_type2_plan, &opts);// make type 2 plan, deafault options
    cufinufftf_setpts(cufinufft_type2_plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    //start = clock();
    //loop through each coil

    cublasHandle_t handle;
    cublasCreate(&handle);
    cuFloatComplex scale_factor_fft = make_cuFloatComplex(1.0f/(Nx*Ny*Nz), 0.0f); //FFT scaling factor

    cudaEvent_t start_add, stop_add;
    cudaEventCreate(&start_add);
    cudaEventCreate(&stop_add);

    const cuFloatComplex *d_SE; //Pointer to current coil map's starting address (SINGLE COIL)

    bool is_verbose = false; 
    for (int coil = 0; coil < Ncoils; coil++)
    {
        //Calculate pointer to current coil map's starting address
        d_SE = d_SEs + coil * Nx * Ny * Nz;

        AhA_by_coil(d_im, d_SE, d_Phi2Cartesian, d_im_out, cufft_plan, cufinufft_type1_plan, cufinufft_type2_plan, dataParams, is_verbose);
          

        //Add the output from each coil to d_AhAx and scale by FFT scaling factor

        float milliseconds = 0;
        
        cudaEventRecord(start_add);
        cublasCaxpy(handle, Nx * Ny * Nz * L, &scale_factor_fft, d_im_out, 1, d_AhAx, 1);
        cudaEventRecord(stop_add);
        cudaEventSynchronize(stop_add);
        
        cudaEventElapsedTime(&milliseconds, start_add, stop_add);
        float seconds = milliseconds / 1000.0f;
        if (is_verbose) mexPrintf("cublasCaxpy execution time: %.6f seconds\n", seconds);
        


        is_verbose = false;  
    }
    // end = clock();
    // time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    // mexPrintf("AhA actual execution time: %f seconds\n", time_spent);

    cudaEventDestroy(start_add);
    cudaEventDestroy(stop_add);

    
    // ========= Post-operation clean-up========/


    plhs[0] = mxGPUCreateMxArrayOnGPU(mxGPUArray_AhAx);
    
    // Free device memories
    cudaFree(d_im_out);

    cudaFree(d_x);
    cudaFree(d_y);

    // Destroy cufft and cufinufft plans
    cufftDestroy(cufft_plan);
    cufinufftf_destroy(cufinufft_type1_plan);
    cufinufftf_destroy(cufinufft_type2_plan);
    cublasDestroy(handle);

    // Destroy MathWorks GPU API objects
    mxGPUDestroyGPUArray(mxGPUArray_im);
    mxGPUDestroyGPUArray(mxGPUArray_SEs);
    mxGPUDestroyGPUArray(mxGPUArray_AhAx);
    mxGPUDestroyGPUArray(mxGPUArray_Phi2Cartesian);

    // Free host allocated memory
    free(x);
    free(y);

    return;
}

