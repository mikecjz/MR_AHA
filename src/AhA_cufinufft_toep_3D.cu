/*
 * MATLAB to CUDA interface to run AhA for MR-Multitasking using cufinufft with float precision.
 * Full 3D non-Cartesian trajectory version with 2x oversampling in all dimensions.
 * Vectorized nufft call possible. This Function assumes all major data variables to be in single (float)
 * precision and on CPU at input. All multidimensional arrays are assumed to have column-major order 
 * (https://en.wikipedia.org/wiki/Row-_and_column-major_order), same as MATLAB.
 * Author: Junzhou Chen (junzhou@chen.engineer)
 * Modified for full 3D non-Cartesian trajectory
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
  int Nz; // Number of third image dimension
  int L;  // rank
  int Ncoils; // Number of coils
  int M; // Total number of non-uniform samples in 3D
  int Ntrans; // Number of stacked transformations
} dataParamsStruct;

__constant__ dataParamsStruct d_dataParams;

/**
 * @brief Multiplies image data by sensitivity encoding maps element-wise with 3D padding.
 *
 * This CUDA kernel handles multiplication of complex-valued 3D (plus rank) image data with
 * sensitivity encoding map (single coil). The multiplication is broadcasted and performed element-wise and the 
 * results are stored in the output array which should be padded to the size of 2*Nx, 2*Ny, 2*Nz and initialized to zero before function call.
 * The phase is negated in the result if the coordinate sum is odd. This ensures that the following FFTs produce centered k-space.
 *
 * @param[in] d_im Readonly pointer to the input device array containing the image data. The array is a flattened
 *             4D array with dimensions [Nx, Ny, Nz, L], where L is the number of image sets or time points.
 * @param[in] d_SE Readonly pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param[out] d_im_SE Pointer to the output device array where the result of the multiplication is stored.
 *                Dimensions [2*Nx, 2*Ny, 2*Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void Im_SE_Pad_shift_AddPhase_3D(const cuFloatComplex *d_im, const cuFloatComplex *d_SE,
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
    int z_hat = z + Nz/2; //z hat is the z-coordinate in the padded grid

    x_hat = (x_hat + Nx) % (2 * Nx); //fftshift-ed coordinate
    y_hat = (y_hat + Ny) % (2 * Ny); //fftshift-ed coordinate
    z_hat = (z_hat + Nz) % (2 * Nz); //fftshift-ed coordinate

    if (x < Nx && y < Ny && z < Nz)
    {
        int se_index = ((z * Ny + y) * Nx + x);
        cuFloatComplex se_val = d_SE[se_index];

        //Multiply sensitivity map with image.
        //Phase is negated in the result if the coordinate sum is odd
        for (int l = 0; l < L; l++)
        {
            int im_index =      ((l * Nz + z) * Ny + y) * Nx + x;
            int im_hat_index =  ((l * Nz*2 + z_hat) * Ny*2 + y_hat) * Nx*2 + x_hat;

            float phase = 1-2*((x+y+z)&1); //1 if even, -1 if odd

            d_im_SE[im_hat_index] = cuCmulf(se_val, d_im[im_index]);

            d_im_SE[im_hat_index].x *= phase;
            d_im_SE[im_hat_index].y *= phase;
        }
    }
}

/**
 * @brief Multiplies image data by sensitivity encoding maps element-wise with 3D padding.
 *
 * This CUDA kernel handles multiplication of complex-valued 3D (plus rank) image data with
 * sensitivity encoding map (single coil). The multiplication is broadcasted and performed element-wise and the 
 * results are stored in the output array which should be padded to the size of 2*Nx, 2*Ny, 2*Nz and initialized to zero before function call.
 * The phase is negated in the result if the coordinate sum is odd. This ensures that the following FFTs produce centered k-space.
 *
 * @param[in] d_im Readonly pointer to the input device array containing the image data. The array is a flattened
 *             4D array with dimensions [2*Nx, 2*Ny, 2*Nz, L], where L is the number of image sets or time points. Image is padded and fftshifted.
 * @param[in] d_SE Readonly pointer to the input device array containing the sensitivity encoding map (single coil). The array
 *              is a flattened 3D array with dimensions [Nx, Ny, Nz].
 * @param[out] d_im_SE Pointer to the output device array where the result of the multiplication is stored.
 *                Dimensions [2*Nx, 2*Ny, 2*Nz, L].
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
 __global__ void Im_padded_shifted_SE_3D(cuFloatComplex *d_im, const cuFloatComplex *d_SE,
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
int z_hat = z + Nz/2; //z hat is the z-coordinate in the padded grid

x_hat = (x_hat + Nx) % (2 * Nx); //fftshift-ed coordinate
y_hat = (y_hat + Ny) % (2 * Ny); //fftshift-ed coordinate
z_hat = (z_hat + Nz) % (2 * Nz); //fftshift-ed coordinate

if (x < Nx && y < Ny && z < Nz)
{
int se_index = ((z * Ny + y) * Nx + x);
cuFloatComplex se_val = d_SE[se_index];

//Multiply sensitivity map with image.
//Phase is negated in the result if the coordinate sum is odd
for (int l = 0; l < L; l++)
{
int im_index =      ((l * Nz*2 + z_hat) * Ny*2 + y_hat) * Nx*2 + x_hat;
int im_SEs_index =  ((l * Nz*2 + z_hat) * Ny*2 + y_hat) * Nx*2 + x_hat;

float phase = 1-2*((x+y+z)&1); //1 if even, -1 if odd

d_im_SE[im_SEs_index] = cuCmulf(se_val, d_im[im_index]);


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
 */
__global__ void Im_SE_adj_3D(cuFloatComplex *d_im, const cuFloatComplex *d_SE,
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
        //Phase is negated in the result if the coordinate sum is odd
        for (int l = 0; l < L; l++)
        {
            int im_index = ((l * Nz + z) * Ny + y) * Nx + x;
            
            float phase = 1-2*((x+y+z)&1); //1 if even, -1 if odd

            d_im[im_index] = cuCmulf(cuConjf(se_val), d_im_SE[im_index]);
        }
    }
}

/**
 * @brief Crops a padded 4D image array back to its original size (3D version).
 *
 * This CUDA kernel extracts the central region from a zero-padded 4D complex-valued image array.
 * The input fftshifted and padded array has dimensions [2*Nx, 2*Ny, 2*Nz, L] and is ifftshifted and cropped to [Nx, Ny, Nz, L].
 * The kernel assumes column-major memory layout. Phase is negated in the result if the coordinate sum is odd.
 *
 * @param[in] d_im_padded Pointer to the input device array containing the fftshifted and padded image data.
 *                        Array dimensions are [2*Nx, 2*Ny, 2*Nz, L].
 * @param[out] d_im Pointer to the output device array that will store the ifftshifted and cropped image.
 *                  Array dimensions are [Nx, Ny, Nz, L].
 */
__global__ void shiftAndCropImage_3D(cuFloatComplex *d_im_padded, cuFloatComplex *d_im)
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
    int z_hat = z + Nz/2; //z hat is the z-coordinate in the padded grid

    x_hat = (x_hat + Nx) % (2 * Nx); //fftshift-ed coordinate
    y_hat = (y_hat + Ny) % (2 * Ny); //fftshift-ed coordinate
    z_hat = (z_hat + Nz) % (2 * Nz); //fftshift-ed coordinate

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            int im_index =      ((l * Nz + z) * Ny + y) * Nx + x;
            int im_hat_index =  ((l * Nz*2 + z_hat) * Ny*2 + y_hat) * Nx*2 + x_hat;

            float phase = 1-2*((x+y+z)&1); //1 if even, -1 if odd
            d_im[im_index] = d_im_padded[im_hat_index];

            d_im[im_index].x *= phase;
            d_im[im_index].y *= phase;
        }
    }
}

/**
 * @brief Updates each k-space [1 x L] line with its corresponding (Φ * ΦH) [L x L] (Hermitian, stored in LOWER TRIANGLE including the diagonal). 
 *        Overall, this amounts to a block-diagonal matrix multiplication for 3D data.
 *
 * @param[in,out] d_k     Pointer to the k-space data being updated. Output will also be stored in this memory. In GPU memory. 
 *                        This is a complex-valued array representing the k-space data from zero-padded image. [2*Nx, 2*Ny, 2*Nz, L]
 * @param[in]     d_Phi2  Pointer to the overall (Φ * ΦH) transformation matrix (Toeplitz form), with Hermitian symmetry stored in 
 *                        LOWER TRIANGLE including the diagonal. In GPU memory. 
 *                        This is a complex-valued array representing the transformation to be applied.
 *                        Dimension [2*Nx, 2*Ny, 2*Nz, (L-1)*L/2 +L].
 */
__global__ void kspaceUpdateToeplitzHermitian_3D(cuFloatComplex *d_k, const cuFloatComplex *d_Phi2Cartesian)
{
    //Assign data parameters from constant memory
    int Nx_hat = d_dataParams.Nx * 2; //padded size
    int Ny_hat = d_dataParams.Ny * 2; //padded size
    int Nz_hat = d_dataParams.Nz * 2; //padded size
    int L = d_dataParams.L;

    // Calculate global thread coordinates, blocks will be launched in 1D grid
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate local thread coordinates with column-major order
    int x_hat = global_thread_idx % Nx_hat;             //padded x-coordinate
    int y_hat = (global_thread_idx / Nx_hat) % Ny_hat;  //padded y-coordinate
    int z_hat = global_thread_idx / (Nx_hat * Ny_hat);  //padded z-coordinate

    // Declare shared memory pointer at the start of the kernel
    extern __shared__ cuFloatComplex sharedMem[];

    cuFloatComplex *sharedInput = sharedMem + threadIdx.x * L;

    if (x_hat < Nx_hat && y_hat < Ny_hat && z_hat < Nz_hat)
    {
        size_t input_index;
        size_t Phi2_index;
        size_t output_index;
        //Load input data into shared memory
        for (int innerL = 0; innerL < L; innerL++)
        {
            //calculate input index based on column-major order
            input_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z_hat + Nz_hat * innerL));
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
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z_hat + Nz_hat * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], d_Phi2Cartesian[Phi2_index]));
                }
                else //upper triangle
                {
                    matrix_linear_index = (2 * L -innerL + 1)*innerL/2 + l - innerL; //Sum of arithmetic series, indices swapped
                    Phi2_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z_hat + Nz_hat * matrix_linear_index));
                    tempValue = cuCaddf(tempValue, cuCmulf(sharedInput[innerL], cuConjf(d_Phi2Cartesian[Phi2_index])));
                }
            }
            output_index = x_hat + Nx_hat * (y_hat + Ny_hat * (z_hat + Nz_hat * l));
            d_k[output_index] = tempValue;
        }
    }
}

/**
 * @brief CUDA kernel that performs element-wise addition of two complex arrays.
 *
 * This kernel adds corresponding elements from array b into array a, storing the result in array a.
 * AKA  a = a + b
 */
__global__ void addArrays(cuFloatComplex *a, const cuFloatComplex *b, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize) {
        a[index] = cuCaddf(a[index], b[index]);
    }
}

/**
 * @brief CUDA kernel that performs element-wise multiplication of two complex arrays.
 *
 * This kernel multiplies corresponding elements from array b into array a, storing the result in array a.
 * AKA  a = a * b
 */
__global__ void multiplyArrays(cuFloatComplex *a, const cuFloatComplex *b, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize) {
        a[index] = cuCmulf(a[index], b[index]);
    }
}

/**
 * @brief Validates input arguments for the AhA_3d_cufinufft MEX function
 */
void checkMexInputs(int nrhs, mxArray const *prhs[])
{
    char const *const instruction = "Usage:\n AhAx = AhA_3d_cufinufft(x, st, SEs, Phi2).";

    char errorMessage[1024];
    // Check input number
    if (nrhs != 4)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Exactly 4 inputs are required. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }

    //check if certain inputs are gpuArray
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[2]) || !mxIsGPUArray(prhs[3]))
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be gpuArray and are complex singles. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }

    //check if certain inputs are valid gpuArrays
    if (!mxGPUIsValidGPUData(prhs[0]) || !mxGPUIsValidGPUData(prhs[2]) || !mxGPUIsValidGPUData(prhs[3]))
    {
        snprintf(errorMessage, sizeof(errorMessage), 
                "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be valid gpuArrays. \n"
                "If you see this message, gpuArray might be cleared by gpuDevice(). \n"
                "%s", 
                instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }

    // Check NU trajectories type - now expecting 3D trajectories
    if (!mxIsSingle(mxGetField(prhs[1], 0, "om")))
    {
        snprintf(errorMessage, sizeof(errorMessage), "3D NU trajectories (st.om) must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }
}

/**
 * @brief Validates data types and complexity of GPU arrays used in AhA_3d_cufinufft
 */
void checkGPUArrays(const mxGPUArray *Phi2Cartesian, const mxGPUArray *im, const mxGPUArray *SEs)
{
    char const *const instruction = "Usage:\n AhAx = AhA_3d_cufinufft(x, st, SEs, Phi2).";

    char errorMessage[1024];
    // Check if gpuArrays are singles
    if (mxGPUGetClassID(Phi2Cartesian) != mxSINGLE_CLASS || mxGPUGetClassID(im) != mxSINGLE_CLASS || mxGPUGetClassID(SEs) != mxSINGLE_CLASS)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be singles (and complex too). \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }

    // Check if gpuArrays are complex
    if (mxGPUGetComplexity(Phi2Cartesian) != mxCOMPLEX || mxGPUGetComplexity(im) != mxCOMPLEX || mxGPUGetComplexity(SEs) != mxCOMPLEX)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), Sensitivity maps (SEs) and Phi2 must be complex (and singles too). \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", errorMessage);
    }
}

/**
 * @brief Performs the AhA operation for MR-Multitasking subspace reconstruction with full 3D non-Cartesian trajectory
 * 
 * This function implements the core AhA operation for a single coil, which consists of:
 * 1. Multiplying input image with sensitivity map (SEs * im)
 * 2. Forward 3D cuFFT with 2x oversampling in all dimensions
 * 3. Type 1 3D NUFFT to transform from uniform to non-uniform k-space
 * 4. Block-diagonal k-space update with Phi2
 * 5. Type 2 3D NUFFT to transform back to uniform space
 * 6. Inverse 3D cuFFT
 * 7. Conjugate multiplication with sensitivity map
 */
void AhA_by_coil_3D(const cuFloatComplex *d_im,//input image
                    const cuFloatComplex *d_SE, //single coil map
                    const cuFloatComplex *d_Phi2Cartesian, //(Φ * ΦH)
                    cuFloatComplex *d_k, //temporary k-space storage
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

    // Assign data parameters from dataParams struct
    int Nx = dataParams.Nx;
    int Ny = dataParams.Ny;
    int Nz = dataParams.Nz;
    int L = dataParams.L;
    int M = dataParams.M;
    int Ntrans = dataParams.Ntrans;

    const int imageSize = Nx * Ny * Nz * L;
    const int paddedImageSize = 8 * imageSize; // 2x oversampling in all 3 dimensions

    // Initialize Blocks and Grids
    int blockSize = 256;
    int gridSize = (imageSize + 255) / 256;

    // ==========Amplify image simuplating Type-2 NUFFT========== //

    cuFloatComplex *d_im_amplified;
    cuFloatComplex *d_im_temp;
    cudaMalloc(&d_im_amplified, paddedImageSize * sizeof(cuFloatComplex));
    cudaMemset(d_im_amplified, 0, paddedImageSize * sizeof(cuFloatComplex));
    cudaMalloc(&d_im_temp, imageSize * sizeof(cuFloatComplex));
    cudaMemcpy(d_im_temp, d_im, imageSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);

    //d_im_SE is now in image space.
    cudaEventRecord(start);
    cufinufftf_deconvolve(cufinufft_type2_plan, d_im_temp, d_im_amplified);

    cudaFree(d_im_temp);

    // shiftAndCropImage_3D<<<gridSize, blockSize>>>(d_im_SE, d_im_SE_deconv);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("type2 cufinufftf_deconvolve execution time: %.6f seconds\n", seconds);


    // Setting up pointers for sensitivity encoded images
    cuFloatComplex *d_im_SE;
    cudaMalloc(&d_im_SE, paddedImageSize * sizeof(cuFloatComplex));
    cudaMemset(d_im_SE, 0, paddedImageSize * sizeof(cuFloatComplex));

    // ==========Broadcast SE to image input on GPU. (SEs * im)========== //
    cudaEventRecord(start);
    Im_padded_shifted_SE_3D<<<gridSize, blockSize>>>(d_im_amplified, d_SE, d_im_SE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("Im_SE_Pad_shift_AddPhase_3D execution time: %.6f seconds\n", seconds);

    cudaFree(d_im_amplified);

    // ========== Batched Forward 3D cuFFT ========== /
    cudaEventRecord(start);
    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("cufftExecC2C forward execution time: %.6f seconds\n", seconds);

    // ========== Block-diagonal k-space update in Cartesian space ========== /

    if (L > 1) { //low rank case
        int blockSizeBlkdiag;
        if (L < 24) {
            blockSizeBlkdiag = 256;
        } else if (L < 48) {
            blockSizeBlkdiag = 128;
        } else if (L < 96) {
            blockSizeBlkdiag = 64;
        }

        const int gridSizeBlkdiag = (paddedImageSize + blockSizeBlkdiag - 1) / blockSizeBlkdiag;
        const int sharedMemSizeBlkdiag = blockSizeBlkdiag * L * sizeof(cuFloatComplex);
        
        cudaEventRecord(start);
        kspaceUpdateToeplitzHermitian_3D<<<gridSizeBlkdiag, blockSizeBlkdiag, sharedMemSizeBlkdiag>>>(d_im_SE, d_Phi2Cartesian);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        seconds = milliseconds / 1000.0f;
        if (is_verbose) mexPrintf("kspaceUpdateToeplitzHermitian_3D execution time: %.6f seconds\n", seconds);
    } 
    else { //regular case (rank == 1)
        int blockSizeDiag = 256;
        const int gridSizeDiag = (paddedImageSize + blockSizeDiag - 1) / blockSizeDiag;
        cudaEventRecord(start);
        multiplyArrays<<<gridSizeDiag, blockSizeDiag>>>(d_im_SE, d_Phi2Cartesian, paddedImageSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    // ========== Inverse 3D cuFFT ========== /
    cudaEventRecord(start);
    cufftExecC2C(cufft_plan, d_im_SE, d_im_SE, CUFFT_INVERSE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("cufftExecC2C inverse execution time: %.6f seconds\n", seconds);

    // ========= FINUFFT Type 1 deconvolution ONLY========/

    cuFloatComplex *d_im_SE_deconv;
    cudaMalloc(&d_im_SE_deconv, imageSize * sizeof(cuFloatComplex));

    //d_im_SE is now in image space.
    cudaEventRecord(start);
    cufinufftf_deconvolve(cufinufft_type1_plan, d_im_SE_deconv, d_im_SE);

    // shiftAndCropImage_3D<<<gridSize, blockSize>>>(d_im_SE, d_im_SE_deconv);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("type1 cufinufftf_deconvolve execution time: %.6f seconds\n", seconds);

    cudaFree(d_im_SE); //no longer used

    // ========= Coil combination (adjoint SE operation)========/
    cudaEventRecord(start);
    Im_SE_adj_3D<<<gridSize, blockSize>>>(d_im_out, d_SE, d_im_SE_deconv);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("Im_SE_adj_3D execution time: %.6f seconds\n", seconds);

    cudaFree(d_im_SE_deconv);

    // Overall timing end
    cudaEventRecord(function_stop);
    cudaEventSynchronize(function_stop);
    cudaEventElapsedTime(&milliseconds, function_start, function_stop);
    seconds = milliseconds / 1000.0f;
    if (is_verbose) mexPrintf("AhA_by_coil_3D total execution time: %.6f seconds\n", seconds);

    cudaEventDestroy(function_start);
    cudaEventDestroy(function_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Main function for the MR-Multitasking subspace recon AhA operations for full 3D non-Cartesian trajectory
 * 
 * @param x prhs[0]. Spatial basis function image input of size [Nx, Ny, Nz, L]. 
 * @param st prhs[1]. Matlab structure containing necessary data parameters including 3D trajectories om [M, 3] for kx, ky, kz.
 * @param SEs prhs[2]. Sensitivity maps of dimension [Nx, Ny, Nz, Ncoils].
 * @param Phi2 prhs[3]. The overall (Φ * ΦH) matrix. Dimension [2*Nx, 2*Ny, 2*Nz, (L-1)*L/2 + L].
 * 
 * @param AhAx plhs[0]. The output of the operation AhA(x). Dimension [Nx*Ny*Nz*L, 1]
 * 
 * @note AhAx = AhA_3d_cufinufft(x, st, SEs, Phi2)
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *x, *y, *z;     // 3D trajectory coordinates
    const mxGPUArray *mxGPUArray_im;               // [Nx, Ny, Nz, L]
    const mxGPUArray *mxGPUArray_SEs;              // [Nx, Ny, Nz, Ncoils]
    const mxGPUArray *mxGPUArray_Phi2Cartesian;    // [2*Nx, 2*Ny, 2*Nz, num_lower_coordinates]

    int64_t Nx, Ny, Nz, Ncoils, L;
    size_t M;           // number of 3D NU points
    size_t Ntrans;      // Number of stacked transforms
    dataParamsStruct dataParams;

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Check inputs
    checkMexInputs(nrhs, prhs);

    // Input 1: image domain data
    mxGPUArray_im = mxGPUCreateFromMxArray(prhs[0]);

    // Input 2: 3D Trajectories and Image Dimensions
    size_t numRows = mxGetM(mxGetField(prhs[1], 0, "om"));
    size_t numCols = mxGetN(mxGetField(prhs[1], 0, "om"));
    
    if (numCols != 3) {
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", "3D trajectories (st.om) must have 3 columns [kx, ky, kz].");
    }
    
    float *om = mxGetSingles(mxGetField(prhs[1], 0, "om"));
    x = (float *)malloc(numRows * sizeof(float));
    y = (float *)malloc(numRows * sizeof(float));
    z = (float *)malloc(numRows * sizeof(float));
    
    for (size_t i = 0; i < numRows; i++) {
        x[i] = om[i];
        y[i] = om[numRows + i];
        z[i] = om[2*numRows + i];
    }

    Nx = (int64_t)mxGetDoubles(mxGetField(prhs[1], 0, "Nd"))[0];
    Ny = (int64_t)mxGetDoubles(mxGetField(prhs[1], 0, "Nd"))[1];
    Nz = (int64_t)mxGetDoubles(mxGetField(prhs[1], 0, "Nd"))[2];
    M = (size_t)mxGetScalar(mxGetField(prhs[1], 0, "M"));
    L = (int64_t)mxGetScalar(mxGetField(prhs[1], 0, "L"));

    // Make sure L is not too large
    if (L >= 96) {
        mexErrMsgIdAndTxt("AhA_3d_cufinufft:InputError", "L must be less than 96 (< 96)");
    }

    Ntrans = L; // Number of stacked transforms for each coil

    // Input 3: Sensitivity Maps
    mxGPUArray_SEs = mxGPUCreateFromMxArray(prhs[2]);
    const mwSize *coilDims = mxGPUGetDimensions(mxGPUArray_SEs);
    Ncoils = (int64_t)coilDims[3];

    // Input 4: Phi2
    mxGPUArray_Phi2Cartesian = mxGPUCreateFromMxArray(prhs[3]);

    // Check gpuArrays
    checkGPUArrays(mxGPUArray_Phi2Cartesian, mxGPUArray_im, mxGPUArray_SEs);

    // Assign data parameters to struct
    dataParams.Nx = Nx;
    dataParams.Ny = Ny;
    dataParams.Nz = Nz;
    dataParams.L = L;
    dataParams.Ncoils = Ncoils;
    dataParams.M = M;
    dataParams.Ntrans = Ntrans;

    // Create output gpuArray
    mxGPUArray *mxGPUArray_AhAx;
    cuFloatComplex *d_AhAx;
    mwSize dims[1] = {(mwSize)(Nx * Ny * Nz * L)};
    mxGPUArray_AhAx = mxGPUCreateGPUArray(1, dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    d_AhAx = (cuFloatComplex *)mxGPUGetData(mxGPUArray_AhAx);

    // Assign data parameters to device constant memory
    cudaMemcpyToSymbol(d_dataParams, &dataParams, sizeof(dataParamsStruct));

    // Declare device pointers
    const cuFloatComplex *d_im, *d_SEs;
    cuFloatComplex *d_im_out, *d_k;
    float *d_x, *d_y, *d_z;
    const cuFloatComplex *d_Phi2Cartesian;

    // Allocate device memories
    cudaMalloc(&d_im_out, Nx * Ny * Nz * L * sizeof(cuFloatComplex));
    cudaMalloc(&d_k, M * Ntrans * sizeof(cuFloatComplex));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_z, M * sizeof(float));

    // Process input data
    d_im = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_im);
    d_SEs = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_SEs);
    d_Phi2Cartesian = (cuFloatComplex *)mxGPUGetDataReadOnly(mxGPUArray_Phi2Cartesian);

    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, M * sizeof(float), cudaMemcpyHostToDevice);

    // Make cuFFT and cufinufft plans
    cufftHandle cufft_plan;
    int n[3] = {(int)(Nz*2), (int)(Ny*2), (int)(Nx*2)};
    cufftPlanMany(&cufft_plan, 3, n, NULL, NULL, 8*Nx*Ny*Nz, NULL, NULL, 8*Nx*Ny*Nz, CUFFT_C2C, L);

    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac = 2.0;
    int64_t modes[3] = {Nx, Ny, Nz};

    cufinufftf_plan cufinufft_type1_plan, cufinufft_type2_plan;
    cufinufftf_makeplan(1, 3, modes, 1, Ntrans, 1e-6, &cufinufft_type1_plan, &opts);
    cufinufftf_setpts(cufinufft_type1_plan, M, d_x, d_y, d_z, 0, NULL, NULL, NULL);

    cufinufftf_makeplan(2, 3, modes, -1, Ntrans, 1e-6, &cufinufft_type2_plan, &opts);
    cufinufftf_setpts(cufinufft_type2_plan, M, d_x, d_y, d_z, 0, NULL, NULL, NULL);

    // Loop through each coil
    cublasHandle_t handle;
    cublasCreate(&handle);
    cuFloatComplex scale_factor_fft = make_cuFloatComplex(1.0f/(Nx*Ny*Nz), 0.0f); // FFT scaling factor for 3D 2x oversampling

    const cuFloatComplex *d_SE;
    bool is_verbose = false;
    
    for (int coil = 0; coil < Ncoils; coil++) {
        d_SE = d_SEs + coil * Nx * Ny * Nz;

        AhA_by_coil_3D(d_im, d_SE, d_Phi2Cartesian, d_k, d_im_out, cufft_plan, 
                       cufinufft_type1_plan, cufinufft_type2_plan, dataParams, is_verbose);

        // Add the output from each coil to d_AhAx and scale by FFT scaling factor
        cublasCaxpy(handle, Nx * Ny * Nz * L, &scale_factor_fft, d_im_out, 1, d_AhAx, 1);
        is_verbose = false;
    }

    // Post-operation clean-up
    plhs[0] = mxGPUCreateMxArrayOnGPU(mxGPUArray_AhAx);

    // Free device memories
    cudaFree(d_im_out);
    cudaFree(d_k);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    // Destroy plans
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
    free(z);

    return;
}