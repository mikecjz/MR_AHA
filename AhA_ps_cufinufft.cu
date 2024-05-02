/*
 * MATLAB to CUDA interface to run AhA for MR-Multitasking using cufinufft with float precision.
 * Vectorized nufft call possible. This Function assumes all major data variables to be in single (float)
 * precision and on CPU at input.
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

/**
 * @brief Multiplies image data by sensitivity encoding maps element-wise.
 *
 * This CUDA kernel is designed to handle the multiplication of complex-valued 3D image data with
 * sensitivity encoding maps across multiple coils and time points. The multiplication is performed
 * element-wise and the results are stored in an output array.
 *
 * @param d_im Pointer to the input device array containing the image data. The array is a flattened
 *             4D array with dimensions [Nx, Ny, Nz, L], where L is the number of image sets or time points.
 * @param d_SEs Pointer to the input device array containing the sensitivity encoding maps. The array
 *              is a flattened 4D array with dimensions [Nx, Ny, Nz, Ncoils, 1].
 * @param d_im_SEs Pointer to the output device array where the result of the multiplication is stored.
 *                 It has the same structure and dimensions as `d_SEs`.
 * @param Nx The size of the first dimension (width) of the image and sensitivity maps.
 * @param Ny The size of the second dimension (height) of the image and sensitivity maps.
 * @param Nz The size of the third dimension (depth) of the image and sensitivity maps.
 * @param Ncoils The number of coils, corresponding to the fourth dimension in `d_SEs` and `d_im_SEs`.
 * @param L The number of image sets or time points, corresponding to the last dimension in all arrays.
 *
 * @note The kernel assumes that all input and output arrays are properly allocated and reside in GPU memory.
 *       It is designed to be executed on a CUDA-capable GPU with a grid size that covers the entire volume
 *       of the image (Nx, Ny, Nz).
 */
__global__ void SEs_times_Im( cuFloatComplex *d_im,  cuFloatComplex *d_SEs,
                             cuFloatComplex *d_im_SEs, int Nx, int Ny, int Nz, int Ncoils, int L)
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            for (int coil = 0; coil < Ncoils; coil++)
            {
                // Calculate the index for d_im (without coil dimension)
                // int im_index = ((z * Ny + y) * Nx + x) * L + l;
                int im_index = ((l * Nz + z) * Ny + y) * Nx + x;

                // Calculate the index for d_SEs(without rank dimension)
                // int ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil);
                int ses_index = ((coil * Nz + z) * Ny + y) * Nx + x;

                // Calculate the index for d_im_SEs
                // int im_ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil) * L + l;
                int im_ses_index = (((l * Ncoils + coil) * Nz + z) * Ny + y) * Nx + x;

                // Perform the multiplication
                d_im_SEs[im_ses_index] = cuCmulf(d_im[im_index], d_SEs[ses_index]);
            }
        }
    }
}


__global__ void SEs_times_Im_adj( cuFloatComplex *d_im,  cuFloatComplex *d_SEs,
                             cuFloatComplex *d_im_SEs, int Nx, int Ny, int Nz, int Ncoils, int L)
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            // Calculate the index for d_im (without coil dimension)
            int im_index = ((l * Nz + z) * Ny + y) * Nx + x;

            cuFloatComplex tempValue = {0.0f, 0.0f};

            for (int coil = 0; coil < Ncoils; coil++)
            {
                

                // Calculate the index for d_SEs(without rank dimension)
                // int ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil);
                int ses_index = ((coil * Nz + z) * Ny + y) * Nx + x;

                // Calculate the index for d_im_SEs
                // int im_ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil) * L + l;
                int im_ses_index = (((l * Ncoils + coil) * Nz + z) * Ny + y) * Nx + x;

                // Perform the multiplication
                tempValue = cuCaddf(tempValue, cuCmulf(d_im_SEs[im_ses_index], cuConjf(d_SEs[ses_index])));
            }

            d_im[im_index] = tempValue;
            
        }
    }
}

__global__ void permuteKernelThirdToFirst(cuFloatComplex *input, cuFloatComplex *output,
                                          int Nx, int Ny, int Nz, int Ncoils, int L)
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            for (int coil = 0; coil < Ncoils; coil++)
            {
                
                // Calculate the index for d_im_SEs
                // int im_ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil) * L + l;
                int in_index = (((l * Ncoils + coil) * Nz + z) * Ny + y) * Nx + x;

                int out_index = (((l * Ncoils + coil) * Ny + y) * Nx + x) * Nz + z;

                // Perform index exchange
                output[out_index] = input[in_index];
            }
        }
    }
}


__global__ void permuteKernelFirstToThird(cuFloatComplex *input, cuFloatComplex *output,
                                          int Nx, int Ny, int Nz, int Ncoils, int L)
{
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            for (int coil = 0; coil < Ncoils; coil++)
            {
                
                // Calculate the index for d_im_SEs
                // int im_ses_index = (((z * Ny + y) * Nx + x) * Ncoils + coil) * L + l;
                int out_index = (((l * Ncoils + coil) * Nz + z) * Ny + y) * Nx + x;

                int in_index = (((l * Ncoils + coil) * Ny + y) * Nx + x) * Nz + z;

                // Perform the multiplication
                output[out_index] = input[in_index];
            }
        }
    }
}

void cufinufftf2d2(cuFloatComplex *d_im_SEs, cuFloatComplex * d_k, float* d_x, float *d_y, 
                   const int64_t *modes, size_t M, size_t Ntrans)
{
    cufinufftf_plan plan; //declare cufinufft plan object

    //Assuing all values properly initialzed.
    cufinufftf_makeplan(2, 2, modes, 1, Ntrans, 1e-6, &plan, NULL);// make plan

    //Set Points
    cufinufftf_setpts(plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    //Execute
    cufinufftf_execute(plan, d_k, d_im_SEs);

    cufinufftf_destroy(plan);

}

void cufinufftf2d1(cuFloatComplex *d_im_SEs, cuFloatComplex * d_k, float* d_x, float *d_y, 
                   const int64_t *modes, size_t M, size_t Ntrans)
{
    cufinufftf_plan plan; //declare cufinufft plan object

    //Assuing all values properly initialzed.
    cufinufftf_makeplan(1, 2, modes, 1, Ntrans, 1e-6, &plan, NULL);// make plan

    //Set Points
    cufinufftf_setpts(plan, M, d_x, d_y, NULL, 0, NULL, NULL, NULL);

    //Execute
    cufinufftf_execute(plan, d_k, d_im_SEs);

    cufinufftf_destroy(plan);

}

__global__ void KspaceUpdate(cuFloatComplex *d_k, cuFloatComplex *d_Phi2, cuFloatComplex* temp, 
                            size_t Ntrajs, size_t Nx, size_t Nz, size_t Ncoils, size_t L)
{
    // Calculate global thread coordinates
    int traj = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (traj < Ntrajs && x < Nx && z < Nz)
    {
        for (int l = 0; l < L; l++)
        {
            for (int coil = 0; coil < Ncoils; coil++)
            {
                size_t out_idx = (((l * Ncoils + coil) * Nz + z) * Nx + x) * Ntrajs + traj;

                cuFloatComplex tempValue = {0.0f,0.0f};
                //Vector Matrix Multiplications
                for (int innerL = 0; innerL < L; innerL++)
                {
                    size_t k_index = (((innerL * Ncoils + coil) * Nz + z) * Nx + x) * Ntrajs + traj;
                    //size_t Phi2_index = (((innerL * L + L) * l) * Nz + z) * Ntrajs + traj;
                    size_t Phi2_index = ((z * Ntrajs + traj) * L + innerL) * L + l;
                    
                    tempValue  =  cuCaddf(tempValue, cuCmulf(d_k[k_index], d_Phi2[Phi2_index]));
                }

                temp[out_idx] = tempValue;

            }
        }
    }

}


/*
 * Host code
 * im = AhA_ps_cufinufft(x, st, SEs, Phi2)
 *
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    clock_t start, end;
    double cpu_time_used;

    /* Declare all variables.*/
    float *x;              // om(1)
    float *y;              // om(2)
    mxComplexSingle *im;   // Pointer to complex image-domain data (x), on host [Nx, Ny, Nz, L]
    mxComplexSingle *SEs;  // Pointer to complex Complex sensitivity maps (SEs), on host [Nx, Ny, Nz, Ncoils]
    mxComplexSingle *Phi2; // Pointer to complex Phi2 data (Phi2, aka, Om), on host [L, L, Ntrajs, Nz]
    int isSign = 1;        // Whether to use + sign in complex exponential. Default to true
    float eps = 1e-6;      // NUFFT precision, default 1e-6;
    int64_t Nx;
    int64_t Ny;
    int64_t Nz;
    int64_t Ncoils;
    int64_t L;          // rank
    size_t M;           // number of NU points
    size_t Ntrans;      // Number of stacked transforms
    size_t Ntrajs;      // Number of in-plane tranjectories. e.g. 280
    int GPUDeviceIndex; // Which GPU to use

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
        snprintf(errorMessage, sizeof(errorMessage), "Inputs original image (x), NU trajectories (st.om), and Sensitivity maps (SEs) must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("AhA_ps_cufinufft:InputError", errorMessage);
    }

    // ==========Assign Pointers and values ========== //

    // Input 1: image domain data
    im = mxGetComplexSingles(prhs[0]); // Get input image domain data

    // Input 2: Trajectories and Image Dimensions
    // Separte om(:,1) and om(:,2)
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
    Ntrajs = (size_t)(M/Nx);
    

    // Input 3: Sensitivity Maps
    const mwSize *coilDims = mxGetDimensions(prhs[2]);
    Ncoils = (int64_t)coilDims[3];
    SEs = mxGetComplexSingles(prhs[2]);

    // Input 4: Phi2 (aka, Om)
    const mwSize *phi2Dims = mxGetDimensions(prhs[3]);
    L = (int64_t)phi2Dims[0];
    Phi2 = mxGetComplexSingles(prhs[3]);

    Ntrans = Nz * Ncoils * L;

    // ==========Initialize the MathWorks GPU API. ========== //
    mxInitGPU();

    // ==========Broadcast SEs to image input on GPU. (SEs * im)========== //
    cuFloatComplex *d_im, *d_SEs; // coil-combined image and SEs on device
    cuFloatComplex *d_im_SEs;     // im * SEs on devices

    // Allocate device memories
    cudaMalloc(&d_im, Nx * Ny * Nz * L * sizeof(cuFloatComplex));
    cudaMalloc(&d_SEs, Nx * Ny * Nz * Ncoils * sizeof(cuFloatComplex));
    cudaMalloc(&d_im_SEs, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));

    // Copy data to device
    cudaMemcpy(d_im, im, Nx * Ny * Nz * L * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SEs, SEs, Nx * Ny * Nz * Ncoils * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Intialze Blocks and Grids and Perform Broadcasting
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((Nx + 15) / 16, (Ny + 16) / 16, Nz);

    SEs_times_Im<<<gridSize, blockSize>>>(d_im, d_SEs, d_im_SEs, Nx, Ny, Nz, Ncoils, L);
    //cudaDeviceSynchronize();

    cudaFree(d_im); // Free subsequently unsed variable d_im on device


    // ========== Forward cuFFT in the 3rd (Z) dimension ========== /

    // Permute 3rd dimension to first dimension
    cuFloatComplex *temp;
    cudaMalloc(&temp, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));
    permuteKernelThirdToFirst<<<gridSize, blockSize>>>(d_im_SEs, temp, Nx, Ny, Nz, Ncoils, L);

    cudaFree(d_im_SEs);
    d_im_SEs = temp; //Assign image to permuted memoiry
    temp = nullptr; //Prevent accidental use of temp

    //Make 1D cuFFT Plan
    cufftHandle fft_plan; //Decalre plan object
    cufftPlan1d(&fft_plan, Nz, CUFFT_C2C, Nx * Ny * Ncoils * L);

    //Execute
    cufftExecC2C(fft_plan, d_im_SEs, d_im_SEs, CUFFT_FORWARD);

    // Permute 1st dimension back to to 3rd dimension
    cudaMalloc(&temp, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));
    permuteKernelFirstToThird<<<gridSize, blockSize>>>(d_im_SEs, temp, Nx, Ny, Nz, Ncoils, L);

    cudaFree(d_im_SEs);
    d_im_SEs = temp; //Assign image to permuted memoiry
    temp = nullptr; //Prevent accidental use of temp

    //Destroy cuFFT Plan
    cufftDestroy(fft_plan);


    // ========== FINUFFT Type 2, U -> NU========== /

    //Declare device variables
    float *d_x, *d_y;
    cuFloatComplex *d_k, *d_Phi2;

    int64_t modes[2] = {Nx, Ny};

     // Allocate device memories
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_k, M * Ntrans * sizeof(float _Complex));
    cudaMalloc(&d_Phi2, Ntrajs * Nz * L * L * sizeof(float _Complex));

    // Copy values to device
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Phi2, Phi2, Ntrajs * Nz * L * L * sizeof(float _Complex), cudaMemcpyHostToDevice);


    //Do Type 2 transform
    cufinufftf2d2(d_im_SEs, d_k, d_x, d_y, modes, M, Ntrans);

    //values in d_im_SEs nolong uses, clear
    cudaFree(d_im_SEs);
    d_im_SEs = nullptr;


    // =========Block-Diagonal kspace update ========/
    cudaMalloc(&temp, Ntrajs * Nx * Nz * Ncoils * L * sizeof(cuFloatComplex));
    //cudaMemset(temp, 0, Ntrajs * Nx * Nz * Ncoils * L * sizeof(cuFloatComplex));

    dim3 blockSizeBlkdiag(16, 16, 1);
    dim3 gridSizeBlkdiag((Ntrajs + 15) / 16, (Nx + 16) / 16, Nz);
    KspaceUpdate<<<gridSizeBlkdiag, blockSizeBlkdiag>>>(d_k, d_Phi2, temp, Ntrajs, Nx, Nz, Ncoils, L);

    cudaFree(d_k);
    d_k = temp; //Assign ksspace to updated values memory
    temp = nullptr; //Prevent accidental use of temp




    // ========= FINUFFT Type 1, NU -> U ========/
    cudaMalloc(&d_im_SEs, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));// Reallocate device multi-coil image data mem
    cufinufftf2d1(d_im_SEs, d_k, d_x, d_y, modes, M, Ntrans);

    //Clear out device memories
    cudaFree(d_k);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_Phi2);






    // ========== Inverse cuFFT in the 3rd (Z) dimension ========== /

    // Permute 3rd dimension to first dimension
    cudaMalloc(&temp, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));
    permuteKernelThirdToFirst<<<gridSize, blockSize>>>(d_im_SEs, temp, Nx, Ny, Nz, Ncoils, L);

    cudaFree(d_im_SEs);
    d_im_SEs = temp; //Assign image to permuted memoiry
    temp = nullptr; //Prevent accidental use of temp

    //Make 1D cuFFT Plan
    cufftHandle ifft_plan; //Decalre plan object
    cufftPlan1d(&ifft_plan, Nz, CUFFT_C2C, Nx * Ny * Ncoils * L);

    //Execute
    cufftExecC2C(ifft_plan, d_im_SEs, d_im_SEs, CUFFT_INVERSE);

    // Permute 1st dimension back to to 3rd dimension
    cudaMalloc(&temp, Nx * Ny * Nz * Ncoils * L * sizeof(cuFloatComplex));
    permuteKernelFirstToThird<<<gridSize, blockSize>>>(d_im_SEs, temp, Nx, Ny, Nz, Ncoils, L);

    cudaFree(d_im_SEs);
    d_im_SEs = temp; //Assign image to permuted memoiry
    temp = nullptr; //Prevent accidental use of temp

    //Destroy cuFFT Plan
    cufftDestroy(ifft_plan);

    // ========= coil_combination ========/
    cudaMalloc(&d_im, Nx * Ny * Nz * L * sizeof(cuFloatComplex));// Reallocate device single-coil image data mem
    SEs_times_Im_adj<<<gridSize, blockSize>>>(d_im, d_SEs, d_im_SEs, Nx, Ny, Nz, Ncoils, L);
    

    //clear out device memories
    cudaFree(d_SEs);
    //cudaFree(d_im_SEs);

    // Copy results to host
    mxComplexSingle *AhAx = (mxComplexSingle *)mxMalloc(Ny * Nx * Nz  * Ncoils *L * sizeof(mxComplexSingle));
    cudaMemcpy(AhAx, d_im_SEs, Ny * Nx * Nz  *Ncoils * L * sizeof(mxComplexSingle), cudaMemcpyDeviceToHost);

    cudaFree(d_im);
    cudaFree(d_im_SEs);


    // Create output
    mwSize out_dimensions[1] = {(size_t)Nx * (size_t)Ny * (size_t)Nz * (size_t)Ncoils * L};
    const mwSize *ndims = out_dimensions;
    plhs[0] = mxCreateNumericArray(1, ndims, mxSINGLE_CLASS, mxCOMPLEX);

    // mexPrintf("Create output  Complete\n");

    // Set Array Value
    mxSetComplexSingles(plhs[0], AhAx);

    // Free host allocated memory
    free(x);
    free(y);
    return;
}
