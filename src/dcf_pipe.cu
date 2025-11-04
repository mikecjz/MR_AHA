/*
 * MATLAB to CUDA interface to call cufinufft 3D type 1 (NU->U) Spreading ONLY with float precision. 
 * Vectorized nufft call possible. This Function assumes all major data variables to be in single (float) 
 * precision and on CPU at input. a GPU output is created.
 *
 * Author: Junzhou Chen (junzhou[AT]chen[DOT]engineer)
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
#include <time.h>
#include <string.h>
#include <cufinufft/types.h>

__global__ void initComplexOnes(cuFloatComplex *arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] = make_cuFloatComplex(1.0f, 0.0f);
}

__global__ void divideArraysInPlace(cuFloatComplex *a, cuFloatComplex *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        cuFloatComplex denominator = b[idx];
        // Check for division by zero
        if (cuCrealf(denominator) == 0.0f && cuCimagf(denominator) == 0.0f) {
            a[idx] = make_cuFloatComplex(0.0f, 0.0f);
        } else {
            a[idx] = cuCdivf(a[idx], denominator);
        }
    }
}




/**
 * Host code
 *run in matlab: im = dcf_pipe(x,y,z,eps,Nx,Ny,Nz,M)
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    clock_t start, end;
    double cpu_time_used;

    /* Declare all variables.*/
    float *kx;          // frequency domain dimension 1 coordinates
    float *ky;          // frequency domain dimension 2 coordinates
    float *kz;          // frequency domain dimension 3 coordinates
    float eps;      //NUFFT precision
    int64_t Nx;     //Size of first dimension of image
    int64_t Ny;     //Size of second dimension of image
    int64_t Nz;     //Size of third dimension of image
    size_t M;      // number of NU points
    int num_iterations; // number of iterations for DCF pipeline
    //int GPUDeviceIndex; //Which GPU to use

    char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const *const errMsg = "Invalid input to MEX file.";
    char const *const instruction = "Usage:\n im = dcf_pipe(x,y,z,eps,Nx,Ny,Nz,M,num_iterations).";

    

    char errorMessage[1024]; // remember to ensure this buffer is large enough for your message
    // Check input number
    if (nrhs != 9)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Exactly 9 inputs required. \n%s", instruction);
        mexErrMsgIdAndTxt("dcf_pipe:InputError", errorMessage);
    }

    // Check input type
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]))
    {   
        snprintf(errorMessage, sizeof(errorMessage), "Inputs 1-3 must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("dcf_pipe:InputError", errorMessage);
    }

    // Assign Pointers and value
    kx = mxGetSingles(prhs[0]);
    ky = mxGetSingles(prhs[1]);
    kz = mxGetSingles(prhs[2]);

    eps = (float)mxGetScalar(prhs[3]);
    Nx = (int64_t)mxGetScalar(prhs[4]);
    Ny = (int64_t)mxGetScalar(prhs[5]);
    Nz = (int64_t)mxGetScalar(prhs[6]);
    M = (size_t)mxGetScalar(prhs[7]);
    num_iterations = (int)mxGetScalar(prhs[8]);
    // Initialize the MathWorks GPU API.
    mxInitGPU();
    //cudaSetDevice(GPUDeviceIndex); //Set which GPU to run. This should strictly matach the one being used by MATLAB

    // Create output gpuArray
    mxGPUArray *dm_dcf; // device veraible for dcf for matlab output
    mwSize dims[1] = {M};
    dm_dcf = mxGPUCreateGPUArray(1, dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);


    int64_t im_padded_size = Nx * Ny * Nz * 8;
    int64_t im_size =  Nx * Ny * Nz;



    // Create on device variables
    float *d_kx, *d_ky, *d_kz;
    cuFloatComplex *d_c, *d_dcf, *d_fw, *d_im;

    // Allocate device memories
    cudaMalloc(&d_kx, M * sizeof(float));
    cudaMalloc(&d_ky, M * sizeof(float));
    cudaMalloc(&d_kz, M * sizeof(float));
    cudaMalloc(&d_c, M  * sizeof(cuFloatComplex));
    cudaMalloc(&d_fw, im_padded_size * sizeof(cuFloatComplex));
    cudaMalloc(&d_im, im_size * sizeof(cuFloatComplex));

    //Get pointer to underlying data of dm_dcf
    d_dcf = (cuFloatComplex *)mxGPUGetData(dm_dcf);


    // Copy values to device
    cudaMemcpy(d_kx, kx, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, ky, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz, kz, M * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize d_k and d_dcf = 1 + 0i
    initComplexOnes<<<(M+255)/256, 256>>>(d_c, M);
    initComplexOnes<<<(M+255)/256, 256>>>(d_dcf, M);
    cudaDeviceSynchronize();



    //mexPrintf("CUDA MEM COPY Complete\n");

    // Create default cufinufft plan object
    cufinufftf_plan plan_type1, plan_type2;
    cufinufft_opts opts;

    // Create modes
    int64_t modes[3] = {Nx, Ny, Nz};

    // make cufinufft plan
    /*
     * int cufinufftf_makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntr, float tol, cufinufftf_plan *plan, cufinufft_opts *opts)
     */
    cufinufft_default_opts(&opts); //make default options

    opts.upsampfac = 2.0;

    cufinufftf_makeplan(1, 3, modes, 1, 1, eps, &plan_type1, &opts);
    cufinufftf_makeplan(2, 3, modes, -1, 1, eps, &plan_type2, &opts);


    //mexPrintf("cufinufftf_makeplan Complete\n");


    // Set points
    /*
     * int cufinufftf_setpts(cufinufftf_plan plan, int M, float* x, float* y, float* z, int N, float* s, float* t, float *u)
     */
    cufinufftf_setpts(plan_type1, M, d_kx, d_ky, d_kz, 0, NULL, NULL, NULL);
    cufinufftf_setpts(plan_type2, M, d_kx, d_ky, d_kz, 0, NULL, NULL, NULL);
    //mexPrintf("cufinufftf_setpts Complete\n");





    /************ EXECUTE SPREADING AND EXTRACT************/
    start = clock();
    
    for (int i = 0; i < num_iterations; i++) {
        cufinufftf_extract(plan_type1, d_dcf, d_im, d_fw);
        cufinufftf_extract(plan_type2, d_c, d_im, d_fw);
        divideArraysInPlace<<<(M+255)/256, 256>>>(d_dcf, d_c, M);

    }


    end = clock();
    double timeExecute = ((double)(end-start))/ CLOCKS_PER_SEC;
    mexPrintf("DCF pipeline execution Time (s): %.5f\n", timeExecute);

    //mexPrintf("cufinufftf_execute Complete\n");
    /************ EXECUTE END *****************************/





    cufinufftf_destroy(plan_type1);
    cufinufftf_destroy(plan_type2);
    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);
    cudaFree(d_c);
    cudaFree(d_fw);
    cudaFree(d_im);
    //mexPrintf("cufinufftf_cleancup Complete\n");

    //Create output 
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(dm_dcf);

    //Destroy mxGPUArray
    mxGPUDestroyGPUArray(dm_dcf);

    return;


} 