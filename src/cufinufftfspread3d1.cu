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

/**
 * Host code
 *run in matlab: im = cufinufftfspread3d1(x,y,z,k,isign,eps,Nx,Ny,Nz,M,Ntrans)
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    clock_t start, end;
    double cpu_time_used;

    /* Declare all variables.*/
    float *x;          // frequency domain dimension 1 coordinates
    float *y;          // frequency domain dimension 2 coordinates
    float *z;          // frequency domain dimension 3 coordinates
    float _Complex *k; // Pointer to complex frequency domain strengths, on host
    int isSign;     //Whether to use + or - sign for complex exponential
    float eps;      //NUFFT precision
    int64_t Nx;     //Size of first dimension of image
    int64_t Ny;     //Size of second dimension of image
    int64_t Nz;     //Size of third dimension of image
    size_t M;      // number of NU points
    size_t Ntrans; // Number of stacked transforms
    //int GPUDeviceIndex; //Which GPU to use

    char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const *const errMsg = "Invalid input to MEX file.";
    char const *const instruction = "Usage:\n im = cufinufftfspread3d1(x,y,z,k,isign,eps,Nx,Ny,Nz,M,Ntrans).";

    

    char errorMessage[1024]; // remember to ensure this buffer is large enough for your message
    // Check input number
    if (nrhs != 11)
    {
        snprintf(errorMessage, sizeof(errorMessage), "Exactly 11 inputs required. \n%s", instruction);
        mexErrMsgIdAndTxt("cufinufftfspread3d1:InputError", errorMessage);
    }

    // Check input type
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]) || !mxIsSingle(prhs[3]))
    {   
        snprintf(errorMessage, sizeof(errorMessage), "Inputs 1-4 must be single. \n%s", instruction);
        mexErrMsgIdAndTxt("cufinufftfspread3d1:InputError", errorMessage);
    }

    // Assign Pointers and value
    x = mxGetSingles(prhs[0]);
    y = mxGetSingles(prhs[1]);
    z = mxGetSingles(prhs[2]);
    k = (float _Complex *)mxGetComplexSingles(prhs[3]);
    
    isSign = (int)mxGetScalar(prhs[4]);
    eps = (float)mxGetScalar(prhs[5]);
    Nx = (int64_t)mxGetScalar(prhs[6]);
    Ny = (int64_t)mxGetScalar(prhs[7]);
    Nz = (int64_t)mxGetScalar(prhs[8]);
    M = (size_t)mxGetScalar(prhs[9]);
    Ntrans = (size_t)mxGetScalar(prhs[10]);

    // Initialize the MathWorks GPU API.
    mxInitGPU();
    //cudaSetDevice(GPUDeviceIndex); //Set which GPU to run. This should strictly matach the one being used by MATLAB

    // Create output gpuArray
    mwSize out_dimensions[4] = {(size_t)Nx * 2, (size_t)Ny * 2, (size_t)Nz * 2, Ntrans};
    const mwSize * ndims = out_dimensions;
    mxGPUArray *d_Phi2Cartesian;
    mwSize dims[4] = {(size_t)Nx * 2, (size_t)Ny * 2, (size_t)Nz * 2, Ntrans};
    d_Phi2Cartesian = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

    // Create modes
    int64_t modes[3] = {Nx, Ny, Nz};

    // Create default cufinufft plan object
    cufinufftf_plan plan;
    cufinufft_opts opts;

    // Create on device variables
    float *d_x, *d_y, *d_z;
    cuFloatComplex *d_k, *d_fw;

    // Allocate device memories
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_z, M * sizeof(float));
    cudaMalloc(&d_k, M * Ntrans * sizeof(cuFloatComplex));
    d_fw = (cuFloatComplex *)mxGPUGetData(d_Phi2Cartesian);


    // Copy values to device
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, M * Ntrans * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    //mexPrintf("CUDA MEM COPY Complete\n");

    // make cufinufft plan
    /*
     * int cufinufftf_makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntr, float tol, cufinufftf_plan *plan, cufinufft_opts *opts)
     */
    cufinufft_default_opts(&opts); //make default options

    opts.upsampfac = 2.0;

    cufinufftf_makeplan(1, 3, modes, isSign, Ntrans, eps, &plan, &opts);


    //mexPrintf("cufinufftf_makeplan Complete\n");


    // Set points
    /*
     * int cufinufftf_setpts(cufinufftf_plan plan, int M, float* x, float* y, float* z, int N, float* s, float* t, float *u)
     */
    cufinufftf_setpts(plan, M, d_x, d_y, d_z, 0, NULL, NULL, NULL);
    //mexPrintf("cufinufftf_setpts Complete\n");





    /************ EXECUTE SPREADING AND EXTRACT************/
    start = clock();
    cufinufftf_extract(plan, d_k, d_fw);
    end = clock();
    double timeExecute = ((double)(end-start))/ CLOCKS_PER_SEC;
    mexPrintf("cufinufftf 3D spreading execution Time (s): %.5f\n", timeExecute);

    //mexPrintf("cufinufftf_execute Complete\n");
    /************ EXECUTE END *****************************/




    

    cuFloatComplex *d_k_spreaded = (cuFloatComplex *)mxGPUGetData(d_Phi2Cartesian);

    // Copy d_fw to d_k_spreaded
    cudaMemcpy(d_k_spreaded, d_fw, (size_t)Nx * 2 * (size_t)Ny * 2 * (size_t)Nz * 2 * Ntrans * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    

    cufinufftf_destroy(plan);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_k);
    //mexPrintf("cufinufftf_cleancup Complete\n");

    //Create output 
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(d_Phi2Cartesian);

    //Destroy mxGPUArray
    mxGPUDestroyGPUArray(d_Phi2Cartesian);

    return;


} 