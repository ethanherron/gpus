#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Original kernel
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >=1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    } 
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}

// Solution
__global__ void MirrorConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (i >= blockDim.x - stride) {
            input[i] += input[i - stride];
        }
        __syncthreads();
    } 
    if(threadIdx.x == blockDim.x - 1) {
        *output = input[blockDim.x - 1];
    }
}

// Helper function for checking CUDA errors
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at line %d in file %s\n", \
               cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int BLOCK_SIZE = 256;
    const int ARRAY_SIZE = BLOCK_SIZE * 2; // Need to allocate enough for the stride
    
    // Allocate host memory
    float *h_input1 = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_input2 = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float h_output1 = 0.0f;
    float h_output2 = 0.0f;
    
    // Initialize input arrays
    float expectedSum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        h_input1[i] = (float)(i + 1);
        h_input2[i] = (float)(i + 1);
        expectedSum += (float)(i + 1);
    }
    // Make sure the rest is zeroed out
    for (int i = BLOCK_SIZE; i < ARRAY_SIZE; i++) {
        h_input1[i] = 0.0f;
        h_input2[i] = 0.0f;
    }
    
    printf("Expected sum: %.2f\n", expectedSum);
    
    // Allocate device memory
    float *d_input1, *d_input2, *d_output1, *d_output2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input1, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input2, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output1, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output2, sizeof(float)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input1, h_input1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input2, h_input2, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch original kernel
    printf("Running original reduction kernel...\n");
    ConvergentSumReductionKernel<<<1, BLOCK_SIZE>>>(d_input1, d_output1);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch mirrored kernel
    printf("Running mirrored reduction kernel...\n");
    MirrorConvergentSumReductionKernel<<<1, BLOCK_SIZE>>>(d_input2, d_output2);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_output1, d_output1, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_output2, d_output2, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    printf("Original kernel result: %.2f\n", h_output1);
    printf("Mirrored kernel result: %.2f\n", h_output2);
    
    if (fabs(h_output1 - expectedSum) < 0.001f && fabs(h_output2 - expectedSum) < 0.001f) {
        printf("PASSED! Both kernels produced the correct sum.\n");
    } else {
        printf("FAILED! Results don't match the expected sum.\n");
    }
    
    // Clean up
    free(h_input1);
    free(h_input2);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);
    
    return 0;
}

