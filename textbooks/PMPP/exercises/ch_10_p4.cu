#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Define constants
#define BLOCK_DIM 256
#define COARSE_FACTOR 4

// Original kernel
__global__ void CoarsenedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        sum += input[i + tile*BLOCK_DIM];
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// Modify the above kernel to perform a max reduction instead of a sum reduction.
// Solution:

__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    // init shared mem
    __shared__ float input_s[BLOCK_DIM];
    // init segment side coarse factor = how many elems you stack per thread
    // 2 for how many values to are summing (x + y)
    // blockDim.x * blockIdx.x for position
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    // i connects threads to each value stacked
    // t is just for thread indexing
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    // I dont think we need to great a sum value so I'll comment it out?
    // actually we will need this to send to shared so we'll change it from sum to max
    float max = input[i];
    // we should be able to use same logic here?
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        // take max between write location and other queried loc with fmaxf func
        max = fmaxf(max, input[i + tile*BLOCK_DIM]);
    }
    input_s[t] = max;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmaxf(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0) {
        *output = input_s[0];
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
    // Set up sizes
    const int numElements = BLOCK_DIM * COARSE_FACTOR * 2;
    const size_t size = numElements * sizeof(float);
    
    printf("Testing with %d elements\n", numElements);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float h_sumOutput = 0.0f;
    float h_maxOutput = 0.0f;
    
    // Initialize input data with random values between 0 and 100
    srand(42); // For reproducibility
    float expectedSum = 0.0f;
    float expectedMax = -INFINITY;
    
    for (int i = 0; i < numElements; i++) {
        h_input[i] = (float)(rand() % 100);
        expectedSum += h_input[i];
        expectedMax = fmaxf(expectedMax, h_input[i]);
    }
    
    printf("Expected sum: %.2f\n", expectedSum);
    printf("Expected max: %.2f\n", expectedMax);
    
    // Allocate device memory
    float *d_input, *d_sumOutput, *d_maxOutput;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sumOutput, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_maxOutput, sizeof(float)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Initialize outputs to 0
    float zero = 0.0f;
    CHECK_CUDA_ERROR(cudaMemcpy(d_sumOutput, &zero, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_maxOutput, &zero, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch sum reduction kernel
    printf("Running sum reduction kernel...\n");
    CoarsenedSumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_sumOutput);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch max reduction kernel
    printf("Running max reduction kernel...\n");
    CoarsenedMaxReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_maxOutput);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_sumOutput, d_sumOutput, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_maxOutput, d_maxOutput, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    printf("Sum reduction result: %.2f\n", h_sumOutput);
    printf("Max reduction result: %.2f\n", h_maxOutput);
    
    if (fabs(h_sumOutput - expectedSum) < 0.001f) {
        printf("PASSED! Sum reduction produced the correct result.\n");
    } else {
        printf("FAILED! Sum reduction result doesn't match the expected sum.\n");
        printf("Difference: %.2f\n", fabs(h_sumOutput - expectedSum));
    }
    
    if (fabs(h_maxOutput - expectedMax) < 0.001f) {
        printf("PASSED! Max reduction produced the correct result.\n");
    } else {
        printf("FAILED! Max reduction result doesn't match the expected max.\n");
        printf("Difference: %.2f\n", fabs(h_maxOutput - expectedMax));
    }
    
    // Clean up
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_sumOutput);
    cudaFree(d_maxOutput);
    
    return 0;
}