#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

// Modify the kernel above to work for an arbitrary length input that is not necessarily a mulitple of COARSE_FACTOR*2*blockDim.x.
// Add an extra parameter N to the kernel that represents the length of the input. 
// Solution

__global__ void ExtendedCoarsenedSumReductionKernel(float* input, float* output, int N) {
    // init shared mem
    __shared__ float input_s[BLOCK_DIM];
    // define segment which is actually defining the global index of values for each thread
    // we might need to change this to accomodate variable length?
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0; // start with 0 instead of potentially invalid data
    if (i < N) {
        sum = input[i];
    }
    // here we to for loop over "stacked" elems in each thread
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        // now I think we'll need an if-else here where if does what we had before
        // and else will stack up N-segment leftover values?
        // If we're careful with how we stack them up we can keep the rest of the code the same?
        unsigned int idx = i + tile*BLOCK_DIM;
        if (idx < N) { // avoid OB
            sum += input[idx];
        }
    }
    // now that we've reduced input far enough to store in shared we move to shared.
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
    // Test with both multiple and non-multiple sizes
    int sizes[] = {
        COARSE_FACTOR*2*BLOCK_DIM,           // Exact multiple - should work with both kernels
        COARSE_FACTOR*2*BLOCK_DIM + 100,     // Non-multiple - should only work with extended kernel
        COARSE_FACTOR*2*BLOCK_DIM*2 - 50     // Another non-multiple size
    };
    
    for (int test = 0; test < 3; test++) {
        int N = sizes[test];
        printf("\n=== Testing with array size %d ===\n", N);
        
        // Allocate host memory
        float *h_input = (float*)malloc(N * sizeof(float));
        float h_output_original = 0.0f;
        float h_output_extended = 0.0f;
        
        // Initialize input array
        float expectedSum = 0.0f;
        for (int i = 0; i < N; i++) {
            h_input[i] = (float)(i % 10);  // Simple pattern of values
            expectedSum += h_input[i];
        }
        
        printf("Expected sum: %.2f\n", expectedSum);
        
        // Allocate device memory
        float *d_input, *d_output_original, *d_output_extended;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_original, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_extended, sizeof(float)));
        
        // Copy data from host to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_output_original, 0, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_output_extended, 0, sizeof(float)));
        
        // Calculate grid dimensions
        int numBlocks = (N + (BLOCK_DIM * COARSE_FACTOR * 2 - 1)) / (BLOCK_DIM * COARSE_FACTOR * 2);
        
        // Launch original kernel (only for the exact multiple size)
        if (test == 0) {
            printf("Running original reduction kernel...\n");
            CoarsenedSumReductionKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output_original);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        
        // Launch extended kernel
        printf("Running extended reduction kernel...\n");
        ExtendedCoarsenedSumReductionKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output_extended, N);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy results back to host
        if (test == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(&h_output_original, d_output_original, sizeof(float), cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_ERROR(cudaMemcpy(&h_output_extended, d_output_extended, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Verify results
        if (test == 0) {
            printf("Original kernel result: %.2f\n", h_output_original);
        }
        printf("Extended kernel result: %.2f\n", h_output_extended);
        
        float tolerance = 0.001f * expectedSum;  // Allow for floating point rounding
        
        if (test == 0) {
            if (fabs(h_output_original - expectedSum) < tolerance) {
                printf("Original kernel: PASSED\n");
            } else {
                printf("Original kernel: FAILED (expected %.2f, got %.2f)\n", 
                       expectedSum, h_output_original);
            }
        }
        
        if (fabs(h_output_extended - expectedSum) < tolerance) {
            printf("Extended kernel: PASSED\n");
        } else {
            printf("Extended kernel: FAILED (expected %.2f, got %.2f)\n", 
                   expectedSum, h_output_extended);
        }
        
        // Clean up
        free(h_input);
        cudaFree(d_input);
        cudaFree(d_output_original);
        cudaFree(d_output_extended);
    }
    
    return 0;
}