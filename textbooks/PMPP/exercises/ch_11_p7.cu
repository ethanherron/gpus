// Define SECTION_SIZE - typically a power of 2 matching the block size
#define SECTION_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper function for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                       \
do {                                                                \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA Error: %s at line %d in file %s\n",            \
               cudaGetErrorString(err), __LINE__, __FILE__);        \
        exit(1);                                                    \
    }                                                               \
} while(0)

// original implementation (inclusive scan)
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    if(i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

// new implementation (exclusive scan)
__global__ void Kogge_Stone_exclusive_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For a proper exclusive scan, we need N-1 elements from the input
    // And we need to handle the Nth element separately
    
    // Initialize shared memory (exclusive scan shifts elements right)
    if(threadIdx.x == 0) {
        // First element is the identity (0 for addition)
        XY[threadIdx.x] = 0.0f;
    } else if(i-1 < N) {
        // Shift right: current position gets value from previous position in input
        XY[threadIdx.x] = X[i-1];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    
    // Perform the scan operation
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    
    // Output results
    if(i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

// CPU implementations for verification
void sequential_inclusive_scan(float *in, float *out, int N) {
    out[0] = in[0];
    for (int i = 1; i < N; i++) {
        out[i] = out[i-1] + in[i];
    }
}

void sequential_exclusive_scan(float *in, float *out, int N) {
    out[0] = 0.0f;
    for (int i = 1; i < N; i++) {
        out[i] = out[i-1] + in[i-1];
    }
}

// Verifies results against CPU implementation
bool verify_results(float *gpu_result, float *cpu_result, int N, bool is_inclusive) {
    const float epsilon = 1e-5;
    for (int i = 0; i < N; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            printf("%s scan error at index %d: GPU = %f, CPU = %f\n", 
                  is_inclusive ? "Inclusive" : "Exclusive", i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // Default to a small array to ensure we only use one block
    int N = 256; // Reduced from 512 to ensure single block usage
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    // Ensure N doesn't exceed maximum block size
    int threadsPerBlock = SECTION_SIZE; // Use the full section size
    if (N > threadsPerBlock) {
        printf("Warning: Input size %d exceeds maximum block size %d.\n", N, threadsPerBlock);
        printf("Reducing input size to %d elements.\n", threadsPerBlock);
        N = threadsPerBlock;
    }
    
    int blocksPerGrid = 1; // Always use one block
    
    printf("Testing Kogge-Stone scan with %d elements\n", N);
    
    // Allocate host memory
    float *h_input = (float*) malloc(N * sizeof(float));
    float *h_inclusive_output = (float*) malloc(N * sizeof(float));
    float *h_exclusive_output = (float*) malloc(N * sizeof(float));
    float *h_cpu_inclusive = (float*) malloc(N * sizeof(float));
    float *h_cpu_exclusive = (float*) malloc(N * sizeof(float));
    
    // Initialize input with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10);
    }
    
    // Compute CPU reference results
    sequential_inclusive_scan(h_input, h_cpu_inclusive, N);
    sequential_exclusive_scan(h_input, h_cpu_exclusive, N);
    
    // Allocate device memory
    float *d_input, *d_inclusive_output, *d_exclusive_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_inclusive_output, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_exclusive_output, N * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    printf("Grid dimensions: %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Launch inclusive scan kernel
    Kogge_Stone_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_inclusive_output, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch exclusive scan kernel
    Kogge_Stone_exclusive_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_exclusive_output, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_inclusive_output, d_inclusive_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_exclusive_output, d_exclusive_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool inclusive_correct = verify_results(h_inclusive_output, h_cpu_inclusive, N, true);
    bool exclusive_correct = verify_results(h_exclusive_output, h_cpu_exclusive, N, false);
    
    if (inclusive_correct) {
        printf("Inclusive scan test PASSED!\n");
    } else {
        printf("Inclusive scan test FAILED!\n");
    }
    
    if (exclusive_correct) {
        printf("Exclusive scan test PASSED!\n");
    } else {
        printf("Exclusive scan test FAILED!\n");
    }
    
    // Print first few elements of the arrays for verification
    printf("\nInput and scan results (first 10 elements):\n");
    printf("Index\tInput\tInclusive\tExclusive\n");
    int display_count = (N < 10) ? N : 10;
    for (int i = 0; i < display_count; i++) {
        printf("%d\t%.1f\t%.1f\t\t%.1f\n", i, h_input[i], h_inclusive_output[i], h_exclusive_output[i]);
    }
    
    // Free memory
    free(h_input);
    free(h_inclusive_output);
    free(h_exclusive_output);
    free(h_cpu_inclusive);
    free(h_cpu_exclusive);
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_inclusive_output));
    CHECK_CUDA_ERROR(cudaFree(d_exclusive_output));
    
    printf("\nAll tests completed.\n");
    
    return 0;
}