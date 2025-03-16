#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Define SECTION_SIZE - typically a power of 2 matching the block size
#define SECTION_SIZE 256

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

// Kernel 1: Kogge-Stone scan on each block and store last element in S
__global__ void Kogge_Stone_scan_with_block_sums(float *X, float *Y, float *S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if(i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    
    // Perform Kogge-Stone scan
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
    
    // Store result back to global memory
    if(i < N) {
        Y[i] = XY[threadIdx.x];
    }
    
    // Last thread in block stores the block sum to S array
    if(threadIdx.x == blockDim.x-1) {
        S[blockIdx.x] = XY[threadIdx.x];
    }
}

// Kernel 2: Kogge-Stone scan on the block sums array S
__global__ void Kogge_Stone_scan_block_sums(float *S, unsigned int num_blocks) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = threadIdx.x;
    
    // Load data into shared memory
    if(i < num_blocks) {
        XY[i] = S[i];
    } else {
        XY[i] = 0.0f;
    }
    
    // Perform Kogge-Stone scan
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(i >= stride)
            temp = XY[i] + XY[i-stride];
        __syncthreads();
        if(i >= stride)
            XY[i] = temp;
    }
    
    // Store result back to global memory
    if(i < num_blocks) {
        S[i] = XY[i];
    }
}

// Kernel 3: Add block sums to produce final scan result
__global__ void add_block_sums(float *Y, float *S, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Skip the first block (it already has correct values)
    if(i < N && blockIdx.x > 0) {
        // Add the previous block's scanned sum to this element
        Y[i] += S[blockIdx.x - 1];
    }
}

// CPU implementation for verification
void sequential_inclusive_scan(float *in, float *out, int N) {
    out[0] = in[0];
    for (int i = 1; i < N; i++) {
        out[i] = out[i-1] + in[i];
    }
}

// Verifies results against CPU implementation
bool verify_results(float *gpu_result, float *cpu_result, int N) {
    const float epsilon = 1e-5;
    for (int i = 0; i < N; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            printf("Scan error at index %d: GPU = %f, CPU = %f\n", 
                  i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // Default array size
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    printf("Testing segmented parallel scan with %d elements\n", N);
    
    // Calculate grid dimensions
    int threadsPerBlock = SECTION_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Grid dimensions: %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Allocate host memory
    float *h_input = (float*) malloc(N * sizeof(float));
    float *h_output = (float*) malloc(N * sizeof(float));
    float *h_cpu_result = (float*) malloc(N * sizeof(float));
    float *h_block_sums = (float*) malloc(blocksPerGrid * sizeof(float));
    
    // Initialize input with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10);
    }
    
    // Compute CPU reference result
    sequential_inclusive_scan(h_input, h_cpu_result, N);
    
    // Allocate device memory
    float *d_input, *d_output, *d_block_sums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Step 1: Run Kogge-Stone scan on each block and collect block sums
    Kogge_Stone_scan_with_block_sums<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_block_sums, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 2: Scan the block sums
    Kogge_Stone_scan_block_sums<<<1, threadsPerBlock>>>(d_block_sums, blocksPerGrid);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 3: Add the scanned block sums to each block's elements
    add_block_sums<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_block_sums, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_block_sums, d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correct = verify_results(h_output, h_cpu_result, N);
    
    if (correct) {
        printf("Segmented parallel scan test PASSED!\n");
    } else {
        printf("Segmented parallel scan test FAILED!\n");
    }
    
    // Print first and last few elements for verification
    printf("\nInput and scan results:\n");
    printf("Index\tInput\tGPU Result\tCPU Result\n");
    
    // Print first 10 elements
    int display_count = (N < 10) ? N : 10;
    for (int i = 0; i < display_count; i++) {
        printf("%d\t%.1f\t%.1f\t\t%.1f\n", i, h_input[i], h_output[i], h_cpu_result[i]);
    }
    
    // Print last 5 elements if N is large enough
    if (N > 20) {
        printf("...\n");
        for (int i = N - 5; i < N; i++) {
            printf("%d\t%.1f\t%.1f\t\t%.1f\n", i, h_input[i], h_output[i], h_cpu_result[i]);
        }
    }
    
    // Print block sums
    printf("\nBlock sums after scan:\n");
    for (int i = 0; i < blocksPerGrid; i++) {
        printf("Block %d sum: %.1f\n", i, h_block_sums[i]);
    }
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_cpu_result);
    free(h_block_sums);
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_block_sums));
    
    printf("\nAll tests completed.\n");
    
    return 0;
}
