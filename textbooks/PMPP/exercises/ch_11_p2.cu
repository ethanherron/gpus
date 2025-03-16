#include "stdio.h"
#include "stdlib.h"
#include "cuda_runtime.h"
#include "math.h"

#define SECTION_SIZE 1024
#define ARRAY_SIZE 1024  // Reduced to match block size
#define BLOCK_SIZE 1024  // Threads per block

// original implementation
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

// implementation with double buffering
__global__ void Kogge_Stone_scan_kernel_dbuffer(float *X, float *Y, unsigned int N) {
    // create the two buffers in shared instead of single XY
    __shared__ float buffer_A[SECTION_SIZE];
    __shared__ float buffer_B[SECTION_SIZE];

    // pointers for ping-pong buffering
    float* read_buff;
    float* write_buff;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // init load into buffer_A
    if(i < N) {
        buffer_A[threadIdx.x] = X[i];
    } else {
        buffer_A[threadIdx.x] = 0.0f;
    }

    // init buffer pointers
    read_buff = buffer_A;
    write_buff = buffer_B;

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        // Keep the first __syncthreads() from the original implementation
        __syncthreads();
        
        // read from read_buff and write to write_buff
        if(threadIdx.x >= stride) {
            write_buff[threadIdx.x] = read_buff[threadIdx.x] + read_buff[threadIdx.x - stride];
        } else {
            write_buff[threadIdx.x] = read_buff[threadIdx.x];
        }

        // No need for a second __syncthreads() here since we're using double buffering
        // Directly swap buffers for next iteration
        float* temp = read_buff;
        read_buff = write_buff;
        write_buff = temp;
    }

    // write final result from the final read_buff
    if(i < N) {
        Y[i] = read_buff[threadIdx.x];
    }
}

// CPU implementation for verification
void cpu_scan(float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for (int i = 1; i < N; i++) {
        Y[i] = Y[i-1] + X[i];
    }
}

// Verify results
bool verify_results(float *A, float *B, unsigned int N) {
    const float epsilon = 1e-5;
    for (int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Allocate host memory
    float *h_input = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_output1 = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_output2 = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_reference = (float*)malloc(ARRAY_SIZE * sizeof(float));

    // Initialize input data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)(rand() % 10);
    }

    // Print first few and last few input elements
    printf("Input data (first 10 elements):            ");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        printf("%.1f ", h_input[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_input, *d_output1, *d_output2;
    cudaMalloc((void**)&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output1, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output2, ARRAY_SIZE * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Using a single block for both kernels
    dim3 gridSize(1);  // One block only
    dim3 blockSize(BLOCK_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds1 = 0, milliseconds2 = 0;

    // Run original kernel and measure time
    cudaEventRecord(start);
    Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output1, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds1, start, stop);

    // Run double-buffered kernel and measure time
    cudaEventRecord(start);
    Kogge_Stone_scan_kernel_dbuffer<<<gridSize, blockSize>>>(d_input, d_output2, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds2, start, stop);

    // Copy results back to host
    cudaMemcpy(h_output1, d_output1, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference solution on CPU
    cpu_scan(h_input, h_reference, ARRAY_SIZE);

    // Print first few output elements from each implementation
    printf("CPU scan (first 10 elements):               ");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        printf("%.1f ", h_reference[i]);
    }
    printf("\n");

    printf("Original kernel (first 10 elements):        ");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        printf("%.1f ", h_output1[i]);
    }
    printf("\n");

    printf("Double-buffered kernel (first 10 elements): ");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
        printf("%.1f ", h_output2[i]);
    }
    printf("\n");

    // Verify results
    bool original_correct = verify_results(h_output1, h_reference, ARRAY_SIZE);
    bool dbuffer_correct = verify_results(h_output2, h_reference, ARRAY_SIZE);

    // Print results
    printf("Original kernel: %s, Time: %.3f ms\n", 
           original_correct ? "CORRECT" : "INCORRECT", milliseconds1);
    printf("Double-buffered kernel: %s, Time: %.3f ms\n", 
           dbuffer_correct ? "CORRECT" : "INCORRECT", milliseconds2);
    
    if (original_correct && dbuffer_correct) {
        printf("Speedup: %.2fx\n", milliseconds1 / milliseconds2);
    }

    // Free memory
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_reference);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}