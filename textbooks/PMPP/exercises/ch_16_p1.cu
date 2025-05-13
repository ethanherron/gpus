#include <stdio.h>   // For printf, stderr
#include <stdlib.h>  // For malloc, free
#include <math.h>    // For fabs
#include <cuda_runtime.h>  // For CUDA functions

// implement a pooling layer lets do mean pooling
// For simplicity we assume square images and square pooling kernel
__global__ void mean_pooling(int C, int H_in, int W_in, int K, float *X, float *Y) {
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;

    int H_out = H_in / K;
    int W_out = W_in / K;

    int h_in = 0;
    int w_in = 0;

    int h_in_start = 0;
    int w_in_start = 0;

    if (h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
         // init input array idx
         h_in_start = h_out * K;
         w_in_start = w_out * K;

         // sum over patch
         for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                h_in = h_in_start + i;
                w_in = w_in_start + j;
                if (h_in < H_in && w_in < W_in) { 
                    sum += X[c * H_in * W_in + h_in * W_in + w_in];  
                }
            }
         }

        Y[c * H_out * W_out + h_out * W_out + w_out] = sum / (K*K);
    }
}

// CPU implementation of mean pooling for verification
void cpu_mean_pooling(int C, int H_in, int W_in, int K, float *X, float *Y) {
    int H_out = H_in / K;
    int W_out = W_in / K;
    
    for (int c = 0; c < C; c++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                float sum = 0.0f;
                int h_in_start = h_out * K;
                int w_in_start = w_out * K;
                
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        int h_in = h_in_start + i;
                        int w_in = w_in_start + j;
                        if (h_in < H_in && w_in < W_in) {
                            sum += X[c * H_in * W_in + h_in * W_in + w_in];
                        }
                    }
                }
                
                Y[c * H_out * W_out + h_out * W_out + w_out] = sum / (K*K);
            }
        }
    }
}

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(1); \
    } \
}

// Helper function to check if results match
bool verify_results(float *cpu_output, float *gpu_output, int size) {
    const float epsilon = 1e-5;
    bool match = true;
    
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_output[i] - gpu_output[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, cpu_output[i], gpu_output[i]);
            match = false;
            break;
        }
    }
    
    return match;
}

int main(int argc, char **argv) {
    // Set test parameters
    int C = 3;             // Number of channels
    int H_in = 32;         // Input height
    int W_in = 32;         // Input width
    int K = 2;             // Pooling kernel size
    int H_out = H_in / K;  // Output height
    int W_out = W_in / K;  // Output width
    
    // Allow command line override
    if (argc > 1) H_in = W_in = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) C = atoi(argv[3]);
    
    // Recalculate output dimensions
    H_out = H_in / K;
    W_out = W_in / K;
    
    printf("Testing mean pooling with:\n");
    printf("  Input: %dx%dx%d\n", C, H_in, W_in);
    printf("  Pooling size: %dx%d\n", K, K);
    printf("  Output: %dx%dx%d\n", C, H_out, W_out);
    
    // Ensure input dimensions are divisible by K
    if (H_in % K != 0 || W_in % K != 0) {
        printf("Error: Input dimensions must be divisible by pooling size K.\n");
        return 1;
    }
    
    // Allocate host memory
    size_t input_size = C * H_in * W_in * sizeof(float);
    size_t output_size = C * H_out * W_out * sizeof(float);
    
    float *h_input = (float*)malloc(input_size);
    float *h_output_cpu = (float*)malloc(output_size);
    float *h_output_gpu = (float*)malloc(output_size);
    
    // Initialize input with pattern data
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H_in; h++) {
            for (int w = 0; w < W_in; w++) {
                // Create a recognizable pattern for debugging
                h_input[c * H_in * W_in + h * W_in + w] = c * 100 + h + w / 10.0f;
            }
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // Execute CPU version
    cpu_mean_pooling(C, H_in, W_in, K, h_input, h_output_cpu);
    
    // Set up grid and block dimensions for GPU kernel
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (W_out + blockDim.x - 1) / blockDim.x,
        (H_out + blockDim.y - 1) / blockDim.y,
        C
    );
    
    printf("Grid dimensions: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("Block dimensions: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    
    // Launch kernel
    mean_pooling<<<gridDim, blockDim>>>(C, H_in, W_in, K, d_input, d_output);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool success = verify_results(h_output_cpu, h_output_gpu, C * H_out * W_out);
    
    if (success) {
        printf("Test PASSED! CPU and GPU results match.\n");
    } else {
        printf("Test FAILED! Results don't match.\n");
    }
    
    // Print sample results for visual inspection
    printf("\nSample results (first channel):\n");
    printf("  Index  |  CPU Result  |  GPU Result\n");
    printf("---------|--------------|--------------\n");
    
    for (int i = 0; i < min(10, H_out * W_out); i++) {
        printf("  %5d  |  %10.4f  |  %10.4f\n", 
               i, h_output_cpu[i], h_output_gpu[i]);
    }
    
    // Clean up
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return 0;
}
