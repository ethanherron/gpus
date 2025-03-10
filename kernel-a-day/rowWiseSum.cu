#include <iostream>
#include <cuda_runtime.h>

#define N 16

// kernel for row-wise summation
__global__ void rowWiseSum(float* A, float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col];
        }
        B[row] = sum;
    }
}


int main() {
    // allocate and init host mem
    size_t size = N * N * sizeof(float);
    size_t size_B = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size_B);

    // simple init by filling a and b with 1s
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
    }

    // allocate device mem
    float* d_A = nullptr;
    float* d_B = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size_B);

    // copy host data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // config and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y -1) / blockDim.y);
    rowWiseSum<<<gridDim, blockDim>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    // copy results back to host from device
    cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost);

    // print results
    std::cout << "After rowWiseSum (row 1):  " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;

    // clean up on this disaster
    free(h_B);
    cudaFree(d_B);

    return 0;
}
