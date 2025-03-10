#include <iostream>
#include <cuda_runtime.h>

#define N 16

// kernel for naive matmul
__global__ void matrixMul(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}


int main() {
    // allocate and init host mem
    size_t size = N * N * sizeof(int);
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    // simple init by filling a and b with 1s
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    // allocate device mem
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy host data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // config and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y -1) / blockDim.y);
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // copy results back to host from device
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print results
    std::cout << "C[0..3] = ";
    for (int i = 0; i < 4; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // clean up on this disaster
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
