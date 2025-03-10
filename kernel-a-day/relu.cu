#include <iostream>
#include <cuda_runtime.h>

#define N 16

// kernel for relu
__global__ void relu(float* A, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}


int main() {
    // allocate and init host mem
    size_t size = N * N * sizeof(float);
    float* h_A = (float*)malloc(size);

    // simple init by filling a and b with 1s
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (i % 2 == 0) ? (float)i : -(float)i;
    }

    // allocate device mem
    float* d_A = nullptr;
    cudaMalloc(&d_A, size);

    // copy host data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // config and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y -1) / blockDim.y);
    relu<<<gridDim, blockDim>>>(d_A, N);
    cudaDeviceSynchronize();

    // copy results back to host from device
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // print results
    std::cout << "After relu (row 1):  " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << std::endl;

    // clean up on this disaster
    free(h_A);
    cudaFree(d_A);

    return 0;
}
