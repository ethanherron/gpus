#include <iostream>
#include <cuda_runtime.h>

#define N 16

// kernel for 2dconv
__global__ void conv2d(const float* A, const float* kernel, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 1 || row >= n-1 || col < 1 || col >= n-1) {
        return;
    }

    float sum = 0.0f;
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j++) {
            int idxA = (row + i) * n + (col + j);
            int idxK = (i + 1) * 3 + (j + 1);
            sum += A[idxA] * kernel[idxK];
        }
    }
    int idxC = (row-1) * (n-2) + (col-1);
    C[idxC] = sum;
}


int main() {
    // allocate and init host mem
    size_t size = N * N * sizeof(float);
    size_t size_C = (N - 2) * (N - 2) * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_C = (float*)malloc(size_C);

    // simple init by filling a and b with 1s
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
    }

    float h_kernel[9] = {
        0.0f, 1.0f, 0.0f,
        1.0f, 2.0f, 1.0f,
        0.0f, 1.0f, 0.0f
    };

    // allocate device mem
    float *d_A, *d_kernel, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    cudaMalloc(&d_C, size_C);

    // copy host data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // config and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y -1) / blockDim.y);
    conv2d<<<gridDim, blockDim>>>(d_A, d_kernel, d_C, N);

    // copy results back to host from device
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // print results
    std::cout << "C[0..3] = ";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // clean up on this disaster
    free(h_A); free(h_C);
    cudaFree(d_A); cudaFree(d_C); cudaFree(d_kernel);

    return 0;
}
