__global__ void tanh(float* A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        float x = A[i];
        float exp_pos = expf(x);
        float exp_neg = expf(-x);
        A[i] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}