#include "localmatrix.hpp"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

__global__ void dgemm_naive_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K){
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)
    if (row >= M || col >= N) return;

    double val = 0.0;
    for (int k = 0; k < K; ++k) 
        val += A[row * K + k] * B[k * N + col];
    
    C[row * N + col] = val;
}

__global__ void add_inplace_kernel(double* __restrict__ C,
                                   const double* __restrict__ CTmp,
                                   int M, int N){
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)
    if (row >= M || col >= N) return;
    const int idx = row * N + col;
    C[idx] += CTmp[idx];
}

// Host wrapper: OUT = A * B (overwrites OUT)
void dgemm_naive_gpu(const double* dA, const double* dB, double* dOUT,
                            int M, int N, int K, cudaStream_t stream = 0){
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    dgemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dOUT, M, N, K);
}

void addMatUpdate(double* dC, const double* dCTmp, int M, int N,
                            cudaStream_t stream = 0){
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    add_inplace_kernel<<<grid, block, 0, stream>>>(dC, dCTmp, M, N);
}


// Write your local DGEMM usign Cuda
extern "C"
void local_dgemm_gpu(double* dC, int Cm, int Cn,
                     const double* dA, int Am, int An,
                     const double* dB, int Bm, int Bn,
                     cudaStream_t stream = 0){
    // dimension check
    if (!(Am == Cm && Bn == Cn && An == Bm)) {
        std::cout << "Dimension size not match" << std::endl;
        return;
    }

    double* dTmp = nullptr;
    cudaMalloc(&dTmp, Cm * Cn * sizeof(double));

    // Compute dTmp = dA * dB
    dgemm_naive_gpu(dA, dB, dTmp, Cm, Cn, An, stream);

    // Accumulate: dC += dTmp
    addMatUpdate(dC, dTmp, Cm, Cn, stream);

    // Free temporary buffer
    cudaFree(dTmp);
}

#endif