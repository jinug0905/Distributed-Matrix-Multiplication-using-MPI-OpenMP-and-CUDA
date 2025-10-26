#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

extern "C"
void local_dgemm_gpu(double* dC,int Cm,int Cn,
                            const double* dA,int Am,int An,
                            const double* dB,int Bm,int Bn,
                            cudaStream_t stream);

// GPU SUMMA
void run_summa_gpu(int N,const Dist2D& d, bool do_verify){
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  LocalMatrix A(N,N,d), B(N,N,d), C(N,N,d);
  A.initialize_A();
  B.initialize_B();
  C.zero();

  // Split rows/cols
  std::vector<int> row_sizes, row_offs, col_sizes, col_offs;
  split_sizes(N, d.P, row_sizes, row_offs);
  split_sizes(N, d.P, col_sizes, col_offs);

  // Local tile sizes
  const int m = A.l_rows;             // rows for A/C
  const int n = B.l_cols;             // cols for B/C

  // Initialize device buffer for C
  double* dC = nullptr;
  const size_t bytesC = m * n * sizeof(double);

  cudaMalloc(&dC, bytesC);
  cudaMemset(dC, 0, bytesC);

  // CUDA stream
  cudaStream_t stream; cudaStreamCreate(&stream);


  for(int kblk = 0; kblk < d.P; ++kblk){
    // width of A panel, height of B panel where k of (m x "k") * ("k" * n)
    const int ksz = col_sizes[kblk];

    // -=-=-=-=-=-=-=-= A -=-=-=-=-=-=-=-=-
    // A-panel (m x ksz) by row
    const double* A_panel_host_ptr = nullptr;
    std::vector<double> A_panel_host_buf;

    if (d.myc == kblk) {
      A_panel_host_ptr = A.data.data();

    } else {
      A_panel_host_buf.resize(m * ksz);
      A_panel_host_ptr = A_panel_host_buf.data();
    }

    MPI_Bcast(const_cast<double*>(A_panel_host_ptr), m*ksz, MPI_DOUBLE, kblk, d.row_comm);

    // -=-=-=-=-=-=-=-= B -=-=-=-=-=-=-=-=-
    // B-panel (ksz x n) by column
    const double* B_panel_host_ptr = nullptr;
    std::vector<double> B_panel_host_buf;

    if (d.myr == kblk) {
      B_panel_host_ptr = B.data.data();

    } else {
      B_panel_host_buf.resize(ksz * n);
      B_panel_host_ptr = B_panel_host_buf.data();
    }

    MPI_Bcast(const_cast<double*>(B_panel_host_ptr), ksz*n, MPI_DOUBLE, kblk, d.col_comm);

    // Local accumulation : C += A_panel * B_panel
    // Copy panels to device
    double *dA = nullptr, *dB = nullptr;
    const size_t bytesA = m * ksz * sizeof(double);
    const size_t bytesB = ksz * n * sizeof(double);
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMemcpyAsync(dA, A_panel_host_ptr, bytesA, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, B_panel_host_ptr, bytesB, cudaMemcpyHostToDevice, stream);

    // Local GPU DGEMM
    local_dgemm_gpu(dC, m, n, dA, m, ksz, dB, ksz, n, stream);

    // Cleanup panel device buffers
    cudaFree(dA);
    cudaFree(dB);
  }

  // Copy device C back to host C
  cudaMemcpyAsync(C.data.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Clean up
  cudaFree(dC);
  cudaStreamDestroy(stream);

  if(do_verify){
    std::vector<double> fullC;
    gather_matrix(C, N, d, fullC, /*root=*/0);
    verify_result(N, fullC);
  }
}

#endif
