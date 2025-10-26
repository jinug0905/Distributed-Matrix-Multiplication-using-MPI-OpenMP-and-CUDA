#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"
#include <iostream>

extern void local_dgemm_cpu(LocalMatrix& C, const double* Arow, int Am, int An,
                            const double* Bcol, int Bm, int Bn);

void run_summa_cpu(int N, const Dist2D& d, bool do_verify){
  LocalMatrix A(N,N,d),B(N,N,d),C(N,N,d);
  A.initialize_A();
  B.initialize_B();
  C.zero();

  // Split rows/cols
  std::vector<int> row_sizes, row_offs, col_sizes, col_offs;
  split_sizes(N, d.P, row_sizes, row_offs);
  split_sizes(N, d.P, col_sizes, col_offs);

  // Local tile sizes
  const int m = A.l_rows;                 // rows
  const int n = B.l_cols;                 // cols

  for(int kblk = 0; kblk < d.P; ++kblk){
    // width of A panel, height of B panel where k of (m x "k") * ("k" * n)
    const int ksz = col_sizes[kblk];

    // -=-=-=-=-=-=-=-= A -=-=-=-=-=-=-=-=-
    // A-panel (m x ksz) by row
    const double* A_panel_ptr = nullptr;
    std::vector<double> A_panel_buf;

    if(d.myc == kblk){                  // root has exactly (m x ksz)
      A_panel_ptr = A.data.data();

    } else {                            // When it s not resize
      A_panel_buf.resize(m * ksz);
      A_panel_ptr = A_panel_buf.data();
    }

    MPI_Bcast(const_cast<double*>(A_panel_ptr), m*ksz, MPI_DOUBLE, kblk, d.row_comm);

    // -=-=-=-=-=-=-=-= B -=-=-=-=-=-=-=-=-
    // B-panel (ksz x n) by column
    const double* B_panel_ptr = nullptr;
    std::vector<double> B_panel_buf;

    if(d.myr == kblk){                   // root has exactly (ksz x n)
      B_panel_ptr = B.data.data();

    } else {                              // When it s not resize
      B_panel_buf.resize(ksz * n);      
      B_panel_ptr = B_panel_buf.data();
    }

    MPI_Bcast(const_cast<double*>(B_panel_ptr), ksz*n, MPI_DOUBLE, kblk, d.col_comm);

    // Local accumulation : C += A_panel * B_panel
    local_dgemm_cpu(C, A_panel_ptr, m, ksz, B_panel_ptr, ksz, n);
  }

  if(do_verify){
    std::vector<double> fullC;
    gather_matrix(C, N, d, fullC, /*root=*/0);
    verify_result(N, fullC);
  }
}