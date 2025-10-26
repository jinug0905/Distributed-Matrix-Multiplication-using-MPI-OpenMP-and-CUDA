#include "verify.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>

#define tol 1e-5

void gather_matrix(const LocalMatrix& C, int N, const Dist2D& d,
                   std::vector<double>& CTmp, int root){
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Local values
    int lrows = C.l_rows;
    int lcols = C.l_cols;
    int roff  = C.row_off;
    int coff  = C.col_off;
    int count = lrows * lcols;

    // Gather locals to root
    std::vector<int> all_lrows, all_lcols, all_roff, all_coff, all_counts;
    if (world_rank == root) {
        all_lrows.resize(world_size);
        all_lcols.resize(world_size);
        all_roff.resize(world_size);
        all_coff.resize(world_size);
        all_counts.resize(world_size);
    }

    MPI_Gather((void*)&lrows, 1, MPI_INT, world_rank==root? all_lrows.data():nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather((void*)&lcols, 1, MPI_INT, world_rank==root? all_lcols.data():nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather((void*)&roff,  1, MPI_INT, world_rank==root? all_roff.data(): nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather((void*)&coff,  1, MPI_INT, world_rank==root? all_coff.data(): nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather((void*)&count, 1, MPI_INT, world_rank==root? all_counts.data():nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Compute displacements and total size for a packed gather
    std::vector<int> displs;
    int total_elems = 0;

    if (world_rank == root) {
        displs.resize(world_size);
        int cnt = 0;

        for (int r = 0; r < world_size; ++r) {
            displs[r] = cnt;
            cnt += all_counts[r];
        }

        total_elems = cnt;
    }

    // Gather tiles into root
    std::vector<double> packed;
    if (world_rank == root)
        packed.resize(total_elems);

    MPI_Gatherv((void*)C.data.data(), count, MPI_DOUBLE,
                world_rank==root? packed.data():nullptr,
                world_rank==root? all_counts.data():nullptr,
                world_rank==root? displs.data():nullptr,
                MPI_DOUBLE, root, MPI_COMM_WORLD);

    // Reassemble N x N on root
    if (world_rank == root) {
        CTmp.assign(N * N, 0.0);

        for (int r = 0; r < world_size; ++r) {
            int mr = all_lrows[r], nr = all_lcols[r];
            int r0 = all_roff[r],  c0 = all_coff[r];
            double* tile = packed.data() + displs[r];

            for (int i = 0; i < mr; ++i) {
                std::memcpy(CTmp.data() + (r0 + i) * N + c0,
                                tile + i * nr,
                                nr * sizeof(double));
            }
        }

    } else {
        CTmp.clear();
    }
}

void verify_result(int N, const std::vector<double>& fullC){
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) return;  // Only root can print fullC

    // A and B for check
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i * N + j] = static_cast<double>(i + j);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B[i * N + j] = static_cast<double>(i - j);

    // Reference Cref = A * B
    std::vector<double> Cref(N * N, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Cref[i * N + j] = 0.0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            
            Cref[i * N + j] = sum;
        }
    }

    // difference check 
    double max_abs = 0.0;

    #pragma omp parallel for reduction(max:max_abs) schedule(static)
    for (int idx = 0; idx < N * N; ++idx) {
        int d = std::abs(fullC[idx] - Cref[idx]);
        if (d > max_abs)
            max_abs = d;
    }

    std::cout << "[VERIFY] max |C - Cref| = " << max_abs << '\n';
    if (max_abs <= tol) {
        std::cout << "[VERIFY] PASS (tol=" << tol << ")\n";
    } else {
        std::cout << "[VERIFY] FAIL (tol=" << tol << ")\n";
    }
}