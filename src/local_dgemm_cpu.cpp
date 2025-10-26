#include "localmatrix.hpp"

// OUT = A(m x k) * B(k x n) (OpenMP)
void dgemm_naive_to(const double* A, const double* B, double* OUT, int m, int n, int k){
    // zero OUT
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            OUT[i * n + j] = 0.0;

    // OUT = A * B
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }

            OUT[i * n + j] = sum;
        }
    }
}

// C(m x n) += CTmp(m x n)
void addMatUpdate(double* C, const double* Ctmp, int m, int n){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C[i*n + j] += Ctmp[i*n + j];
        }
    }
}

// Write your local DGEMM usign OpenMP
// SUMMA's local kernel: C += Arow * Bcol
void local_dgemm_cpu(LocalMatrix& C, const double* Arow, int Am, int An,
                                    const double* Bcol, int Bm, int Bn){
    if(!(Am == C.l_rows && Bn == C.l_cols && An == Bm)) {
        std::cout << "Dimension size not match" << std::endl;
        return;
    }

    std::vector<double> tmp(Am * Bn);
    dgemm_naive_to(Arow, Bcol, tmp.data(), Am, Bn, An);
    addMatUpdate(C.data.data(), tmp.data(), Am, Bn);
}