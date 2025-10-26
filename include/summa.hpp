#pragma once
#include "common.hpp"
void run_summa_cpu(int N,const Dist2D& d, bool do_verify);

// Added for cpu execution
#ifdef ENABLE_CUDA
void run_summa_gpu(int N, const Dist2D& d, bool do_verify);
#endif
