# Distributed Matrix Multiplication using MPI, OpenMP, and CUDA

Author: Jinug Lee  
Course: CSCE 654 — Supercomputing, Texas A&M University  
Hardware: Perlmutter, NERSC. Multi-node HPC cluster (CPU + NVIDIA GPUs)  
Date: Fall 2025

---

## 📌 Overview

This project implements **distributed dense matrix multiplication** using the **SUMMA (Scalable Universal Matrix Multiplication Algorithm)**.  
It demonstrates hybrid parallel programming across **MPI**, **OpenMP**, and **CUDA**:

- **MPI** — Distributes global matrices across a `√P × √P` process grid  
- **OpenMP** — Accelerates local DGEMM computation on multi-core CPUs  
- **CUDA** — Offloads compute-heavy GEMM operations to GPUs  
- **Verification** — Reconstructs global results and checks accuracy vs. reference CPU implementation  

The system efficiently scales matrix multiplication to large problem sizes (e.g., N ≈ 5000).

---

## Key Concepts Implemented

| Component | Technology | Description |
|----------|------------|-------------|
| Distributed Matrix Decomposition | MPI | Assigns local sub-blocks of A, B, and C to each process |
| Communication Pattern | MPI_Bcast | Broadcasting row/column blocks per iteration of SUMMA |
| Local Compute | OpenMP threads | CPU DGEMM with loop tiling (Homework 1 kernel) |
| GPU Acceleration | CUDA kernels | GPU DGEMM (Homework 2 kernel) |
| Correctness | MPI_Gather + Unified compute | Validates global C vs. CPU reference |
