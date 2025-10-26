#!/bin/bash
#SBATCH -A m5101
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH -N 4            
#SBATCH -n 64
#SBATCH -c 8
#SBATCH -J summa-cpu-scale
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load PrgEnv-gnu
make cpu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-5000}

echo "== CPU SCALING: N=${N}, OMP_NUM_THREADS=${OMP_NUM_THREADS} =="

echo "-=- P=4 (2x2) -=-"
srun -N 1 -n 4  -c 8 --cpu-bind=cores ./summa --N ${N} --verify

echo "-=- P=16 (4x4) -=-"
srun -N 1 -n 16 -c 8 --cpu-bind=cores ./summa --N ${N} --verify

echo "-=- P=64 (8x8) -=-"
srun -N 4 -n 64 -c 8 --cpu-bind=cores ./summa --N ${N} --verify
