#!/bin/bash
#SBATCH -A m5101
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 1          
#SBATCH -n 4              
#SBATCH -c 8        
#SBATCH -J summa-verify
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load PrgEnv-gnu

make cpu

# 8 threads
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-1000} 

echo "-=-=- CPU VERIFY -=-=-"
echo "N=${N}  P=4  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
srun -n 4 -c 8 --cpu-bind=cores ./summa --N ${N} --verify