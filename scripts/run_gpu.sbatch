#!/bin/bash
#SBATCH -A m5101
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH -N 4          
#SBATCH -n 16
#SBATCH -c 8
#SBATCH --gpus-per-task=1   
#SBATCH -J summa-gpu-scale
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

module load PrgEnv-gnu
module load cudatoolkit

make clean && make gpu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-5000}

echo " -=-=- GPU SCALING: N=${N}, 1 rank/GPU -=-=-"

echo "-=-=- P=4 (2x2) on ONE GPU node (4 GPUs) -=-=-"
srun -N 1 -n 4  -c 8 --gpus-per-task=1 --gpu-bind=single:1 ./summa --N ${N} --gpu --verify

echo "-=-=- P=16 (4x4) across FOUR GPU nodes (16 GPUs) -=-=-"
srun -N 4 -n 16 -c 8 --gpus-per-task=1 --gpu-bind=single:1 ./summa --N ${N} --gpu --verify