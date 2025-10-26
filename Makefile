# Use Cray MPI C++ wrapper on NERSC
MPICXX := CC
NVCC   ?= nvcc

# Optional MPI include for NVCC host
MPI_INC :=
ifdef CRAY_MPICH_DIR
  MPI_INC := -I$(CRAY_MPICH_DIR)/include
endif

BASE_CXXFLAGS := -O3 -std=c++17 -Iinclude
OMP_FLAG      := -fopenmp

# CPU build
CXXFLAGS_CPU  := $(BASE_CXXFLAGS) $(OMP_FLAG) $(MPI_INC)

# GPU build
CXXFLAGS_GPU  := $(BASE_CXXFLAGS) -DENABLE_CUDA $(OMP_FLAG) $(MPI_INC)

# NVCC flags for .cu files
NVCCFLAGS     := -O3 -std=c++17 -Iinclude -DENABLE_CUDA -ccbin=$(MPICXX) -Xcompiler="$(OMP_FLAG)" $(MPI_INC)

# Sources
CPU_SRCS  := src/main.cpp src/summa_cpu.cpp src/local_dgemm_cpu.cpp src/verify.cpp
GPU_SRCS  := src/summa_gpu.cu src/local_dgemm_gpu.cu

# build dirs
CPU_OBJS_CPU := $(patsubst src/%.cpp,build/cpu/%.o,$(CPU_SRCS))
CPU_OBJS_GPU := $(patsubst src/%.cpp,build/gpu/%.o,$(CPU_SRCS))
GPU_OBJS     := $(patsubst src/%.cu, build/gpu/%.o,$(GPU_SRCS))

## Target
TARGET := summa

.PHONY: all cpu gpu clean

# Default: CPU-only
all: cpu

# CPU-only binary
cpu: $(CPU_OBJS_CPU)
	$(MPICXX) $(CXXFLAGS_CPU) $^ -o $(TARGET)

# GPU-enabled binary
gpu: $(CPU_OBJS_GPU) $(GPU_OBJS)
	$(MPICXX) $(CXXFLAGS_GPU) $^ -o $(TARGET) -L$(CUDA_HOME)/lib64 -lcudart

build/cpu/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS_CPU) -c $< -o $@

build/gpu/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS_GPU) -c $< -o $@

build/gpu/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf build $(TARGET)