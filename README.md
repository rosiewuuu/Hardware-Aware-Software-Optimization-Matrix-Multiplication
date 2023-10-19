# ENSC 254 Lab 5: Hardware-Aware Software Optimizations

## Introduction
The objective of this project is to deepen the comprehension of hardware-aware software optimizations learned in ENSC 254, to enable writing well-optimized software for commodity CPUs. This lab covers three primary optimizations: 
1. Data tiling (via loop tiling) to leverage CPU caches for better memory system performance.
2. Vectorization (using X86 SIMD intrinsics) to leverage CPU SIMD instructions for enhanced fine-grained data-level parallelism; this is a single-core optimization.
3. Parallelization (using OpenMP multi-threading) to leverage multiple CPU cores for thread-level parallelism.

The General Matrix Multiply (GeMM) algorithm, a crucial component for deep learning applications, will be optimized on a lab machine equipped with a 10-core X86 CPU, by applying these optimizations incrementally. Lab 5 includes a performance competition segment.

## Hardware Configurations
The lab machine has the following specifications:
- CPU: 10 Intel X86 CPU cores with 20 hyper-threads (Intel(R) Core(TM) i9-10900 CPU @ 2.80GHz)
- SIMD support: AVX2 (256-bit) supported, AVX-512 (512-bit) not supported
- Cache:
  - L1: 32 KiB private cache per core, associativity = 8
  - L2: 256 KiB private cache per core, associativity = 4
  - L3: 20 MiB shared cache across all 10 cores, associativity = 16
- Cache line size: 64 bytes for all three cache levels

## GeMM Algorithm
The core computation in general matrix multiplication (GeMM) is represented by the equation `C = alpha * AB + beta * C`, where A, B, and C are matrices of size 4096x4096, and alpha and beta are constants. The core code is located in the `mm.cpp` file.

### TODO
The tasks include implementing the following functions in `main.c`:
- `gemm_tile`: Apply data tiling optimization to enhance cache locality and memory access performance.
- `gemm_tile_simd`: Build upon data tiling optimization by applying vectorization (SIMD) optimizations using X86 intrinsics.
- `gemm_tile_simd_par`: Extend previous optimizations with parallelization using OpenMP pragmas to parallelize loop execution.

Additionally, a document detailing the speedup achieved through each optimization should be created and included in the `lab5` folder.

## Building the Project
Inside the `lab` folder, run the following command to build the GeMM project. This command will compile the code and generate an executable named mm.

## Checking Running Processes
Before running any tests, it's a good practice to check the currently running processes on the machine to ensure no other heavy processes are running which might affect the performance measurement. You can use the top command for this, and exit top by pressing q.

## Running Tests
Now you can run the tests using the mm executable. There are four versions of the GeMM function that you can test, each corresponding to a different optimization strategy. You can specify which version to run by providing an argument (0, 1, 2, or 3) to the mm executable.

## Baseline Version
`./mm 0`
This command runs the baseline version of the GeMM function (gemm_base) and prints its execution time along with a verification of the result.

## Data Tiling Optimization
`./mm 1`
This command runs the gemm_tile function, which applies the data

## Data Tiling and SIMD Optimization
`./mm 2`
This command runs the gemm_tile_simd function, which builds upon the data tiling optimization by additionally applying the vectorization (SIMD) optimization, and prints its execution time along with a verification of the result.

## Data Tiling, SIMD, and Parallelization Optimization
`./mm 3`
This command runs the gemm_tile_simd_par function, which extends the previous optimizations with parallelization using OpenMP pragmas, and prints its execution time along with a verification of the result.
