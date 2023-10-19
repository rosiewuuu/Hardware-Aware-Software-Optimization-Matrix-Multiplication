#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

//helper funcitons
float** allocateMatrix(int rows, int cols) {
    float** matrix = (float**) malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*) calloc(cols, sizeof(float)); // Initialize to 0
    }
    return matrix;
}

void freeMatrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}



/* Main computational kernel: with tiling optimization. */
static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int TILE_SIZE = 16;
  int i, j, k, i1, j1, k1;
// Initiate each element of C with beta*C
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
  }
  // Tiling starts here
  for (i = 0; i < NI; i += TILE_SIZE) {
    for (j = 0; j < NJ; j += TILE_SIZE) {
      for (k = 0; k < NK; k += TILE_SIZE) {

        // Compute mini matrix multiplication
        for (i1 = i; i1 < i + TILE_SIZE && i1 < NI; ++i1) {
          for (j1 = j; j1 < j + TILE_SIZE && j1 < NJ; ++j1) {
            for (k1 = k; k1 < k + TILE_SIZE && k1 < NK; ++k1) {
              C[i1*NJ+j1] += alpha * A[i1*NK+k1] * B[k1*NJ+j1];
            }
          }
        }
      }
    }
  }


}

/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

  int TILE_SIZE_i = 16;
  int TILE_SIZE_j= 2048;
  int TILE_SIZE_k = 16;
  int i, j, k, i1, j1, k1;
  

// Initiate each element of C with beta*C
  __m256 beta_vec = _mm256_broadcast_ss(&beta);
  __m256 alpha_vec = _mm256_broadcast_ss(&alpha);
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j+=8) {
      __m256 c_vec = _mm256_loadu_ps(&C[i*NJ+j]);
      __m256 betaC_vec = _mm256_mul_ps(beta_vec, c_vec);
      _mm256_storeu_ps(&C[i*NJ+j], betaC_vec);
      //C[i*NJ+j] *= beta;
    }
  }

  // Tiling starts here
  for (i = 0; i < NI; i += TILE_SIZE_i) {
    for (j = 0; j < NJ; j += TILE_SIZE_j) {
      for (k = 0; k < NK; k += TILE_SIZE_k) {

        // Compute mini matrix multiplication
        for (i1 = i; i1 < i + TILE_SIZE_i && i1 < NI; i1++) {
          for (k1 = k; k1 < k + TILE_SIZE_k && k1 < NK; k1++) {
            //__m256 data type holds 32-bit floating-point values

            __m256 a_vec = _mm256_broadcast_ss(&A[i1*NK+k1]);
            __m256 alphaA_vec = _mm256_mul_ps(alpha_vec, a_vec);

            //for (k1 = k; k1 < k + TILE_SIZE && k1 < NK; ++k1) {
            for (j1 = j; j1 < j + TILE_SIZE_j && j1 < NJ; j1+=8) {
              /*Load 256-bits (composed of 8 packed single-precision (32-bit) 
              floating-point elements) from memory into dst. mem_addr does not 
              need to be aligned on any particular boundary.*/
              __m256 c_vec = _mm256_loadu_ps(&C[i1*NJ+j1]);
              __m256 b_vec = _mm256_loadu_ps(&B[k1*NJ+j1]);
              __m256 ab_vec = _mm256_mul_ps(alphaA_vec, b_vec);
              __m256 ABPlusC_vec = _mm256_add_ps(ab_vec, c_vec);
              _mm256_storeu_ps(&C[i1*NJ+j1], ABPlusC_vec);
            }
          }
        }
      }
    }
  }
}
/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{

  // => Form C := alpha*A*B + beta*C,
  //A is NIxNK
  //B is NKxNJ
  //C is NIxNJ

  int TILE_SIZE_i =16;
  int TILE_SIZE_j= 1024;
  int TILE_SIZE_k = 4;
  int i, j, k, i1, j1, k1;

  // Set the number of threads dynamically based on the system's available cores
  omp_set_num_threads(20);
  __m256 alpha_vec = _mm256_broadcast_ss(&alpha);
  __m256 beta_vec = _mm256_broadcast_ss(&beta);

  #pragma omp parallel for private(j) schedule(dynamic, 1)
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ - 8; j+=16) {
      __m256 c_vec = _mm256_loadu_ps(&C[i*NJ+j]);
      __m256 betaC_vec = _mm256_mul_ps(beta_vec, c_vec);
      _mm256_storeu_ps(&C[i*NJ+j], betaC_vec);

      __m256 c_vec2 = _mm256_loadu_ps(&C[i*NJ+j+8]);
      __m256 betaC_vec2 = _mm256_mul_ps(beta_vec, c_vec2);
      _mm256_storeu_ps(&C[i*NJ+j+8], betaC_vec2);
    }
  }

  // Tiling starts here
  #pragma omp parallel for private(j, k, i1, j1, k1) schedule(dynamic, 1)
  for (i = 0; i < NI; i += TILE_SIZE_i) {
    for (j = 0; j < NJ; j += TILE_SIZE_j) {
      for (k = 0; k < NK; k += TILE_SIZE_k) {

        // Compute mini matrix multiplication
        for (i1 = i; i1 < i + TILE_SIZE_i && i1 < NI; i1++) {
          for (k1 = k; k1 < k + TILE_SIZE_k && k1 < NK; k1++) {

            __m256 a_vec = _mm256_broadcast_ss(&A[i1*NK+k1]);
            //__m256 alpha_vec = _mm256_broadcast_ss(&alpha);
            __m256 alphaA_vec = _mm256_mul_ps(alpha_vec, a_vec);

            for (j1 = j; j1 < j + TILE_SIZE_j && j1 < NJ; j1+=16) {  // Unroll the loop (two vecotrs 8 numbers each)
              __m256 c_vec1 = _mm256_loadu_ps(&C[i1*NJ+j1]);
              __m256 b_vec1 = _mm256_loadu_ps(&B[k1*NJ+j1]);
              __m256 ab_vec1 = _mm256_mul_ps(alphaA_vec, b_vec1);
              __m256 ABPlusC_vec1 = _mm256_add_ps(ab_vec1, c_vec1);

              __m256 c_vec2 = _mm256_loadu_ps(&C[i1*NJ+j1+8]);
              __m256 b_vec2 = _mm256_loadu_ps(&B[k1*NJ+j1+8]);
              __m256 ab_vec2 = _mm256_mul_ps(alphaA_vec, b_vec2);
              __m256 ABPlusC_vec2 = _mm256_add_ps(ab_vec2, c_vec2);

              _mm256_storeu_ps(&C[i1*NJ+j1], ABPlusC_vec1);
              _mm256_storeu_ps(&C[i1*NJ+j1+8], ABPlusC_vec2);
            }
          }
        }
      }
    }
  }
}



int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}