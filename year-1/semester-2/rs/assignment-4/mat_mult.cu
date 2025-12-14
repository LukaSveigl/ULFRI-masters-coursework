#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#define N 4096
#define BLOCKSIZE 16
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WARP_SIZE 32

void init_matrices(half *a, half *b, float *c, int matsize) {
    for (int i = 0; i < matsize; ++i) {
        for (int j = 0; j < matsize; ++j) {
            a[i * matsize + j] = __float2half(1.0);
            b[i * matsize + j] = __float2half(1.0);
            c[i * matsize + j] = 1.0;
        }
    }
}


__global__ void mm_block_tc(int mat_size, half *A, half *B, float *C) {
    // Tile using a 2D grid
    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    //printf("warpM = %d, warpN = %d\n", warpX, warpY);
    int block_size = mat_size ;
    int aRow, aCol, bRow, bCol, cRow, cCol, dRow, dCol;
   
    // number of tiles 
    int num_tiles = CEIL_DIV(mat_size, BLOCKSIZE);
    int threadCol = threadIdx.x % BLOCKSIZE;
    int threadRow = threadIdx.x / BLOCKSIZE;

    // load A, B, C    
    half* A_block;
    half* B_block;
    float* C_block;

    // Continue

    // Define the WMMA matrix fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    for (int tile = 0; tile < num_tiles; ++tile) {
      // Calculate the starting points of the blocks in the A, B, and C matrices
      A_block = A + warpY * BLOCKSIZE * mat_size + tile * BLOCKSIZE; 
      B_block = B + tile * BLOCKSIZE * mat_size + warpX * BLOCKSIZE; 
      C_block = C + warpY * BLOCKSIZE * mat_size + warpX * BLOCKSIZE;

      // Load the A and B blocks into the WMMA fragments
      nvcuda::wmma::load_matrix_sync(a_frag, A_block, mat_size);
      nvcuda::wmma::load_matrix_sync(b_frag, B_block, mat_size);

      // Perform the matrix multiplication
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

      // Store the result back to the C block
      nvcuda::wmma::store_matrix_sync(C_block, c_frag, mat_size, nvcuda::wmma::mem_row_major);
    }
    
}


__global__ void mm_naive(int mat_size,  half *A, half *B, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < mat_size && y < mat_size) {
    float tmp = 0.0;
    for (int i = 0; i < mat_size; ++i) {
      tmp += (float)A[x * mat_size + i] * (float)B[i * mat_size + y];
    }

    C[x * mat_size + y] = tmp ;
  }
}


int main() {
  half *mat_a, *mat_b;
  float *mat_c;

  mat_a = (half*)malloc(N * N * sizeof(half));
  mat_b = (half*)malloc(N * N * sizeof(half));
  mat_c = (float*)malloc(N * N * sizeof(float));

  printf("Matrix size: %d x %d\n", N, N);
  fflush(stdout);

  // init matrices 
  init_matrices(mat_a, mat_b, mat_c, N);

  // instantiate buffers on the device
  half *d_mat_a, *d_mat_b;
  float *d_mat_c;
  cudaMalloc(&d_mat_a, N * N * sizeof(half));
  cudaMalloc(&d_mat_b, N * N * sizeof(half));
  cudaMalloc(&d_mat_c, N * N * sizeof(float));

  printf("Matrices initialized\n");
  fflush(stdout);

  // copy data from host to device
  cudaMemcpy(d_mat_a, mat_a, N * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat_b, mat_b, N * N * sizeof(half), cudaMemcpyHostToDevice);
  
  printf("Data copied to device\n");
  fflush(stdout);

    // First: using WMM
  dim3 gridDim;
  dim3 gridDimNaive;
  dim3 blockDim; 
  // launch kernel
  blockDim.x = 32;
  blockDim.y = 1;
  gridDim.x = CEIL_DIV(N, BLOCKSIZE * blockDim.x / 32); 
  gridDim.y = CEIL_DIV(N, BLOCKSIZE * blockDim.y);
  gridDimNaive.x = CEIL_DIV(N, blockDim.x);
  gridDimNaive.y = CEIL_DIV(N, blockDim.y);

  cudaEvent_t start1, stop1, start2, stop2;
  float milliseconds = 0;

  // to avoid CUDA compile overhead
  mm_naive<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);

  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);

  printf("Running gemm with Tensor cores...\n");
  cudaEventRecord(start2);
  // Uncommment here to use TC
  mm_block_tc<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);
  cudaEventRecord(stop2);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&milliseconds, start2, stop2);
  printf("Time taken with TC: %f ms\n", milliseconds);

  // Print last cuda error
  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
  }


  printf("Running gemm without Tensor cores...\n");
  cudaEventRecord(start1);
  mm_naive<<<gridDimNaive, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  printf("Time taken without TC: %f ms\n", milliseconds);
  
  cudaMemcpy(mat_c, d_mat_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);  

  // free memory
  free(mat_a);
  free(mat_b);
  free(mat_c);
  cudaFree(d_mat_a);
  cudaFree(d_mat_b);
  cudaFree(d_mat_c);
  
  return 0;
}