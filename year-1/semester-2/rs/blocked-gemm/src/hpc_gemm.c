#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_SIZE 1024
#define MIN_SIZE 16

/* Code to clear cache */
/* Pentium III has 512K L2 cache, which is 128K ints */
#define ASIZE (1 << 17)
/* Cache block size is 32 bytes */
#define STRIDE 8
static int stuff[ASIZE];
static int sink;

/**
 * @brief Function to clear the cache.
 *
 * This function iterates over the 'stuff' array with a stride of 8 (32 bytes) 
 * to clear the cache.
 */
static void clear_cache() {
  int x = sink;
  int i;
  for (i = 0; i < ASIZE; i += STRIDE)
    x += stuff[i];
  sink = x;
}


/**
 * @brief Function to read the Time Stamp Counter.
 *
 * This function uses inline assembly to read the Time Stamp Counter (TSC) 
 * of the CPU. The TSC is a 64-bit register present on all x86 processors 
 * that counts the number of cycles since reset.
 *
 * @return The current value of the TSC.
 */
double rdtsc() {
    volatile unsigned int hi, lo;
    asm("rdtsc; movl %%edx,%0; movl %%eax,%1"
      : "=r" (hi), "=r" (lo)
      : /* No input */
      : "%edx", "%eax");
    double dtmp = lo * 1.0;
    dtmp = dtmp + (double)hi * (1<<30)*4;

    return dtmp;
}


/**
 * @brief Function to initialize a matrix with a specific value.
 *
 * This function iterates over a 2D array (matrix) and initializes each element 
 * with a specific value.
 *
 * @param mat The matrix to be initialized.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param value The value to initialize each element of the matrix.
 */
void mat_init_v1(int **mat, int m, int n, int value) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = value;
        }
    }
}


/**
 * @brief Performs a blocked General Matrix Multiply (GEMM) operation.
 *
 * This function multiplies matrix `a` by matrix `b` and adds the result to matrix `c`.
 * The matrices `a`, `b`, and `c` are assumed to have dimensions `n` x `k`, `k` x `m`, and `n` x `m` respectively.
 * The operation is performed in blocks of size `nc` x `kc`, `kc` x `mc`, and `nc` x `mc` for matrices `a`, `b`, and `c` respectively.
 * This is a blocked implementation which can be more efficient than the naive implementation for large matrices. The block 
 * implementation uses a cache-aware and tiling approach to improve the performance.
 *
 * @param a The first input matrix of dimensions `n` x `k`.
 * @param b The second input matrix of dimensions `k` x `m`.
 * @param c The output matrix of dimensions `n` x `m`. The result of the operation is added to this matrix.
 * @param n The number of rows in the first input matrix and the output matrix.
 * @param m The number of columns in the second input matrix and the output matrix.
 * @param k The number of columns in the first input matrix and the number of rows in the second input matrix.
 * @param nc The number of rows in a block of the first input matrix and the output matrix.
 * @param mc The number of columns in a block of the second input matrix and the output matrix.
 * @param kc The number of columns in a block of the first input matrix and the number of rows in a block of the second input matrix.
 */
void gemm_block(int **a, int **b, int **c, int n, int m, int k, int nc, int mc, int kc) {
    int **Bc = (int**)malloc(sizeof(int*)*kc);
    Bc[0] = (int*)malloc(sizeof(int)*kc*nc);
    for (int i = 0; i < kc; i++) {
        Bc[i] = Bc[0] + i*nc;
    }

    int **Ac = (int**)malloc(sizeof(int*)*mc);
    Ac[0] = (int*)malloc(sizeof(int)*mc*kc);
    for (int i = 0; i < mc; i++) {
        Ac[i] = Ac[0] + i*kc;
    }

    int **Cc = (int**)malloc(sizeof(int*)*mc);
    Cc[0] = (int*)malloc(sizeof(int)*mc*nc);
    for (int i = 0; i < mc; i++) {
        Cc[i] = Cc[0] + i*nc;
    }

    for (int jc = 0; jc < n; jc += nc) {
        for (int pc = 0; pc < k; pc += kc) {
            // Store the block into a temporary matrix (cache + tiling approach).
            for (int i = 0; i < kc; i++) {
                memcpy(Bc[i], &b[i + pc][jc], nc * sizeof(int));
            }

            for (int ic = 0; ic < m; ic += mc) {
                // Store the block into a temporary matrix (cache + tiling approach).
                for (int i = 0; i < mc; i++) {
                    memcpy(Ac[i], &a[i + ic][pc], kc * sizeof(int));
                }

                // Initialize the temporary matrix with zeros.
                for (int i = 0; i < mc; i++) {
                    memset(Cc[i], 0, nc * sizeof(int));
                }

                for (int i = 0; i < mc; i++) {
                    for (int j = 0; j < nc; j++) {
                        for (int l = 0; l < kc; l++) {
                            Cc[i][j] += Ac[i][l] * Bc[l][j];
                        }
                    }
                }

                // Copy the temporary matrix back to the original matrix.
                for (int i = 0; i < mc; i++) {
                    memcpy(&c[i + ic][jc], Cc[i], nc * sizeof(int));
                }            
            }
        }
    }

    free(Ac[0]);
    free(Ac);

    free(Bc[0]);
    free(Bc);
}


/**
 * @brief Performs a naive General Matrix Multiply (GEMM) operation.
 *
 * This function multiplies matrix `a` by matrix `b` and adds the result to matrix `c`.
 * The matrices `a`, `b`, and `c` are assumed to have dimensions `n` x `k`, `k` x `m`, and `n` x `m` respectively.
 * This is a naive implementation with cubic time complexity.
 *
 * @param a The first input matrix of dimensions `n` x `k`.
 * @param b The second input matrix of dimensions `k` x `m`.
 * @param c The output matrix of dimensions `n` x `m`. The result of the operation is added to this matrix.
 * @param n The number of rows in the first input matrix and the output matrix.
 * @param m The number of columns in the second input matrix and the output matrix.
 * @param k The number of columns in the first input matrix and the number of rows in the second input matrix.
 */
void gemm_naive(int **a, int **b, int **c, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Parameters for matrix/block sizes, first row the 'mc' parameter, second row the 'nc' parameter.
    int param_combinations[] = {
        56,  56, 
        224, 56, 
        224, 224, 
        896, 224
    };

    for (int i = 0; i < 8; i += 2) {
        int mc = param_combinations[i];
        int nc = param_combinations[i + 1];
        printf("mc = %d, nc = %d\n", mc, nc);
        printf("mc,nc,kc,BW\n");

        int kc, n, m, k;
        double  GFLOPS, elems;
        for (kc = 10; kc <= 600; kc = kc + 10) {            
            n = 4 * nc;
            m = 4 * mc;
            k = 4 * kc;

            int **mat1 = (int**)malloc(sizeof(int*) * m);
            mat1[0] = (int*)malloc(sizeof(int) * m * k);
            for (int i = 0; i < m; i++) {
                mat1[i] = mat1[0] + i * k;
            }

            int **mat2 = (int**)malloc(sizeof(int*) * k);
            mat2[0] = (int*)malloc(sizeof(int) * k * n);
            for (int i = 0; i < k; i++) {
                mat2[i] = mat2[0] + i * n;
            }

            int **mat3 = (int**)malloc(sizeof(int*) * m);
            mat3[0] = (int*)malloc(sizeof(int) * n * m);
            for (int i = 0; i < m; i++) {
                mat3[i] = mat3[0] + i * n;
            }

            mat_init_v1(mat1, m, k, 1);
            mat_init_v1(mat2, k, n, 1);
            mat_init_v1(mat3, m, n, 0);

            double sec = 0;
            double CPU_FREQ = 2.0 * 1000 * 1000 * 1000; // 2GHz for AMD
            double volatile t1, t2;

            clear_cache();

            t1 = rdtsc();
            // gemm_block(mat1, mat2, mat3, n, m, k, mc, nc, kc);
            gemm_naive(mat1, mat2, mat3, n, m, k);
            t2 = rdtsc();

            sec = (t2 - t1) / CPU_FREQ;
            elems = ((double) n) * m * k * (double) sizeof(int);
            double BW = (elems / (sec)) / 1000000; // MB/s
            GFLOPS = (2.0 * n * m * k) / (sec * 1000000000); // GFLOPS

            printf("%d,%d,%d,%f\n", mc, nc, kc, BW);

            free(mat1[0]);
            free(mat2[0]);
            free(mat3[0]);

            free(mat1);
            free(mat2);
            free(mat3);
        }
    }

    return 0;
}
