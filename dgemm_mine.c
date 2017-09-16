#include <string.h>
#include <x86intrin.h>
/*
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
*/
const char* dgemm_desc = "Nikko Mitrano Schaff's dgem.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{
	
    int i, j, k;
    double cij;
    /*
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
    */
    double b1, b2, b3, b4, b5, b6, b7, b8;
    for (j = 0; j < N; ++j) {
 		for (k = 0; k < (K - 7); k += 8) {
 			b1 = B[j*lda + k];
 			b2 = B[j*lda + k + 1];
 			b3 = B[j*lda + k + 2];
 			b4 = B[j*lda + k + 3];
 			b5 = B[j*lda + k + 4];
			b6 = B[j*lda + k + 5];
 			b7 = B[j*lda + k + 6];
 			b8 = B[j*lda + k + 7];
 			for (i = 0; i < M; ++i) {
 				C[j*lda + i] += A[k*lda + i] * b1;
 				C[j*lda + i] += A[(k+1)*lda + i] * b2;
 				C[j*lda + i] += A[(k+2)*lda + i] * b3;
 				C[j*lda + i] += A[(k+3)*lda + i] * b4;
 				C[j*lda + i] += A[(k+4)*lda + i] * b5;
				C[j*lda + i] += A[(k+5)*lda + i] * b6;
 				C[j*lda + i] += A[(k+6)*lda + i] * b7;
 				C[j*lda + i] += A[(k+7)*lda + i] * b8;
			 }
		 }
 	if(K % 8) {
 		do {
 			 b1 = B[j*lda + k];
 			for (i = 0; i < M; ++i) {
 				C[j*lda + i] += A[k*lda + i] * b1;
 			}
 		}
 		while(++k < K);
 	}
 }
}

void basic_dgemm_jki(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{
	double b;
 	for (int j = 0; j < lda; ++j) {
 		for (int k = 0; k < lda; ++k) {
 		b = B[j*lda + k];
 			for (int i = 0; i < lda; ++i) {
 				C[j*lda + i] += A[k*lda + i] * b;
 			}
		}
 	}
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk, I, J, K;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                //do_block(M, A, B, C, i, j, k);
		I = (i+BLOCK_SIZE > M ? M-i : BLOCK_SIZE);
		J = (j+BLOCK_SIZE > M ? M-j : BLOCK_SIZE);
		K = (k+BLOCK_SIZE > M ? M-k : BLOCK_SIZE);
		basic_dgemm(M, I, J, K, A + i + k*M, 
				B + k + j*M, C + i + j*M);
            }
        }
    }
}

/**
 * Enhancements:
 * Loop reordering (jki)
 * Copy optimization (b block)
 *
 * */
void square_dgemm_jki(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    int i, j, k;
    double bjk;
    for (j = 0; j < M; ++j) {
        for (k = 0; k < M; ++k) {
            bjk = B[j*M+k];
	    for (i = 0; i < M; ++i) {
                C[j*M + i] += A[k*M + i] * bjk;
	    }
	    
	}
    }
}

void square_dgemm_basic(const int M, 
                  const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
}

