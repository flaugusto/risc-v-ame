#ifndef ZTT_GEMM_H
#define ZTT_GEMM_H

#include "ztt_ame.h"


/* FP32 Strided GEMM (operates directly on row-major matrices, no software packing) */
ame_err_t ztt_sgemm_strided(ame_state_t *ame,
                            int M, int K, int P,
                            float *C, const float *A, const float *B,
                            size_t ldc, size_t lda, size_t ldb);

/* Int32 Strided GEMM (operates directly on row-major matrices, no software packing) */
ame_err_t ztt_igemm_strided(ame_state_t *ame,
                            int M, int K, int P,
                            int32_t *C, const int32_t *A, const int32_t *B,
                            size_t ldc, size_t lda, size_t ldb);

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * High-Level Tiled GEMM Kernels using AME Instructions
 * Computes: C[M x P] += A[M x K] * B[K x P]
 *
 * A, B and C use block-tiled row-major layout: every N x N tile (N = AME_N)
 * is contiguous and row-major, and the tile grid is row-major. M, K and P
 * must be multiples of N; otherwise AME_ERR_ILLEGAL_INSN is returned.
 ******************************************************************************/

/* FP32 Single-Precision GEMM (SGEMM) */
ame_err_t ztt_sgemm(ame_state_t *ame,
                    int M, int K, int P,
                    float *C, const float *A, const float *B);

/* Int32 Integer GEMM (IGEMM) */
ame_err_t ztt_igemm(ame_state_t *ame,
                    int M, int K, int P,
                    int32_t *C, const int32_t *A, const int32_t *B);

#ifdef __cplusplus
}
#endif

#endif /* ZTT_GEMM_H */
