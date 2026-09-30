#include "ztt_gemm.h"
#include <string.h>

/******************************************************************************
 * FP32 Single-Precision GEMM
 * Computes: C[M x P] += A[M x K] * B[K x P]
 ******************************************************************************/
ame_err_t ztt_sgemm(ame_state_t *ame,
                    int M, int K, int P,
                    float *C, const float *A, const float *B) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;

    /* Step 1: Configure register datatypes */
    err = ame_msettyp(ame, m0, AME_DTYPE_FP32); /* m0: Tile A */
    if (err != AME_OK) return err;

    err = ame_msettyp(ame, m1, AME_DTYPE_FP32); /* m1: Tile B */
    if (err != AME_OK) return err;

    err = ame_asettyp(ame, acc0, AME_DTYPE_FP32); /* acc0: Accumulator for Tile C */
    if (err != AME_OK) return err;

    const int N = AME_N;
    float tile_A[AME_NELEM];
    float tile_B[AME_NELEM];
    float tile_C[AME_NELEM];

    /* Triple nested tiling loop over M, P, K */
    for (int i = 0; i < M; i += N) {
        for (int j = 0; j < P; j += N) {
            /* Clear accumulator acc0 for current C tile */
            err = ame_mzero_2d_acc(ame, acc0);
            if (err != AME_OK) return err;

            for (int k = 0; k < K; k += N) {
                /* Pack sub-matrix tile from A into contiguous row-major buffer */
                memset(tile_A, 0, sizeof(tile_A));
                int rows_a = (i + N <= M) ? N : (M - i);
                int cols_a = (k + N <= K) ? N : (K - k);
                for (int r = 0; r < rows_a; ++r) {
                    for (int c = 0; c < cols_a; ++c) {
                        tile_A[r * N + c] = A[(i + r) * K + (k + c)];
                    }
                }

                /* Pack sub-matrix tile from B into contiguous row-major buffer */
                memset(tile_B, 0, sizeof(tile_B));
                int rows_b = (k + N <= K) ? N : (K - k);
                int cols_b = (j + N <= P) ? N : (P - j);
                for (int r = 0; r < rows_b; ++r) {
                    for (int c = 0; c < cols_b; ++c) {
                        tile_B[r * N + c] = B[(k + r) * P + (j + c)];
                    }
                }

                /* Load tiles into M registers */
                err = ame_mls_rm(ame, m0, tile_A);
                if (err != AME_OK) return err;

                err = ame_mls_rm(ame, m1, tile_B);
                if (err != AME_OK) return err;

                /* Multiply and accumulate: acc0 += m0 * m1 */
                err = ame_mmulacc_2d(ame, acc0, m0, m1);
                if (err != AME_OK) return err;
            }

            /* Move accumulated tile to m0 */
            err = ame_mmov_m_a(ame, m0, acc0);
            if (err != AME_OK) return err;

            /* Store result tile to memory buffer */
            err = ame_mss_rm(ame, m0, tile_C);
            if (err != AME_OK) return err;

            /* Accumulate result back into C matrix (handling boundary edges) */
            int rows_c = (i + N <= M) ? N : (M - i);
            int cols_c = (j + N <= P) ? N : (P - j);
            for (int r = 0; r < rows_c; ++r) {
                for (int c = 0; c < cols_c; ++c) {
                    C[(i + r) * P + (j + c)] += tile_C[r * N + c];
                }
            }
        }
    }

    return AME_OK;
}

/******************************************************************************
 * Int32 Integer GEMM
 * Computes: C[M x P] += A[M x K] * B[K x P]
 ******************************************************************************/
ame_err_t ztt_igemm(ame_state_t *ame,
                    int M, int K, int P,
                    int32_t *C, const int32_t *A, const int32_t *B) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;

    /* Step 1: Configure register datatypes */
    err = ame_msettyp(ame, m0, AME_DTYPE_INT32); /* m0: Tile A */
    if (err != AME_OK) return err;

    err = ame_msettyp(ame, m1, AME_DTYPE_INT32); /* m1: Tile B */
    if (err != AME_OK) return err;

    err = ame_asettyp(ame, acc0, AME_DTYPE_INT32); /* acc0: Accumulator for Tile C */
    if (err != AME_OK) return err;

    const int N = AME_N;
    int32_t tile_A[AME_NELEM];
    int32_t tile_B[AME_NELEM];
    int32_t tile_C[AME_NELEM];

    /* Triple nested tiling loop over M, P, K */
    for (int i = 0; i < M; i += N) {
        for (int j = 0; j < P; j += N) {
            /* Clear accumulator acc0 for current C tile */
            err = ame_mzero_2d_acc(ame, acc0);
            if (err != AME_OK) return err;

            for (int k = 0; k < K; k += N) {
                memset(tile_A, 0, sizeof(tile_A));
                int rows_a = (i + N <= M) ? N : (M - i);
                int cols_a = (k + N <= K) ? N : (K - k);
                for (int r = 0; r < rows_a; ++r) {
                    for (int c = 0; c < cols_a; ++c) {
                        tile_A[r * N + c] = A[(i + r) * K + (k + c)];
                    }
                }

                memset(tile_B, 0, sizeof(tile_B));
                int rows_b = (k + N <= K) ? N : (K - k);
                int cols_b = (j + N <= P) ? N : (P - j);
                for (int r = 0; r < rows_b; ++r) {
                    for (int c = 0; c < cols_b; ++c) {
                        tile_B[r * N + c] = B[(k + r) * P + (j + c)];
                    }
                }

                /* Load tiles into M registers */
                err = ame_mls_rm(ame, m0, tile_A);
                if (err != AME_OK) return err;

                err = ame_mls_rm(ame, m1, tile_B);
                if (err != AME_OK) return err;

                /* Multiply and accumulate: acc0 += m0 * m1 */
                err = ame_mmulacc_2d(ame, acc0, m0, m1);
                if (err != AME_OK) return err;
            }

            /* Move accumulated tile to m0 */
            err = ame_mmov_m_a(ame, m0, acc0);
            if (err != AME_OK) return err;

            /* Store result tile to memory buffer */
            err = ame_mss_rm(ame, m0, tile_C);
            if (err != AME_OK) return err;

            /* Accumulate result back into C matrix */
            int rows_c = (i + N <= M) ? N : (M - i);
            int cols_c = (j + N <= P) ? N : (P - j);
            for (int r = 0; r < rows_c; ++r) {
                for (int c = 0; c < cols_c; ++c) {
                    C[(i + r) * P + (j + c)] += tile_C[r * N + c];
                }
            }
        }
    }

    return AME_OK;
}
