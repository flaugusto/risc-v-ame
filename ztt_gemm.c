#include "ztt_gemm.h"

/******************************************************************************
 * Tiled GEMM: C[M x P] += A[M x K] * B[K x P]
 *
 * A, B and C are in block-tiled row-major layout (Ztt v0.6 Section 7.2.2):
 * every N x N tile is contiguous and row-major, and the tile grid is
 * row-major. M, K and P must be multiples of N = AME_N.
 ******************************************************************************/
static ame_err_t ztt_gemm_tiled(ame_state_t *ame, uint32_t dtype, size_t elem_bytes,
                                int M, int K, int P,
                                void *C, const void *A, const void *B) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    if (M % AME_N != 0 || K % AME_N != 0 || P % AME_N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;

    err = ame_msettyp(ame, m0, dtype); /* m0: Tile A, then Tile C */
    if (err != AME_OK) return err;

    err = ame_msettyp(ame, m1, dtype); /* m1: Tile B */
    if (err != AME_OK) return err;

    err = ame_asettyp(ame, acc0, dtype); /* acc0: Accumulator for Tile C */
    if (err != AME_OK) return err;

    const size_t square_bytes = (size_t)AME_NELEM * elem_bytes;
    const size_t A_row_stride = (size_t)(K / AME_N) * square_bytes;
    const size_t CB_row_stride = (size_t)(P / AME_N) * square_bytes;

    uint8_t *C_row_base = (uint8_t *)C;
    const uint8_t *A_row_base = (const uint8_t *)A;

    for (int i = 0; i < M; i += AME_N) {
        uint8_t *C_tile = C_row_base;
        const uint8_t *B_col_base = (const uint8_t *)B;

        for (int j = 0; j < P; j += AME_N) {
            /* acc0 = C[i, j] */
            err = ame_mls_rm(ame, m0, C_tile);
            if (err != AME_OK) return err;
            err = ame_mmov_a_m(ame, acc0, m0);
            if (err != AME_OK) return err;

            const uint8_t *A_tile = A_row_base;
            const uint8_t *B_tile = B_col_base;

            for (int k = 0; k < K; k += AME_N) {
                err = ame_mls_rm(ame, m0, A_tile);
                if (err != AME_OK) return err;
                err = ame_mls_rm(ame, m1, B_tile);
                if (err != AME_OK) return err;

                /* acc0 += m0 * m1 */
                err = ame_mmulacc_2d(ame, acc0, m0, m1);
                if (err != AME_OK) return err;

                A_tile += square_bytes;
                B_tile += CB_row_stride;
            }

            err = ame_mmov_m_a(ame, m0, acc0);
            if (err != AME_OK) return err;
            err = ame_mss_rm(ame, m0, C_tile);
            if (err != AME_OK) return err;

            C_tile += square_bytes;
            B_col_base += square_bytes;
        }

        C_row_base += CB_row_stride;
        A_row_base += A_row_stride;
    }

    return AME_OK;
}

ame_err_t ztt_sgemm(ame_state_t *ame,
                    int M, int K, int P,
                    float *C, const float *A, const float *B) {
    return ztt_gemm_tiled(ame, AME_DTYPE_FP32, sizeof(float), M, K, P, C, A, B);
}

ame_err_t ztt_igemm(ame_state_t *ame,
                    int M, int K, int P,
                    int32_t *C, const int32_t *A, const int32_t *B) {
    return ztt_gemm_tiled(ame, AME_DTYPE_INT32, sizeof(int32_t), M, K, P, C, A, B);
}
