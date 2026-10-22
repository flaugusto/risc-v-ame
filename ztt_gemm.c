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

/******************************************************************************
 * Strided GEMM: C[M x P] += A[M x K] * B[K x P]
 *
 * Operates directly on standard row-major (or strided) matrices in place,
 * without any host packing or temporary tile buffers!
 * lda, ldb, ldc are the row strides (leading dimensions) in elements.
 * M, K and P must be multiples of N = AME_N.
 ******************************************************************************/
static ame_err_t ztt_gemm_strided_core(ame_state_t *ame, uint32_t dtype, size_t elem_bytes,
                                       int M, int K, int P,
                                       void *C, const void *A, const void *B,
                                       size_t ldc, size_t lda, size_t ldb) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    if (M % AME_N != 0 || K % AME_N != 0 || P % AME_N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;

    err = ame_msettyp(ame, m0, dtype);
    if (err != AME_OK) return err;

    err = ame_msettyp(ame, m1, dtype);
    if (err != AME_OK) return err;

    err = ame_asettyp(ame, acc0, dtype);
    if (err != AME_OK) return err;

    const size_t byte_stride_a = lda * elem_bytes;
    const size_t byte_stride_b = ldb * elem_bytes;
    const size_t byte_stride_c = ldc * elem_bytes;

    const int N = AME_N;

    for (int i = 0; i < M; i += N) {
        for (int j = 0; j < P; j += N) {
            uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)i * ldc + j) * elem_bytes;

            /* acc0 = C[i, j] */
            err = ame_mls_st(ame, m0, C_tile_ptr, byte_stride_c);
            if (err != AME_OK) return err;
            err = ame_mmov_a_m(ame, acc0, m0);
            if (err != AME_OK) return err;

            for (int k = 0; k < K; k += N) {
                const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)i * lda + k) * elem_bytes;
                const uint8_t *B_tile_ptr = (const uint8_t *)B + ((size_t)k * ldb + j) * elem_bytes;

                err = ame_mls_st(ame, m0, A_tile_ptr, byte_stride_a);
                if (err != AME_OK) return err;
                err = ame_mls_st(ame, m1, B_tile_ptr, byte_stride_b);
                if (err != AME_OK) return err;

                err = ame_mmulacc_2d(ame, acc0, m0, m1);
                if (err != AME_OK) return err;
            }

            err = ame_mmov_m_a(ame, m0, acc0);
            if (err != AME_OK) return err;
            err = ame_mss_st(ame, m0, C_tile_ptr, byte_stride_c);
            if (err != AME_OK) return err;
        }
    }

    return AME_OK;
}

ame_err_t ztt_sgemm_strided(ame_state_t *ame,
                            int M, int K, int P,
                            float *C, const float *A, const float *B,
                            size_t ldc, size_t lda, size_t ldb) {
    return ztt_gemm_strided_core(ame, AME_DTYPE_FP32, sizeof(float), M, K, P, C, A, B, ldc, lda, ldb);
}

ame_err_t ztt_igemm_strided(ame_state_t *ame,
                            int M, int K, int P,
                            int32_t *C, const int32_t *A, const int32_t *B,
                            size_t ldc, size_t lda, size_t ldb) {
    return ztt_gemm_strided_core(ame, AME_DTYPE_INT32, sizeof(int32_t), M, K, P, C, A, B, ldc, lda, ldb);
}

/******************************************************************************
 * Multi-Register Strided SGEMM Kernels
 ******************************************************************************/

/* 2x4 Micro-kernel: 8 accumulators (acc0..acc7), 6 matrix registers (m0..m5)
 * m0..m1: 2 rows of A
 * m2..m5: 4 columns of B
 * acc(r*4 + c): accumulator for C[i + r, j + c] */
ame_err_t ztt_sgemm_strided_2x4(ame_state_t *ame,
                                int M, int K, int P,
                                float *C, const float *A, const float *B,
                                size_t ldc, size_t lda, size_t ldb) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    const int N = AME_N;
    if (M % (2 * N) != 0 || P % (4 * N) != 0 || K % N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;
    const uint32_t dtype = AME_DTYPE_FP32;
    const size_t elem_bytes = sizeof(float);

    for (int r = 0; r < 6; ++r) {
        err = ame_msettyp(ame, r, dtype);
        if (err != AME_OK) return err;
    }
    for (int a = 0; a < 8; ++a) {
        err = ame_asettyp(ame, a, dtype);
        if (err != AME_OK) return err;
    }

    const size_t byte_stride_a = lda * elem_bytes;
    const size_t byte_stride_b = ldb * elem_bytes;
    const size_t byte_stride_c = ldc * elem_bytes;

    for (int i = 0; i < M; i += 2 * N) {
        for (int j = 0; j < P; j += 4 * N) {
            /* Load 2x4 block of C into acc0..acc7 */
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mls_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                    err = ame_mmov_a_m(ame, acc_idx, m0);
                    if (err != AME_OK) return err;
                }
            }

            /* Inner loop along K */
            for (int k = 0; k < K; k += N) {
                /* Load 2 A tiles into m0..m1 */
                for (int r = 0; r < 2; ++r) {
                    const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + k) * elem_bytes;
                    err = ame_mls_st(ame, r, A_tile_ptr, byte_stride_a);
                    if (err != AME_OK) return err;
                }
                /* Load 4 B tiles into m2..m5 */
                for (int c = 0; c < 4; ++c) {
                    const uint8_t *B_tile_ptr = (const uint8_t *)B + ((size_t)k * ldb + (j + c * N)) * elem_bytes;
                    err = ame_mls_st(ame, 2 + c, B_tile_ptr, byte_stride_b);
                    if (err != AME_OK) return err;
                }

                /* Compute 8 MACs */
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        int acc_idx = r * 4 + c;
                        err = ame_mmulacc_2d(ame, acc_idx, r, 2 + c);
                        if (err != AME_OK) return err;
                    }
                }
            }

            /* Store 2x4 block of C from acc0..acc7 */
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mmov_m_a(ame, m0, acc_idx);
                    if (err != AME_OK) return err;
                    err = ame_mss_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                }
            }
        }
    }
    return AME_OK;
}

/* 2x4 with Double-Buffering (Prefetch):
 * Buffer 0: A in m0..m1,   B in m2..m5
 * Buffer 1: A in m6..m7,   B in m8..m11
 * Alternates ping-pong buffers while overlapping loads of (k+1) with compute of k */
ame_err_t ztt_sgemm_strided_2x4_db(ame_state_t *ame,
                                   int M, int K, int P,
                                   float *C, const float *A, const float *B,
                                   size_t ldc, size_t lda, size_t ldb) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    const int N = AME_N;
    if (M % (2 * N) != 0 || P % (4 * N) != 0 || K % N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;
    const uint32_t dtype = AME_DTYPE_FP32;
    const size_t elem_bytes = sizeof(float);

    for (int r = 0; r < 12; ++r) {
        err = ame_msettyp(ame, r, dtype);
        if (err != AME_OK) return err;
    }
    for (int a = 0; a < 8; ++a) {
        err = ame_asettyp(ame, a, dtype);
        if (err != AME_OK) return err;
    }

    const size_t byte_stride_a = lda * elem_bytes;
    const size_t byte_stride_b = ldb * elem_bytes;
    const size_t byte_stride_c = ldc * elem_bytes;

    for (int i = 0; i < M; i += 2 * N) {
        for (int j = 0; j < P; j += 4 * N) {
            /* Load 2x4 block of C */
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mls_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                    err = ame_mmov_a_m(ame, acc_idx, m0);
                    if (err != AME_OK) return err;
                }
            }

            /* Prologue: prefetch iteration k=0 into buffer 0 (m0..m5) */
            for (int r = 0; r < 2; ++r) {
                const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + 0) * elem_bytes;
                err = ame_mls_st(ame, r, A_tile_ptr, byte_stride_a);
                if (err != AME_OK) return err;
            }
            for (int c = 0; c < 4; ++c) {
                const uint8_t *B_tile_ptr = (const uint8_t *)B + (0 + (j + c * N)) * elem_bytes;
                err = ame_mls_st(ame, 2 + c, B_tile_ptr, byte_stride_b);
                if (err != AME_OK) return err;
            }

            int curr_buf = 0;
            for (int k = 0; k < K; k += N) {
                int next_buf = 1 - curr_buf;
                int next_k = k + N;

                /* Prefetch next iteration (k + N) into next_buf */
                if (next_k < K) {
                    int next_a_base = next_buf * 6;
                    int next_b_base = next_buf * 6 + 2;
                    for (int r = 0; r < 2; ++r) {
                        const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + next_k) * elem_bytes;
                        err = ame_mls_st(ame, next_a_base + r, A_tile_ptr, byte_stride_a);
                        if (err != AME_OK) return err;
                    }
                    for (int c = 0; c < 4; ++c) {
                        const uint8_t *B_tile_ptr = (const uint8_t *)B + ((size_t)next_k * ldb + (j + c * N)) * elem_bytes;
                        err = ame_mls_st(ame, next_b_base + c, B_tile_ptr, byte_stride_b);
                        if (err != AME_OK) return err;
                    }
                }

                /* Compute current iteration on curr_buf */
                int curr_a_base = curr_buf * 6;
                int curr_b_base = curr_buf * 6 + 2;
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        int acc_idx = r * 4 + c;
                        err = ame_mmulacc_2d(ame, acc_idx, curr_a_base + r, curr_b_base + c);
                        if (err != AME_OK) return err;
                    }
                }

                curr_buf = next_buf;
            }

            /* Store 2x4 block of C */
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mmov_m_a(ame, m0, acc_idx);
                    if (err != AME_OK) return err;
                    err = ame_mss_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                }
            }
        }
    }
    return AME_OK;
}

/* 4x4 Micro-kernel: 16 accumulators (acc0..acc15), 8 matrix registers (m0..m7)
 * m0..m3: 4 rows of A
 * m4..m7: 4 columns of B
 * acc(r*4 + c): accumulator for C[i + r, j + c] */
ame_err_t ztt_sgemm_strided_4x4(ame_state_t *ame,
                                int M, int K, int P,
                                float *C, const float *A, const float *B,
                                size_t ldc, size_t lda, size_t ldb) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    const int N = AME_N;
    if (M % (4 * N) != 0 || P % (4 * N) != 0 || K % N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;
    const uint32_t dtype = AME_DTYPE_FP32;
    const size_t elem_bytes = sizeof(float);

    for (int r = 0; r < 8; ++r) {
        err = ame_msettyp(ame, r, dtype);
        if (err != AME_OK) return err;
    }
    for (int a = 0; a < 16; ++a) {
        err = ame_asettyp(ame, a, dtype);
        if (err != AME_OK) return err;
    }

    const size_t byte_stride_a = lda * elem_bytes;
    const size_t byte_stride_b = ldb * elem_bytes;
    const size_t byte_stride_c = ldc * elem_bytes;

    for (int i = 0; i < M; i += 4 * N) {
        for (int j = 0; j < P; j += 4 * N) {
            /* Load 4x4 block of C into acc0..acc15 */
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mls_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                    err = ame_mmov_a_m(ame, acc_idx, m0);
                    if (err != AME_OK) return err;
                }
            }

            /* Inner loop along K */
            for (int k = 0; k < K; k += N) {
                /* Load 4 A tiles into m0..m3 */
                for (int r = 0; r < 4; ++r) {
                    const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + k) * elem_bytes;
                    err = ame_mls_st(ame, r, A_tile_ptr, byte_stride_a);
                    if (err != AME_OK) return err;
                }
                /* Load 4 B tiles into m4..m7 */
                for (int c = 0; c < 4; ++c) {
                    const uint8_t *B_tile_ptr = (const uint8_t *)B + ((size_t)k * ldb + (j + c * N)) * elem_bytes;
                    err = ame_mls_st(ame, 4 + c, B_tile_ptr, byte_stride_b);
                    if (err != AME_OK) return err;
                }

                /* Compute 16 MACs */
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        int acc_idx = r * 4 + c;
                        err = ame_mmulacc_2d(ame, acc_idx, r, 4 + c);
                        if (err != AME_OK) return err;
                    }
                }
            }

            /* Store 4x4 block of C from acc0..acc15 */
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mmov_m_a(ame, m0, acc_idx);
                    if (err != AME_OK) return err;
                    err = ame_mss_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                }
            }
        }
    }
    return AME_OK;
}

/* 4x4 with Double-Buffering (Prefetch):
 * Buffer 0: A in m0..m3,   B in m4..m7
 * Buffer 1: A in m8..m11,  B in m12..m15
 * Total 16 M registers (within 32 registers limit) */
ame_err_t ztt_sgemm_strided_4x4_db(ame_state_t *ame,
                                   int M, int K, int P,
                                   float *C, const float *A, const float *B,
                                   size_t ldc, size_t lda, size_t ldb) {
    if (!ame || !C || !A || !B) return AME_ERR_ILLEGAL_INSN;
    const int N = AME_N;
    if (M % (4 * N) != 0 || P % (4 * N) != 0 || K % N != 0) return AME_ERR_ILLEGAL_INSN;

    ame_err_t err;
    const uint32_t dtype = AME_DTYPE_FP32;
    const size_t elem_bytes = sizeof(float);

    for (int r = 0; r < 16; ++r) {
        err = ame_msettyp(ame, r, dtype);
        if (err != AME_OK) return err;
    }
    for (int a = 0; a < 16; ++a) {
        err = ame_asettyp(ame, a, dtype);
        if (err != AME_OK) return err;
    }

    const size_t byte_stride_a = lda * elem_bytes;
    const size_t byte_stride_b = ldb * elem_bytes;
    const size_t byte_stride_c = ldc * elem_bytes;

    for (int i = 0; i < M; i += 4 * N) {
        for (int j = 0; j < P; j += 4 * N) {
            /* Load 4x4 block of C */
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mls_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                    err = ame_mmov_a_m(ame, acc_idx, m0);
                    if (err != AME_OK) return err;
                }
            }

            /* Prologue: prefetch k=0 into buffer 0 (m0..m7) */
            for (int r = 0; r < 4; ++r) {
                const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + 0) * elem_bytes;
                err = ame_mls_st(ame, r, A_tile_ptr, byte_stride_a);
                if (err != AME_OK) return err;
            }
            for (int c = 0; c < 4; ++c) {
                const uint8_t *B_tile_ptr = (const uint8_t *)B + (0 + (j + c * N)) * elem_bytes;
                err = ame_mls_st(ame, 4 + c, B_tile_ptr, byte_stride_b);
                if (err != AME_OK) return err;
            }

            int curr_buf = 0;
            for (int k = 0; k < K; k += N) {
                int next_buf = 1 - curr_buf;
                int next_k = k + N;

                /* Prefetch next iteration (k + N) into next_buf */
                if (next_k < K) {
                    int next_a_base = next_buf * 8;
                    int next_b_base = next_buf * 8 + 4;
                    for (int r = 0; r < 4; ++r) {
                        const uint8_t *A_tile_ptr = (const uint8_t *)A + ((size_t)(i + r * N) * lda + next_k) * elem_bytes;
                        err = ame_mls_st(ame, next_a_base + r, A_tile_ptr, byte_stride_a);
                        if (err != AME_OK) return err;
                    }
                    for (int c = 0; c < 4; ++c) {
                        const uint8_t *B_tile_ptr = (const uint8_t *)B + ((size_t)next_k * ldb + (j + c * N)) * elem_bytes;
                        err = ame_mls_st(ame, next_b_base + c, B_tile_ptr, byte_stride_b);
                        if (err != AME_OK) return err;
                    }
                }

                /* Compute current iteration on curr_buf */
                int curr_a_base = curr_buf * 8;
                int curr_b_base = curr_buf * 8 + 4;
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        int acc_idx = r * 4 + c;
                        err = ame_mmulacc_2d(ame, acc_idx, curr_a_base + r, curr_b_base + c);
                        if (err != AME_OK) return err;
                    }
                }

                curr_buf = next_buf;
            }

            /* Store 4x4 block of C */
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    uint8_t *C_tile_ptr = (uint8_t *)C + ((size_t)(i + r * N) * ldc + (j + c * N)) * elem_bytes;
                    int acc_idx = r * 4 + c;
                    err = ame_mmov_m_a(ame, m0, acc_idx);
                    if (err != AME_OK) return err;
                    err = ame_mss_st(ame, m0, C_tile_ptr, byte_stride_c);
                    if (err != AME_OK) return err;
                }
            }
        }
    }
    return AME_OK;
}
