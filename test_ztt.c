#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "ztt_ame.h"
#include "ztt_gemm.h"
#include "packing.h"

#define COLOR_GREEN "\033[0;32m"
#define COLOR_RED "\033[0;31m"
#define COLOR_BLUE "\033[0;34m"
#define COLOR_RESET "\033[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg)                                                                      \
    do                                                                                              \
    {                                                                                               \
        if (!(cond))                                                                                \
        {                                                                                           \
            printf(COLOR_RED "  [FAIL] %s: line %d: %s" COLOR_RESET "\n", __func__, __LINE__, msg); \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

/******************************************************************************
 * Test 1: Ownership & Backend Access Control
 ******************************************************************************/
static bool test_ownership(void)
{
    printf(COLOR_BLUE "[RUN] test_ownership" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);

    /* Initially unowned: any instruction should fail with NOT_OWNED */
    ame_err_t err = ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_ERR_NOT_OWNED, "Instruction before acquire must fail with NOT_OWNED");

    /* Acquire backend */
    uint32_t rd = 0;
    err = ame_acquire(&ame, &rd, 0);
    TEST_ASSERT(err == AME_OK, "ame_acquire failed");
    TEST_ASSERT((rd & AME_ACQ_SUCCESS) != 0, "Acquire success bit must be 1");
    TEST_ASSERT((ame.ameown & AME_OWN_OWNED) != 0, "ameown.OWNED must be 1");

    /* Now instructions should succeed */
    err = ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "msettyp must succeed when owned");

    /* Release backend */
    err = ame_release(&ame);
    TEST_ASSERT(err == AME_OK, "ame_release failed");
    TEST_ASSERT((ame.ameown & AME_OWN_OWNED) == 0, "ameown.OWNED must be 0 after release");

    /* Instructions must fail again */
    err = ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_ERR_NOT_OWNED, "Instruction after release must fail with NOT_OWNED");

    return true;
}

/******************************************************************************
 * Test 2: Datatype Descriptors and Register Zeroing
 ******************************************************************************/
static bool test_datatypes_and_zeroing(void)
{
    printf(COLOR_BLUE "[RUN] test_datatypes_and_zeroing" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    /* Fill M0 with dirty byte pattern */
    memset(ame.M[m0], 0xAA, AME_M_REG_SIZE_BYTES);

    /* msettyp m0, AME_DTYPE_FP32 -> must clear raw bits to 0 */
    ame_err_t err = ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "msettyp failed");

    for (size_t b = 0; b < AME_M_REG_SIZE_BYTES; ++b)
    {
        TEST_ASSERT(ame.M[m0][b] == 0, "msettyp must raw-clear all physical bits of M register");
    }

    /* Verify mgettyp reads back the same value */
    uint32_t rd = 0;
    err = ame_mgettyp(&ame, &rd, m0);
    TEST_ASSERT(err == AME_OK, "mgettyp failed");
    TEST_ASSERT(rd == AME_DTYPE_FP32, "mgettyp value mismatch");

    /* Fill Acc0 with dirty pattern and asettyp */
    memset(ame.Acc[acc0], 0x55, AME_ACC_REG_SIZE_BYTES);
    err = ame_asettyp(&ame, acc0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "asettyp failed");

    for (size_t b = 0; b < AME_ACC_REG_SIZE_BYTES; ++b)
    {
        TEST_ASSERT(ame.Acc[acc0][b] == 0, "asettyp must raw-clear all physical bits of Acc register");
    }

    err = ame_agettyp(&ame, &rd, acc0);
    TEST_ASSERT(err == AME_OK, "agettyp failed");
    TEST_ASSERT(rd == AME_DTYPE_FP32, "agettyp value mismatch");

    return true;
}

/******************************************************************************
 * Test 3: Row-Major Memory Load and Store Round-Trip
 ******************************************************************************/
static bool test_memory_roundtrip(void)
{
    printf(COLOR_BLUE "[RUN] test_memory_roundtrip" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    ame_msettyp(&ame, m2, AME_DTYPE_FP32);

    float src_tile[AME_NELEM];
    float dst_tile[AME_NELEM];

    for (int i = 0; i < AME_NELEM; ++i)
    {
        src_tile[i] = (float)(i + 1) * 0.5f;
        dst_tile[i] = 0.0f;
    }

    /* mls.rm m2, src_tile */
    ame_err_t err = ame_mls_rm(&ame, m2, src_tile);
    TEST_ASSERT(err == AME_OK, "mls.rm failed");

    /* mss.rm m2, dst_tile */
    err = ame_mss_rm(&ame, m2, dst_tile);
    TEST_ASSERT(err == AME_OK, "mss.rm failed");

    /* Compare */
    for (int i = 0; i < AME_NELEM; ++i)
    {
        TEST_ASSERT(src_tile[i] == dst_tile[i], "Memory round-trip data mismatch");
    }

    return true;
}

/******************************************************************************
 * Test 3b: Strided Memory Load and Store Round-Trip (mls.st / mss.st)
 ******************************************************************************/
static bool test_strided_memory_roundtrip(void)
{
    printf(COLOR_BLUE "[RUN] test_strided_memory_roundtrip" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    ame_msettyp(&ame, m2, AME_DTYPE_FP32);

    const int N = AME_N;
    const int stride_elements = 2 * N + 5; /* non-contiguous stride */
    const size_t byte_stride = (size_t)stride_elements * sizeof(float);

    float *src_matrix = (float *)calloc((size_t)N * stride_elements, sizeof(float));
    float *dst_matrix = (float *)calloc((size_t)N * stride_elements, sizeof(float));

    /* Initialize distinct values in the N x N region embedded in the strided buffer */
    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            src_matrix[r * stride_elements + c] = (float)(r * 100 + c + 1);
        }
    }

    /* mls.st m2, (src_matrix), byte_stride */
    ame_err_t err = ame_mls_st(&ame, m2, src_matrix, byte_stride);
    TEST_ASSERT(err == AME_OK, "mls.st failed");

    /* mss.st m2, (dst_matrix), byte_stride */
    err = ame_mss_st(&ame, m2, dst_matrix, byte_stride);
    TEST_ASSERT(err == AME_OK, "mss.st failed");

    /* Verify that N x N elements were correctly stored back with stride */
    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            TEST_ASSERT(dst_matrix[r * stride_elements + c] == src_matrix[r * stride_elements + c],
                        "Strided round-trip data mismatch in N x N region");
        }
        /* Verify elements outside N columns remained zero */
        for (int c = N; c < stride_elements; ++c)
        {
            TEST_ASSERT(dst_matrix[r * stride_elements + c] == 0.0f,
                        "Elements outside tile row segment must not be overwritten");
        }
    }

    /* Check misaligned base address fails */
    char *misaligned_ptr = ((char *)src_matrix) + 1;
    err = ame_mls_st(&ame, m2, misaligned_ptr, byte_stride);
    TEST_ASSERT(err == AME_ERR_MISALIGNED, "mls.st with misaligned base must fail with AME_ERR_MISALIGNED");

    err = ame_mss_st(&ame, m2, misaligned_ptr, byte_stride);
    TEST_ASSERT(err == AME_ERR_MISALIGNED, "mss.st with misaligned base must fail with AME_ERR_MISALIGNED");

    free(src_matrix);
    free(dst_matrix);
    return true;
}

/******************************************************************************
 * Test 4: Datatype Mismatch & amestatus.UN Sticky Flag
 ******************************************************************************/
static bool test_type_mismatch(void)
{
    printf(COLOR_BLUE "[RUN] test_type_mismatch" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    /* Configure m0 as FP32 and acc0 as INT32 */
    ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    ame_asettyp(&ame, acc0, AME_DTYPE_INT32);

    /* Try mmov.m.a: types differ! */
    TEST_ASSERT((ame.amestatus & AME_STATUS_UN) == 0, "amestatus.UN must initially be 0");
    ame_err_t err = ame_mmov_m_a(&ame, m0, acc0);
    TEST_ASSERT(err == AME_ERR_UNSUPPORTED, "Mismatched mmov.m.a must return AME_ERR_UNSUPPORTED");
    TEST_ASSERT((ame.amestatus & AME_STATUS_UN) != 0, "amestatus.UN must be set after mismatch");

    /* Try mmulacc with mismatched operands */
    ame_msettyp(&ame, m1, AME_DTYPE_INT32);
    err = ame_mmulacc_2d(&ame, acc0, m0, m1); /* acc=INT32, m0=FP32, m1=INT32 */
    TEST_ASSERT(err == AME_ERR_UNSUPPORTED, "Mismatched mmulacc.2d must return AME_ERR_UNSUPPORTED");

    return true;
}

/******************************************************************************
 * Test 5: Single-Tile Matrix Multiply-Accumulate (FP32)
 ******************************************************************************/
static bool test_single_tile_sgemm(void)
{
    printf(COLOR_BLUE "[RUN] test_single_tile_sgemm" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    float A[AME_NELEM];
    float B[AME_NELEM];
    float C_hw[AME_NELEM];
    float C_ref[AME_NELEM];

    /* Initialize with deterministic values */
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i * N + j] = (float)(i + 1);
            B[i * N + j] = (float)(j + 1);
            C_hw[i * N + j] = 0.0f;
            C_ref[i * N + j] = 0.0f;
        }
    }

    /* CPU reference */
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    /* Hardware simulation */
    ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    ame_msettyp(&ame, m1, AME_DTYPE_FP32);
    ame_asettyp(&ame, acc0, AME_DTYPE_FP32);

    ame_mzero_2d_acc(&ame, acc0);
    ame_mls_rm(&ame, m0, A);
    ame_mls_rm(&ame, m1, B);
    ame_mmulacc_2d(&ame, acc0, m0, m1);
    ame_mmov_m_a(&ame, m0, acc0);
    ame_mss_rm(&ame, m0, C_hw);

    /* Verify equality */
    for (int i = 0; i < AME_NELEM; ++i)
    {
        float diff = fabsf(C_hw[i] - C_ref[i]);
        TEST_ASSERT(diff < 1e-5f, "Single tile SGEMM result mismatch with reference");
    }

    return true;
}

/******************************************************************************
 * Test 6: Single-Tile Matrix Multiply-Accumulate (Int32)
 ******************************************************************************/
static bool test_single_tile_igemm(void)
{
    printf(COLOR_BLUE "[RUN] test_single_tile_igemm" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    int32_t A[AME_NELEM];
    int32_t B[AME_NELEM];
    int32_t C_hw[AME_NELEM];
    int32_t C_ref[AME_NELEM];

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i * N + j] = (i - 2);
            B[i * N + j] = (j + 3);
            C_hw[i * N + j] = 0;
            C_ref[i * N + j] = 0;
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int32_t sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    ame_msettyp(&ame, m0, AME_DTYPE_INT32);
    ame_msettyp(&ame, m1, AME_DTYPE_INT32);
    ame_asettyp(&ame, acc0, AME_DTYPE_INT32);

    ame_mzero_2d_acc(&ame, acc0);
    ame_mls_rm(&ame, m0, A);
    ame_mls_rm(&ame, m1, B);
    ame_mmulacc_2d(&ame, acc0, m0, m1);
    ame_mmov_m_a(&ame, m0, acc0);
    ame_mss_rm(&ame, m0, C_hw);

    for (int i = 0; i < AME_NELEM; ++i)
    {
        TEST_ASSERT(C_hw[i] == C_ref[i], "Single tile IGEMM result mismatch with reference");
    }

    return true;
}

/******************************************************************************
 * Test 7: Multi-Tile End-to-End SGEMM, C += A * B (M x K x P not multiples of N)
 ******************************************************************************/
static bool test_full_sgemm(void)
{
    printf(COLOR_BLUE "[RUN] test_full_sgemm (Multi-tile M=24, K=20, P=28, C accumulated)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int M = 24;
    const int K = 20;
    const int P = 28;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * P * sizeof(float));
    float *C0 = (float *)malloc(M * P * sizeof(float));
    float *C_ref = (float *)malloc(M * P * sizeof(float));
    float *C_hw = (float *)malloc(M * P * sizeof(float));

    for (int i = 0; i < M * K; ++i)
        A[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;
    for (int i = 0; i < K * P; ++i)
        B[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;
    for (int i = 0; i < M * P; ++i)
        C0[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    /* CPU reference: C = C0 + A * B */
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            float sum = C0[i * P + j];
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * P + j];
            C_ref[i * P + j] = sum;
        }
    }

    float *Ap = (float *)pack_tiles(A, M, K, sizeof(float), NULL);
    float *Bp = (float *)pack_tiles(B, K, P, sizeof(float), NULL);
    float *Cp = (float *)pack_tiles(C0, M, P, sizeof(float), NULL);

    ame_err_t err = ztt_sgemm(&ame, round_up_to_tile(M), round_up_to_tile(K), round_up_to_tile(P), Cp, Ap, Bp);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm failed");

    unpack_tiles(Cp, M, P, sizeof(float), C_hw, NULL);

    float max_diff = 0.0f;
    for (int i = 0; i < M * P; ++i)
    {
        float diff = fabsf(C_hw[i] - C_ref[i]);
        if (diff > max_diff)
            max_diff = diff;
        TEST_ASSERT(diff < 1e-4f, "Multi-tile SGEMM result differs from CPU reference C0 + A*B");
    }
    printf("       Max diff: %e\n", max_diff);

    free(A);
    free(B);
    free(C0);
    free(C_ref);
    free(C_hw);
    free(Ap);
    free(Bp);
    free(Cp);
    return true;
}

/******************************************************************************
 * Test 8: Multi-Tile End-to-End IGEMM, C += A * B (M x K x P not multiples of N)
 ******************************************************************************/
static bool test_full_igemm(void)
{
    printf(COLOR_BLUE "[RUN] test_full_igemm (Multi-tile M=24, K=20, P=28, C accumulated)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int M = 24;
    const int K = 20;
    const int P = 28;

    int32_t *A = (int32_t *)malloc(M * K * sizeof(int32_t));
    int32_t *B = (int32_t *)malloc(K * P * sizeof(int32_t));
    int32_t *C0 = (int32_t *)malloc(M * P * sizeof(int32_t));
    int32_t *C_ref = (int32_t *)malloc(M * P * sizeof(int32_t));
    int32_t *C_hw = (int32_t *)malloc(M * P * sizeof(int32_t));

    for (int i = 0; i < M * K; ++i)
        A[i] = (rand() % 20) - 10;
    for (int i = 0; i < K * P; ++i)
        B[i] = (rand() % 20) - 10;
    for (int i = 0; i < M * P; ++i)
        C0[i] = (rand() % 200) - 100;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            int32_t sum = C0[i * P + j];
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * P + j];
            C_ref[i * P + j] = sum;
        }
    }

    int32_t *Ap = (int32_t *)pack_tiles(A, M, K, sizeof(int32_t), NULL);
    int32_t *Bp = (int32_t *)pack_tiles(B, K, P, sizeof(int32_t), NULL);
    int32_t *Cp = (int32_t *)pack_tiles(C0, M, P, sizeof(int32_t), NULL);

    ame_err_t err = ztt_igemm(&ame, round_up_to_tile(M), round_up_to_tile(K), round_up_to_tile(P), Cp, Ap, Bp);
    TEST_ASSERT(err == AME_OK, "ztt_igemm failed");

    unpack_tiles(Cp, M, P, sizeof(int32_t), C_hw, NULL);

    for (int i = 0; i < M * P; ++i)
        TEST_ASSERT(C_hw[i] == C_ref[i], "Multi-tile IGEMM result differs from CPU reference C0 + A*B");

    free(A);
    free(B);
    free(C0);
    free(C_ref);
    free(C_hw);
    free(Ap);
    free(Bp);
    free(Cp);
    return true;
}

/******************************************************************************
 * Test 8c: Multi-Tile Strided SGEMM & IGEMM (Zero Host Packing)
 ******************************************************************************/
static bool test_full_strided_gemm(void)
{
    printf(COLOR_BLUE "[RUN] test_full_strided_gemm (Multi-tile SGEMM & IGEMM with mls.st/mss.st)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    const int M = 2 * N;
    const int K = 3 * N;
    const int P = 2 * N;

    /* Leading dimensions with extra padding to test arbitrary row strides */
    const size_t lda = K + 7;
    const size_t ldb = P + 5;
    const size_t ldc = P + 3;

    /* Allocate and initialize FP32 matrices */
    float *A = (float *)calloc((size_t)M * lda, sizeof(float));
    float *B = (float *)calloc((size_t)K * ldb, sizeof(float));
    float *C0 = (float *)calloc((size_t)M * ldc, sizeof(float));
    float *C_hw = (float *)calloc((size_t)M * ldc, sizeof(float));
    float *C_ref = (float *)calloc((size_t)M * ldc, sizeof(float));

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[i * lda + k] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    for (int k = 0; k < K; ++k)
        for (int j = 0; j < P; ++j)
            B[k * ldb + j] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
        {
            float val = ((float)(rand() % 100) - 50.0f) / 10.0f;
            C0[i * ldc + j] = val;
            C_hw[i * ldc + j] = val;
        }

    /* CPU reference: C += A * B */
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            float sum = C0[i * ldc + j];
            for (int k = 0; k < K; ++k)
                sum += A[i * lda + k] * B[k * ldb + j];
            C_ref[i * ldc + j] = sum;
        }
    }

    ame_stats_reset(&ame);
    ame_err_t err = ztt_sgemm_strided(&ame, M, K, P, C_hw, A, B, ldc, lda, ldb);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm_strided failed");

    /* Validate numerical accuracy */
    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            float diff = fabsf(C_hw[i * ldc + j] - C_ref[i * ldc + j]);
            if (diff > max_diff)
                max_diff = diff;
            TEST_ASSERT(diff < 1e-4f, "ztt_sgemm_strided mismatch with CPU reference");
        }
    }

    /* Verify stats: all loads/stores were strided (.st), zero row-major (.rm) */
    TEST_ASSERT(ame.stats.mls_rm == 0 && ame.stats.mss_rm == 0, "Strided GEMM must not use mls.rm/mss.rm");
    const unsigned long long Mt = M / N, Kt = K / N, Pt = P / N;
    TEST_ASSERT(ame.stats.mls_st == Mt * Pt * (1 + 2 * Kt), "mls.st count mismatch");
    TEST_ASSERT(ame.stats.mss_st == Mt * Pt, "mss.st count mismatch");
    TEST_ASSERT(ame.stats.mmulacc_2d == Mt * Pt * Kt, "mmulacc.2d count mismatch");

    free(A); free(B); free(C0); free(C_hw); free(C_ref);
    return true;
}

/******************************************************************************
 * Test 8b: GEMM kernels reject dimensions that are not multiples of N
 ******************************************************************************/
static bool test_gemm_rejects_unpadded_dims(void)
{
    printf(COLOR_BLUE "[RUN] test_gemm_rejects_unpadded_dims" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    const int bad = N + 1;
    const int good = 2 * N;

    float *fbuf = (float *)calloc((size_t)good * good * 4, sizeof(float));
    int32_t *ibuf = (int32_t *)calloc((size_t)good * good * 4, sizeof(int32_t));

    TEST_ASSERT(ztt_sgemm(&ame, bad, good, good, fbuf, fbuf, fbuf) == AME_ERR_ILLEGAL_INSN, "sgemm must reject M % N != 0");
    TEST_ASSERT(ztt_sgemm(&ame, good, bad, good, fbuf, fbuf, fbuf) == AME_ERR_ILLEGAL_INSN, "sgemm must reject K % N != 0");
    TEST_ASSERT(ztt_sgemm(&ame, good, good, bad, fbuf, fbuf, fbuf) == AME_ERR_ILLEGAL_INSN, "sgemm must reject P % N != 0");
    TEST_ASSERT(ztt_igemm(&ame, bad, good, good, ibuf, ibuf, ibuf) == AME_ERR_ILLEGAL_INSN, "igemm must reject M % N != 0");
    TEST_ASSERT(ztt_igemm(&ame, good, bad, good, ibuf, ibuf, ibuf) == AME_ERR_ILLEGAL_INSN, "igemm must reject K % N != 0");
    TEST_ASSERT(ztt_igemm(&ame, good, good, bad, ibuf, ibuf, ibuf) == AME_ERR_ILLEGAL_INSN, "igemm must reject P % N != 0");

    free(fbuf);
    free(ibuf);
    return true;
}

/******************************************************************************
 * Instruction and traffic statistics
 ******************************************************************************/

/* Runs a GEMM of the given padded size and checks the counters against the
 * closed-form instruction count of the tiled algorithm. */
static bool check_gemm_stats(ame_state_t *ame, bool use_float, int M, int K, int P)
{
    const int N = AME_N;
    const unsigned long long Mt = M / N, Kt = K / N, Pt = P / N;
    const unsigned long long elem = 4;
    const unsigned long long tile_bytes = (unsigned long long)AME_NELEM * elem;

    float *fA = (float *)calloc((size_t)M * K, sizeof(float));
    float *fB = (float *)calloc((size_t)K * P, sizeof(float));
    float *fC = (float *)calloc((size_t)M * P, sizeof(float));
    ame_stats_reset(ame);

    ame_err_t err = use_float
                        ? ztt_sgemm(ame, M, K, P, fC, fA, fB)
                        : ztt_igemm(ame, M, K, P, (int32_t *)fC, (const int32_t *)fA, (const int32_t *)fB);
    free(fA);
    free(fB);
    free(fC);
    TEST_ASSERT(err == AME_OK, "gemm failed");

    const ame_stats_t *st = &ame->stats;
    const unsigned long long loads = Mt * Pt * (1 + 2 * Kt);
    TEST_ASSERT(st->mls_rm == loads, "mls.rm count mismatch");
    TEST_ASSERT(st->mss_rm == Mt * Pt, "mss.rm count mismatch");
    TEST_ASSERT(st->mmulacc_2d == Mt * Pt * Kt, "mmulacc.2d count mismatch");
    TEST_ASSERT(st->mmov_a_m == Mt * Pt, "mmov.a.m count mismatch");
    TEST_ASSERT(st->mmov_m_a == Mt * Pt, "mmov.m.a count mismatch");
    TEST_ASSERT(st->bytes_loaded == loads * tile_bytes, "bytes_loaded mismatch");
    TEST_ASSERT(st->bytes_stored == Mt * Pt * tile_bytes, "bytes_stored mismatch");
    TEST_ASSERT(st->macs == (unsigned long long)M * K * P, "macs mismatch");
    return true;
}

static bool test_gemm_stats_counts(void)
{
    printf(COLOR_BLUE "[RUN] test_gemm_stats_counts" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    const int shapes[][3] = {{N, N, N}, {2 * N, 3 * N, N}, {N, 2 * N, 3 * N}, {3 * N, N, 2 * N}};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        TEST_ASSERT(check_gemm_stats(&ame, true, shapes[s][0], shapes[s][1], shapes[s][2]), "sgemm stats");
        TEST_ASSERT(check_gemm_stats(&ame, false, shapes[s][0], shapes[s][1], shapes[s][2]), "igemm stats");
    }
    return true;
}

static bool test_stats_ignore_failed_instructions(void)
{
    printf(COLOR_BLUE "[RUN] test_stats_ignore_failed_instructions" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);

    float tile[AME_NELEM] = {0};

    /* Not owned: the load fails and must not count. */
    TEST_ASSERT(ame_mls_rm(&ame, m0, tile) == AME_ERR_NOT_OWNED, "mls.rm must fail when not owned");
    TEST_ASSERT(ame.stats.mls_rm == 0 && ame.stats.bytes_loaded == 0, "failed mls.rm must not count");

    /* Datatype mismatch: the move fails and must not count. */
    ame_acquire(&ame, NULL, 0);
    ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    ame_asettyp(&ame, acc0, AME_DTYPE_INT32);
    TEST_ASSERT(ame_mmov_m_a(&ame, m0, acc0) == AME_ERR_UNSUPPORTED, "mismatched mmov.m.a must fail");
    TEST_ASSERT(ame_mmov_a_m(&ame, acc0, m0) == AME_ERR_UNSUPPORTED, "mismatched mmov.a.m must fail");
    TEST_ASSERT(ame.stats.mmov_m_a == 0 && ame.stats.mmov_a_m == 0, "failed mmov must not count");

    /* Mismatched multiply: fails and must not count MACs. */
    ame_msettyp(&ame, m1, AME_DTYPE_INT32);
    TEST_ASSERT(ame_mmulacc_2d(&ame, acc0, m0, m1) == AME_ERR_UNSUPPORTED, "mismatched mmulacc.2d must fail");
    TEST_ASSERT(ame.stats.mmulacc_2d == 0 && ame.stats.macs == 0, "failed mmulacc.2d must not count");
    return true;
}

static bool test_stats_reset(void)
{
    printf(COLOR_BLUE "[RUN] test_stats_reset" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    TEST_ASSERT(check_gemm_stats(&ame, true, AME_N, AME_N, AME_N), "gemm stats");
    TEST_ASSERT(ame.stats.mmulacc_2d != 0, "counters must be non-zero after a GEMM");

    ame_stats_reset(&ame);
    const ame_stats_t *st = &ame.stats;
    TEST_ASSERT(st->mls_rm == 0 && st->mss_rm == 0 && st->mmulacc_2d == 0 &&
                    st->mmov_m_a == 0 && st->mmov_a_m == 0 &&
                    st->bytes_loaded == 0 && st->bytes_stored == 0 && st->macs == 0,
                "all counters must be zero after reset");
    return true;
}

static bool test_stats_report(void)
{
    printf(COLOR_BLUE "[RUN] test_stats_report" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int M = 2 * AME_N, K = AME_N, P = AME_N;
    TEST_ASSERT(check_gemm_stats(&ame, true, M, K, P), "gemm stats");

    FILE *f = tmpfile();
    TEST_ASSERT(f != NULL, "tmpfile failed");
    ame_stats_report(&ame, f);
    rewind(f);
    char text[2048] = {0};
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    fclose(f);
    text[n] = '\0';

    char expected[64];
    snprintf(expected, sizeof(expected), "%llu", (unsigned long long)M * K * P);
    TEST_ASSERT(strstr(text, expected) != NULL, "report must show the MAC count");

    double intensity = (double)ame.stats.macs / (double)(ame.stats.bytes_loaded + ame.stats.bytes_stored);
    snprintf(expected, sizeof(expected), "%.2f", intensity);
    TEST_ASSERT(strstr(text, expected) != NULL, "report must show arithmetic intensity (MACs per byte)");

    double loads_per_mma = (double)ame.stats.mls_rm / (double)ame.stats.mmulacc_2d;
    snprintf(expected, sizeof(expected), "%.2f", loads_per_mma);
    TEST_ASSERT(strstr(text, expected) != NULL, "report must show tile loads per mmulacc.2d");
    return true;
}

/******************************************************************************
 * Host-side packing statistics
 ******************************************************************************/
static bool test_pack_stats_unaligned(void)
{
    printf(COLOR_BLUE "[RUN] test_pack_stats_unaligned" COLOR_RESET "\n");
    const int N = AME_N;
    const int rows = N + 1, cols = N;
    float *src = (float *)calloc((size_t)rows * cols, sizeof(float));
    ztt_pack_stats_t st = {0};

    void *packed = pack_tiles(src, rows, cols, sizeof(float), &st);
    TEST_ASSERT(st.bytes_read == (uint64_t)rows * cols * sizeof(float), "pack must read only the logical elements");
    TEST_ASSERT(st.bytes_written == (uint64_t)(2 * N) * N * sizeof(float), "pack must write the full padded buffer");

    free(packed);
    free(src);
    return true;
}

static bool test_pack_stats_aligned(void)
{
    printf(COLOR_BLUE "[RUN] test_pack_stats_aligned" COLOR_RESET "\n");
    const int N = AME_N;
    const int rows = 2 * N, cols = 3 * N;
    int32_t *src = (int32_t *)calloc((size_t)rows * cols, sizeof(int32_t));
    ztt_pack_stats_t st = {0};

    void *packed = pack_tiles(src, rows, cols, sizeof(int32_t), &st);
    TEST_ASSERT(st.bytes_read == (uint64_t)rows * cols * sizeof(int32_t), "pack read bytes mismatch");
    TEST_ASSERT(st.bytes_written == st.bytes_read, "aligned pack must write exactly what it reads");

    free(packed);
    free(src);
    return true;
}

static bool test_unpack_stats(void)
{
    printf(COLOR_BLUE "[RUN] test_unpack_stats" COLOR_RESET "\n");
    const int N = AME_N;
    const int rows = N + 1, cols = N + 2;
    float *src = (float *)calloc((size_t)rows * cols, sizeof(float));
    float *dst = (float *)calloc((size_t)rows * cols, sizeof(float));
    void *packed = pack_tiles(src, rows, cols, sizeof(float), NULL);

    ztt_pack_stats_t st = {0};
    unpack_tiles(packed, rows, cols, sizeof(float), dst, &st);
    TEST_ASSERT(st.bytes_read == (uint64_t)rows * cols * sizeof(float), "unpack must read only the logical elements");
    TEST_ASSERT(st.bytes_written == st.bytes_read, "unpack must write the logical elements");

    free(packed);
    free(src);
    free(dst);
    return true;
}

static bool test_pack_stats_accumulate(void)
{
    printf(COLOR_BLUE "[RUN] test_pack_stats_accumulate" COLOR_RESET "\n");
    const int N = AME_N;
    float *src = (float *)calloc((size_t)N * N, sizeof(float));
    ztt_pack_stats_t st = {0};

    free(pack_tiles(src, N, N, sizeof(float), &st));
    free(pack_tiles(src, N, N, sizeof(float), &st));
    const uint64_t one_call = (uint64_t)N * N * sizeof(float);
    TEST_ASSERT(st.bytes_read == 2 * one_call, "stats must accumulate across calls (read)");
    TEST_ASSERT(st.bytes_written == 2 * one_call, "stats must accumulate across calls (written)");

    free(src);
    return true;
}

/******************************************************************************
 * Test 9: Register Aliases Readability & Functionality
 ******************************************************************************/
static bool test_register_aliases(void)
{
    printf(COLOR_BLUE "[RUN] test_register_aliases" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    uint32_t rd = 0;
    ame_acquire(&ame, &rd, 0);

    /* Use symbolic aliases m0, m1, and acc0 */
    ame_err_t err = ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "ame_msettyp with m0 must succeed");

    err = ame_msettyp(&ame, m1, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "ame_msettyp with m1 must succeed");

    err = ame_asettyp(&ame, acc0, AME_DTYPE_FP32);
    TEST_ASSERT(err == AME_OK, "ame_asettyp with acc0 must succeed");

    err = ame_mzero_2d_acc(&ame, acc0);
    TEST_ASSERT(err == AME_OK, "ame_mzero_2d_acc with acc0 must succeed");

    err = ame_mmulacc_2d(&ame, acc0, m0, m1);
    TEST_ASSERT(err == AME_OK, "ame_mmulacc_2d with acc0, m0, m1 must succeed");

    err = ame_mmov_m_a(&ame, m0, acc0);
    TEST_ASSERT(err == AME_OK, "ame_mmov_m_a with m0, acc0 must succeed");

    return true;
}

/******************************************************************************
 * Test 14: Multi-Register Strided Kernels (2x4, 4x4, and Double-Buffering)
 ******************************************************************************/
static bool test_multi_register_gemm(void)
{
    printf(COLOR_BLUE "[RUN] test_multi_register_gemm (2x4, 4x4 and Double-Buffering variants)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int N = AME_N;
    /* Dimensions multiple of 4*N to accommodate 1x1, 2x4, and 4x4 cleanly */
    const int M = 4 * N;
    const int K = 4 * N;
    const int P = 4 * N;

    const size_t lda = K + 2;
    const size_t ldb = P + 4;
    const size_t ldc = P + 1;

    float *A = (float *)calloc((size_t)M * lda, sizeof(float));
    float *B = (float *)calloc((size_t)K * ldb, sizeof(float));
    float *C0 = (float *)calloc((size_t)M * ldc, sizeof(float));
    float *C_ref = (float *)calloc((size_t)M * ldc, sizeof(float));
    float *C_test = (float *)calloc((size_t)M * ldc, sizeof(float));

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[i * lda + k] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    for (int k = 0; k < K; ++k)
        for (int j = 0; j < P; ++j)
            B[k * ldb + j] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            C0[i * ldc + j] = ((float)(rand() % 100) - 50.0f) / 10.0f;

    /* Ground truth C_ref += A * B */
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            float sum = C0[i * ldc + j];
            for (int k = 0; k < K; ++k)
                sum += A[i * lda + k] * B[k * ldb + j];
            C_ref[i * ldc + j] = sum;
        }
    }

    const uint64_t Mt = M / N, Kt = K / N, Pt = P / N;
    const uint64_t expected_macs = Mt * Kt * Pt;

    /* 1. Test 2x4 without prefetch */
    memcpy(C_test, C0, (size_t)M * ldc * sizeof(float));
    ame_stats_reset(&ame);
    ame_err_t err = ztt_sgemm_strided_2x4(&ame, M, K, P, C_test, A, B, ldc, lda, ldb);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm_strided_2x4 failed");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            TEST_ASSERT(fabsf(C_test[i * ldc + j] - C_ref[i * ldc + j]) < 1e-4f, "2x4 numerical mismatch");
    TEST_ASSERT(ame.stats.mmulacc_2d == expected_macs, "2x4 MAC count mismatch");
    /* In 2x4: C is loaded Mt*Pt times. A is loaded (Mt/2)*Pt*Kt*2 = Mt*Pt*Kt.
       B is loaded Mt*(Pt/4)*Kt*4 = Mt*Pt*Kt. Total loads: Mt*Pt + (1/4 + 1/2)*Mt*Pt*Kt*something.
       More precisely: outer loop Mt/2 * Pt/4.
       Per (i,j) block of 2x4 tiles: 2*4 = 8 C loads.
       Inner loop k (Kt times): 2 loads for A, 4 loads for B -> (2+4)*Kt = 6*Kt loads.
       Total mls_st = (Mt/2) * (Pt/4) * (8 + 6 * Kt). */
    const uint64_t expected_loads_2x4 = (Mt / 2) * (Pt / 4) * (8 + 6 * Kt);
    TEST_ASSERT(ame.stats.mls_st == expected_loads_2x4, "2x4 mls.st load count mismatch");

    /* 2. Test 2x4 with double buffering (prefetch) */
    memcpy(C_test, C0, (size_t)M * ldc * sizeof(float));
    ame_stats_reset(&ame);
    err = ztt_sgemm_strided_2x4_db(&ame, M, K, P, C_test, A, B, ldc, lda, ldb);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm_strided_2x4_db failed");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            TEST_ASSERT(fabsf(C_test[i * ldc + j] - C_ref[i * ldc + j]) < 1e-4f, "2x4_db numerical mismatch");
    TEST_ASSERT(ame.stats.mmulacc_2d == expected_macs, "2x4_db MAC count mismatch");
    TEST_ASSERT(ame.stats.mls_st == expected_loads_2x4, "2x4_db mls.st load count mismatch");

    /* 3. Test 4x4 without prefetch */
    memcpy(C_test, C0, (size_t)M * ldc * sizeof(float));
    ame_stats_reset(&ame);
    err = ztt_sgemm_strided_4x4(&ame, M, K, P, C_test, A, B, ldc, lda, ldb);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm_strided_4x4 failed");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            TEST_ASSERT(fabsf(C_test[i * ldc + j] - C_ref[i * ldc + j]) < 1e-4f, "4x4 numerical mismatch");
    TEST_ASSERT(ame.stats.mmulacc_2d == expected_macs, "4x4 MAC count mismatch");
    /* In 4x4: (Mt/4) * (Pt/4) blocks.
       Per block: 16 C loads.
       Inner loop k (Kt times): 4 loads for A, 4 loads for B -> 8*Kt loads.
       Total mls_st = (Mt/4) * (Pt/4) * (16 + 8 * Kt). */
    const uint64_t expected_loads_4x4 = (Mt / 4) * (Pt / 4) * (16 + 8 * Kt);
    TEST_ASSERT(ame.stats.mls_st == expected_loads_4x4, "4x4 mls.st load count mismatch");

    /* 4. Test 4x4 with double buffering (prefetch) */
    memcpy(C_test, C0, (size_t)M * ldc * sizeof(float));
    ame_stats_reset(&ame);
    err = ztt_sgemm_strided_4x4_db(&ame, M, K, P, C_test, A, B, ldc, lda, ldb);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm_strided_4x4_db failed");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            TEST_ASSERT(fabsf(C_test[i * ldc + j] - C_ref[i * ldc + j]) < 1e-4f, "4x4_db numerical mismatch");
    TEST_ASSERT(ame.stats.mmulacc_2d == expected_macs, "4x4_db MAC count mismatch");
    TEST_ASSERT(ame.stats.mls_st == expected_loads_4x4, "4x4_db mls.st load count mismatch");

    /* 5. Alignment check: non-multiples of 2N or 4N must return AME_ERR_ILLEGAL_INSN */
    TEST_ASSERT(ztt_sgemm_strided_2x4(&ame, N, K, P, C_test, A, B, ldc, lda, ldb) == AME_ERR_ILLEGAL_INSN,
                "2x4 must reject M not multiple of 2N");
    TEST_ASSERT(ztt_sgemm_strided_4x4(&ame, 2 * N, K, P, C_test, A, B, ldc, lda, ldb) == AME_ERR_ILLEGAL_INSN,
                "4x4 must reject M not multiple of 4N");

    free(A); free(B); free(C0); free(C_ref); free(C_test);
    return true;
}

/******************************************************************************
 * Main Runner
 ******************************************************************************/
static void run_test(bool (*test_func)(void), const char *name)
{
    tests_run++;
    if (test_func())
    {
        printf(COLOR_GREEN "  [PASS] %s" COLOR_RESET "\n", name);
        tests_passed++;
    }
    else
    {
        printf(COLOR_RED "  [FAIL] %s" COLOR_RESET "\n", name);
    }
}

int main(void)
{
    srand(42);
    printf("============================================================\n");
    printf("  RISC-V AME Mini-Simulator - Phase 1 Test Suite\n");
    printf("  Config: AME_NELEM=%d, N=%d, UNIT_SIZE=%d bits\n", AME_NELEM, AME_N, AME_UNIT_DATATYPE_SIZE);
    printf("============================================================\n\n");

    run_test(test_ownership, "Ownership and backend access control");
    run_test(test_datatypes_and_zeroing, "Datatype descriptors and register zeroing");
    run_test(test_memory_roundtrip, "Row-major memory load/store roundtrip");
    run_test(test_strided_memory_roundtrip, "Strided memory load/store roundtrip");
    run_test(test_type_mismatch, "Datatype mismatch & amestatus.UN flag");
    run_test(test_single_tile_sgemm, "Single-tile FP32 SGEMM");
    run_test(test_single_tile_igemm, "Single-tile Int32 IGEMM");
    run_test(test_full_sgemm, "Multi-tile End-to-End SGEMM");
    run_test(test_full_igemm, "Multi-tile End-to-End IGEMM");
    run_test(test_full_strided_gemm, "Multi-tile Strided SGEMM (mls.st/mss.st)");
    run_test(test_gemm_rejects_unpadded_dims, "GEMM rejects dimensions not multiple of N");
    run_test(test_gemm_stats_counts, "GEMM instruction and traffic counts");
    run_test(test_stats_ignore_failed_instructions, "Failed instructions are not counted");
    run_test(test_stats_reset, "Statistics reset");
    run_test(test_stats_report, "Statistics report");
    run_test(test_pack_stats_unaligned, "Pack statistics, padded shape");
    run_test(test_pack_stats_aligned, "Pack statistics, aligned shape");
    run_test(test_unpack_stats, "Unpack statistics");
    run_test(test_pack_stats_accumulate, "Pack statistics accumulate across calls");
    run_test(test_register_aliases, "Register aliases m0, m1, acc0 functionality");
    run_test(test_multi_register_gemm, "Multi-register strided kernels (2x4, 4x4, DB)");

    printf("\n============================================================\n");
    printf("  Summary: %d / %d tests passed (%s)\n",
           tests_passed, tests_run,
           (tests_passed == tests_run) ? COLOR_GREEN "SUCCESS" COLOR_RESET : COLOR_RED "FAILURE" COLOR_RESET);
    printf("============================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
