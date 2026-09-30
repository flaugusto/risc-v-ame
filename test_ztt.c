#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "ztt_ame.h"
#include "ztt_gemm.h"

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
 * Test 7: Multi-Tile End-to-End SGEMM (M x K x P)
 ******************************************************************************/
static bool test_full_sgemm(void)
{
    printf(COLOR_BLUE "[RUN] test_full_sgemm (Multi-tile M=24, K=20, P=28)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int M = 24;
    const int K = 20;
    const int P = 28;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * P * sizeof(float));
    float *C_hw = (float *)calloc(M * P, sizeof(float));
    float *C_ref = (float *)calloc(M * P, sizeof(float));

    /* Populate matrices */
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;
    }
    for (int i = 0; i < K * P; ++i)
    {
        B[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;
    }

    /* CPU Reference: C += A * B */
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * P + j];
            }
            C_ref[i * P + j] = sum;
        }
    }

    /* Ztt Accelerator GEMM */
    ame_err_t err = ztt_sgemm(&ame, M, K, P, C_hw, A, B);
    TEST_ASSERT(err == AME_OK, "ztt_sgemm failed");

    /* Validate results */
    float max_diff = 0.0f;
    for (int i = 0; i < M * P; ++i)
    {
        float diff = fabsf(C_hw[i] - C_ref[i]);
        if (diff > max_diff)
            max_diff = diff;
        TEST_ASSERT(diff < 1e-4f, "Multi-tile SGEMM result differs from CPU reference");
    }

    printf("       Max diff: %e\n", max_diff);

    free(A);
    free(B);
    free(C_hw);
    free(C_ref);
    return true;
}

/******************************************************************************
 * Test 8: Multi-Tile End-to-End IGEMM (M x K x P)
 ******************************************************************************/
static bool test_full_igemm(void)
{
    printf(COLOR_BLUE "[RUN] test_full_igemm (Multi-tile M=24, K=20, P=28)" COLOR_RESET "\n");
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    const int M = 24;
    const int K = 20;
    const int P = 28;

    int32_t *A = (int32_t *)malloc(M * K * sizeof(int32_t));
    int32_t *B = (int32_t *)malloc(K * P * sizeof(int32_t));
    int32_t *C_hw = (int32_t *)calloc(M * P, sizeof(int32_t));
    int32_t *C_ref = (int32_t *)calloc(M * P, sizeof(int32_t));

    for (int i = 0; i < M * K; ++i)
    {
        A[i] = (rand() % 20) - 10;
    }
    for (int i = 0; i < K * P; ++i)
    {
        B[i] = (rand() % 20) - 10;
    }

    /* CPU Reference */
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < P; ++j)
        {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * P + j];
            }
            C_ref[i * P + j] = sum;
        }
    }

    /* Ztt Accelerator GEMM */
    ame_err_t err = ztt_igemm(&ame, M, K, P, C_hw, A, B);
    TEST_ASSERT(err == AME_OK, "ztt_igemm failed");

    /* Exact integer match validation */
    for (int i = 0; i < M * P; ++i)
    {
        TEST_ASSERT(C_hw[i] == C_ref[i], "Multi-tile IGEMM result differs from CPU reference");
    }

    free(A);
    free(B);
    free(C_hw);
    free(C_ref);
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
    run_test(test_type_mismatch, "Datatype mismatch & amestatus.UN flag");
    run_test(test_single_tile_sgemm, "Single-tile FP32 SGEMM");
    run_test(test_single_tile_igemm, "Single-tile Int32 IGEMM");
    run_test(test_full_sgemm, "Multi-tile End-to-End SGEMM");
    run_test(test_full_igemm, "Multi-tile End-to-End IGEMM");
    run_test(test_register_aliases, "Register aliases m0, m1, acc0 functionality");

    printf("\n============================================================\n");
    printf("  Summary: %d / %d tests passed (%s)\n",
           tests_passed, tests_run,
           (tests_passed == tests_run) ? COLOR_GREEN "SUCCESS" COLOR_RESET : COLOR_RED "FAILURE" COLOR_RESET);
    printf("============================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
