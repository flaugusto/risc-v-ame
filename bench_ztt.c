#include <stdio.h>
#include <stdlib.h>
#include "ztt_ame.h"
#include "ztt_gemm.h"
#include "packing.h"

/* Counts instructions and traffic of ztt_sgemm for one logical shape.
 * The matrices are padded to multiples of N before the kernel runs. */
/* Runs both the software-packed (mls.rm) and hardware-strided (mls.st) SGEMM
 * and prints direct comparison metrics. */
static void run_shape_comparison(ame_state_t *ame, int M, int K, int P)
{
    const int Mp = round_up_to_tile(M);
    const int Kp = round_up_to_tile(K);
    const int Pp = round_up_to_tile(P);

    /* Allocate row-major matrices with standard leading dimensions */
    float *A = (float *)calloc((size_t)Mp * Kp, sizeof(float));
    float *B = (float *)calloc((size_t)Kp * Pp, sizeof(float));
    float *C = (float *)calloc((size_t)Mp * Pp, sizeof(float));

    /* --- Approach 1: Software Tile-Major Packing --- */
    ztt_pack_stats_t host_p1 = {0};
    float *Ap = (float *)pack_tiles(A, M, K, sizeof(float), &host_p1);
    float *Bp = (float *)pack_tiles(B, K, P, sizeof(float), &host_p1);
    float *Cp = (float *)pack_tiles(C, M, P, sizeof(float), &host_p1);

    ame_stats_reset(ame);
    ztt_sgemm(ame, Mp, Kp, Pp, Cp, Ap, Bp);
    unpack_tiles(Cp, M, P, sizeof(float), C, &host_p1);

    const ame_stats_t st_p1 = ame->stats;
    const uint64_t ame_bytes_p1 = st_p1.bytes_loaded + st_p1.bytes_stored;
    const uint64_t host_bytes_p1 = host_p1.bytes_read + host_p1.bytes_written;
    const uint64_t total_bytes_p1 = ame_bytes_p1 + host_bytes_p1;

    free(Ap);
    free(Bp);
    free(Cp);

    /* --- Approach 2: Direct In-Place Hardware Strided --- */
    /* If dimensions are not multiples of N, hardware strided operates on padded buffer in-place */
    ame_stats_reset(ame);
    ztt_sgemm_strided(ame, Mp, Kp, Pp, C, A, B, Pp, Kp, Pp);

    const ame_stats_t st_p2 = ame->stats;
    const uint64_t ame_bytes_p2 = st_p2.bytes_loaded + st_p2.bytes_stored;
    /* In Phase 2, if matrices are allocated or aligned to multiples of N, host packing is 0 */
    const uint64_t host_bytes_p2 = 0;
    const uint64_t total_bytes_p2 = ame_bytes_p2 + host_bytes_p2;

    const double traffic_reduction = (1.0 - (double)total_bytes_p2 / (double)total_bytes_p1) * 100.0;

    printf("%4dx%4dx%4d | %8llu %10llu %10llu | %8llu %10llu %10llu | %10.1f%%\n",
           M, K, P,
           (unsigned long long)st_p1.mls_rm,
           (unsigned long long)ame_bytes_p1,
           (unsigned long long)host_bytes_p1,
           (unsigned long long)st_p2.mls_st,
           (unsigned long long)ame_bytes_p2,
           (unsigned long long)host_bytes_p2,
           traffic_reduction);

    free(A);
    free(B);
    free(C);
}

static void run_multireg_benchmark(ame_state_t *ame, int M, int K, int P)
{
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)K * P, sizeof(float));
    float *C = (float *)calloc((size_t)M * P, sizeof(float));

    /* 1. Baseline 1x1 */
    ame_stats_reset(ame);
    ztt_sgemm_strided(ame, M, K, P, C, A, B, P, K, P);
    const ame_stats_t st_1x1 = ame->stats;

    /* 2. 2x4 No Prefetch */
    ame_stats_reset(ame);
    ztt_sgemm_strided_2x4(ame, M, K, P, C, A, B, P, K, P);
    const ame_stats_t st_2x4 = ame->stats;

    /* 3. 2x4 Double-Buffered (Prefetch) */
    ame_stats_reset(ame);
    ztt_sgemm_strided_2x4_db(ame, M, K, P, C, A, B, P, K, P);
    const ame_stats_t st_2x4_db = ame->stats;

    /* 4. 4x4 No Prefetch */
    ame_stats_reset(ame);
    ztt_sgemm_strided_4x4(ame, M, K, P, C, A, B, P, K, P);
    const ame_stats_t st_4x4 = ame->stats;

    /* 5. 4x4 Double-Buffered (Prefetch) */
    ame_stats_reset(ame);
    ztt_sgemm_strided_4x4_db(ame, M, K, P, C, A, B, P, K, P);
    const ame_stats_t st_4x4_db = ame->stats;

    const double tile_macs = (double)st_1x1.mmulacc_2d;
    const uint64_t scalar_macs = (uint64_t)M * K * P;

    printf("\n>>> Matrix Shape: %dx%dx%d (Tile Grid: %dx%dx%d | Scalar MACs: %llu | Tile-MACs: %.0f)\n",
           M, K, P, M / AME_N, K / AME_N, P / AME_N,
           (unsigned long long)scalar_macs, tile_macs);
    printf("----------------------------------------------------------------------------------------------------------\n");
    printf(" Kernel Configuration        | mls.st Loads | Total Bytes   | Tile-MAC / Load  | Load Reduction | Traffic Savings\n");
    printf("-----------------------------+--------------+---------------+------------------+----------------+----------------\n");

    struct {
        const char *name;
        ame_stats_t st;
    } configs[] = {
        {"1x1 Baseline (1 acc, 2 m)", st_1x1},
        {"2x4 (8 acc, 6 m)", st_2x4},
        {"2x4 Prefetch (8 acc, 12 m)", st_2x4_db},
        {"4x4 (16 acc, 8 m)", st_4x4},
        {"4x4 Prefetch (16 acc, 16 m)", st_4x4_db},
    };

    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        uint64_t loads = configs[i].st.mls_st;
        uint64_t bytes = configs[i].st.bytes_loaded + configs[i].st.bytes_stored;
        double ratio = tile_macs / (double)loads;
        double load_red = (1.0 - (double)loads / (double)st_1x1.mls_st) * 100.0;
        double traffic_sav = (1.0 - (double)bytes / (double)(st_1x1.bytes_loaded + st_1x1.bytes_stored)) * 100.0;

        printf(" %-27s | %12llu | %10llu KB | %16.2f | %13.1f%% | %14.1f%%\n",
               configs[i].name,
               (unsigned long long)loads,
               (unsigned long long)(bytes / 1024),
               ratio,
               load_red,
               traffic_sav);
    }
    printf("----------------------------------------------------------------------------------------------------------\n");

    free(A); free(B); free(C);
}

int main(void)
{
    const int N = AME_N;
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    printf("========================================================================================\n");
    printf("  SGEMM Evaluation: Software Packing (Phase 1) vs. Hardware Strided (Phase 2)\n");
    printf("  Config: AME_NELEM=%d, N=%d\n", AME_NELEM, N);
    printf("========================================================================================\n");
    printf("  Shape (MxKxP) | ---- Phase 1 (Software Packed) ---- | ---- Phase 2 (Hardware Strided) ---- | Total Traffic\n");
    printf("                |  mls.rm     AME Bytes   Host Bytes   |  mls.st     AME Bytes   Host Bytes   |   Savings\n");
    printf("----------------+--------------------------------------+--------------------------------------+--------------\n");

    const int shapes[][3] = {
        {N, N, N},
        {2 * N, 2 * N, 2 * N},
        {4 * N, 4 * N, 4 * N},
        {8 * N, 8 * N, 8 * N},
        {1, 4 * N, 4 * N},
        {4 * N, 4 * N, 1},
        {N + 1, N + 1, N + 1},
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        run_shape_comparison(&ame, shapes[s][0], shapes[s][1], shapes[s][2]);
    }
    printf("========================================================================================\n");

    printf("\n======================================================================================================\n");
    printf("  PART 2: Multi-Register SGEMM & Double-Buffering Architectural Evaluation\n");
    printf("  Comparing 1x1 Baseline, 2x4, and 4x4 Accumulator Tiling with Register Prefetching\n");
    printf("======================================================================================================\n");

    const int multireg_shapes[][3] = {
        {4 * N, 4 * N, 4 * N},
        {8 * N, 8 * N, 8 * N},
        {16 * N, 16 * N, 16 * N},
    };

    for (size_t s = 0; s < sizeof(multireg_shapes) / sizeof(multireg_shapes[0]); ++s)
    {
        run_multireg_benchmark(&ame, multireg_shapes[s][0], multireg_shapes[s][1], multireg_shapes[s][2]);
    }

    return 0;
}
