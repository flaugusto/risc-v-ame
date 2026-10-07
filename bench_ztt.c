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
    return 0;
}
