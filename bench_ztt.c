#include <stdio.h>
#include <stdlib.h>
#include "ztt_ame.h"
#include "ztt_gemm.h"
#include "ztt_tile_pack.h"

/* Counts instructions and traffic of ztt_sgemm for one logical shape.
 * The matrices are padded to multiples of N before the kernel runs. */
static void run_shape(ame_state_t *ame, int M, int K, int P)
{
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)K * P, sizeof(float));
    float *C = (float *)calloc((size_t)M * P, sizeof(float));

    ztt_pack_stats_t host = {0};
    float *Ap = (float *)pack_tiles(A, M, K, sizeof(float), &host);
    float *Bp = (float *)pack_tiles(B, K, P, sizeof(float), &host);
    float *Cp = (float *)pack_tiles(C, M, P, sizeof(float), &host);

    ame_stats_reset(ame);
    ame_err_t err = ztt_sgemm(ame, round_up_to_tile(M), round_up_to_tile(K), round_up_to_tile(P), Cp, Ap, Bp);
    if (err != AME_OK)
    {
        fprintf(stderr, "ztt_sgemm failed for %dx%dx%d (error %d)\n", M, K, P, (int)err);
        exit(1);
    }

    unpack_tiles(Cp, M, P, sizeof(float), C, &host);

    const ame_stats_t *st = &ame->stats;
    const double bytes = (double)(st->bytes_loaded + st->bytes_stored);
    const double host_bytes = (double)(host.bytes_read + host.bytes_written);
    const double useful_macs = (double)M * K * P;

    printf("%6d %6d %6d | %9llu %9llu %9llu %9llu | %13llu | %8.2f %8.2f %8.1f%% | %10.0f %8.2f %9.3f\n",
           M, K, P,
           (unsigned long long)st->mls_rm, (unsigned long long)st->mss_rm,
           (unsigned long long)st->mmulacc_2d, (unsigned long long)(st->mmov_m_a + st->mmov_a_m),
           (unsigned long long)st->macs,
           (double)st->macs / bytes,
           (double)st->mls_rm / (double)st->mmulacc_2d,
           100.0 * useful_macs / (double)st->macs,
           host_bytes, host_bytes / bytes, host_bytes / useful_macs);

    free(A); free(B); free(C);
    free(Ap); free(Bp); free(Cp);
}

int main(void)
{
    const int N = AME_N;
    ame_state_t ame;
    ame_init(&ame);
    ame_acquire(&ame, NULL, 0);

    printf("SGEMM instruction and traffic counts (AME_NELEM=%d, N=%d)\n", AME_NELEM, N);
    printf("%6s %6s %6s | %9s %9s %9s %9s | %13s | %8s %8s %9s | %10s %8s %9s\n",
           "M", "K", "P", "mls.rm", "mss.rm", "mmulacc", "mmov", "MACs",
           "MAC/byte", "ld/mma", "useful", "host B", "host/AME", "host B/MAC");

    const int shapes[][3] = {
        {N, N, N},
        {2 * N, 2 * N, 2 * N},
        {4 * N, 4 * N, 4 * N},
        {1, 4 * N, 4 * N},
        {4 * N, 4 * N, 1},
        {N + 1, N + 1, N + 1},
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        run_shape(&ame, shapes[s][0], shapes[s][1], shapes[s][2]);
    }
    return 0;
}
