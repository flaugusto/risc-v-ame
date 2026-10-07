#ifndef PACKING_H
#define PACKING_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "ztt_ame.h"

/******************************************************************************
 * Host-side helpers that convert row-major matrices to and from the
 * block-tiled row-major layout consumed by the GEMM kernels: each N x N tile
 * is contiguous and row-major, and the tile grid is row-major.
 * Used by tests and benchmarks; not part of the kernel library.
 ******************************************************************************/

/* Host-side memory traffic of packing and unpacking, accumulated across calls. */
typedef struct
{
    uint64_t bytes_read;
    uint64_t bytes_written;
} ztt_pack_stats_t;

static inline int round_up_to_tile(int x)
{
    return ((x + AME_N - 1) / AME_N) * AME_N;
}

/* Packs a rows x cols row-major matrix into a zero-padded tile-major buffer.
 * stats may be NULL. Reads count the logical elements; writes count the whole
 * padded buffer, including the zero fill. */
static inline void *pack_tiles(const void *src, int rows, int cols, size_t elem_size,
                               ztt_pack_stats_t *stats)
{
    const int N = AME_N;
    const int prows = round_up_to_tile(rows);
    const int pcols = round_up_to_tile(cols);
    uint8_t *dst = (uint8_t *)calloc((size_t)prows * pcols, elem_size);
    const uint8_t *in = (const uint8_t *)src;

    if (stats)
    {
        stats->bytes_read += (uint64_t)rows * cols * elem_size;
        stats->bytes_written += (uint64_t)prows * pcols * elem_size;
    }

    for (int ti = 0; ti < prows / N; ++ti)
    {
        for (int tj = 0; tj < pcols / N; ++tj)
        {
            uint8_t *tile = dst + ((size_t)(ti * (pcols / N) + tj) * N * N) * elem_size;
            for (int r = 0; r < N && ti * N + r < rows; ++r)
            {
                for (int c = 0; c < N && tj * N + c < cols; ++c)
                {
                    memcpy(tile + ((size_t)r * N + c) * elem_size,
                           in + ((size_t)(ti * N + r) * cols + (tj * N + c)) * elem_size,
                           elem_size);
                }
            }
        }
    }
    return dst;
}

/* Extracts the rows x cols region of a tile-major buffer into a row-major matrix.
 * stats may be NULL. */
static inline void unpack_tiles(const void *packed, int rows, int cols, size_t elem_size, void *dst,
                                ztt_pack_stats_t *stats)
{
    const int N = AME_N;
    const int pcols = round_up_to_tile(cols);
    const uint8_t *in = (const uint8_t *)packed;
    uint8_t *out = (uint8_t *)dst;

    if (stats)
    {
        stats->bytes_read += (uint64_t)rows * cols * elem_size;
        stats->bytes_written += (uint64_t)rows * cols * elem_size;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            const uint8_t *tile = in + ((size_t)((i / N) * (pcols / N) + (j / N)) * N * N) * elem_size;
            memcpy(out + ((size_t)i * cols + j) * elem_size,
                   tile + ((size_t)(i % N) * N + (j % N)) * elem_size,
                   elem_size);
        }
    }
}

#endif /* PACKING_H */
