#ifndef ZTT_AME_H
#define ZTT_AME_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Hardware Parameters (Ztt v0.6 Chapter 4)
 ******************************************************************************/
#ifndef AME_NELEM
/* Number of elements in an AME logical square (power of 4) */
#define AME_NELEM                 64
#endif

/* Tile side length: N = sqrt(AME_NELEM) (power of 2) */
#if AME_NELEM == 16
#define AME_N                     4
#elif AME_NELEM == 64
#define AME_N                     8
#elif AME_NELEM == 256
#define AME_N                     16
#elif AME_NELEM == 1024
#define AME_N                     32
#elif AME_NELEM == 4096
#define AME_N                     64
#elif AME_NELEM == 16384
#define AME_N                     128
#else
#error "AME_NELEM must be a recognized power of 4 (16, 64, 256, 1024, 4096, 16384)"
#endif

/* Number of general matrix registers M0..M[AME_NUM_M_REGS-1] (16 or 32) */
#ifndef AME_NUM_M_REGS
#define AME_NUM_M_REGS            32
#endif

/* Number of accumulator registers Acc0..Acc[AME_NUM_ACC_REGS-1] (1, 2, 4, 8, 16) */
#ifndef AME_NUM_ACC_REGS
#define AME_NUM_ACC_REGS          8
#endif

/* Unit datatype size in bits (power of 2) */
#ifndef AME_UNIT_DATATYPE_SIZE
#define AME_UNIT_DATATYPE_SIZE    32
#endif

/* Physical size of an M register in bytes */
#define AME_M_REG_SIZE_BYTES      ((AME_NELEM * AME_UNIT_DATATYPE_SIZE) / 8)

/* Maximum supported datatype size in bits (for accumulator physical sizing) */
#define AME_MAX_DATATYPE_SIZE     64
/* Physical size of an Acc register in bytes */
#define AME_ACC_REG_SIZE_BYTES    ((AME_NELEM * AME_MAX_DATATYPE_SIZE) / 8)

/******************************************************************************
 * Register Aliases (Ztt v0.6 Section 2.2 & 2.3)
 ******************************************************************************/
/* Matrix register aliases m0..m31 */
#define m0   0
#define m1   1
#define m2   2
#define m3   3
#define m4   4
#define m5   5
#define m6   6
#define m7   7
#define m8   8
#define m9   9
#define m10  10
#define m11  11
#define m12  12
#define m13  13
#define m14  14
#define m15  15
#define m16  16
#define m17  17
#define m18  18
#define m19  19
#define m20  20
#define m21  21
#define m22  22
#define m23  23
#define m24  24
#define m25  25
#define m26  26
#define m27  27
#define m28  28
#define m29  29
#define m30  30
#define m31  31

/* Accumulator register aliases acc0..acc15 */
#define acc0  0
#define acc1  1
#define acc2  2
#define acc3  3
#define acc4  4
#define acc5  5
#define acc6  6
#define acc7  7
#define acc8  8
#define acc9  9
#define acc10 10
#define acc11 11
#define acc12 12
#define acc13 13
#define acc14 14
#define acc15 15

/******************************************************************************
 * Datatype Descriptors (Ztt v0.6 Section 2.4)
 * 32-bit specifier:
 * - Bit 31: 0 = Standard, 1 = Custom
 * - Bits 30:8: Properties
 * - Bits 7:0: Size in bits
 ******************************************************************************/

/* Standard Integer Datatypes */
#define AME_DTYPE_INT8            0x40000008U  /* Signed, Non-sat, RNU, Int, 8 bits */
#define AME_DTYPE_UINT8           0x00000008U  /* Unsigned, Non-sat, RNU, Int, 8 bits */
#define AME_DTYPE_INT16           0x40000010U  /* Signed, Non-sat, RNU, Int, 16 bits */
#define AME_DTYPE_UINT16          0x00000010U  /* Unsigned, Non-sat, RNU, Int, 16 bits */
#define AME_DTYPE_INT32           0x40000020U  /* Signed, Non-sat, RNU, Int, 32 bits */
#define AME_DTYPE_UINT32          0x00000020U  /* Unsigned, Non-sat, RNU, Int, 32 bits */
#define AME_DTYPE_INT64           0x40000040U  /* Signed, Non-sat, RNU, Int, 64 bits */
#define AME_DTYPE_UINT64          0x00000040U  /* Unsigned, Non-sat, RNU, Int, 64 bits */

/* Standard Floating-Point Datatypes */
#define AME_DTYPE_FP16            0x14300110U  /* Exp=5, RNE=0, Inf=1, Denorm=1, FP=1, 16 bits */
#define AME_DTYPE_FP32            0x20300120U  /* Exp=8, RNE=0, Inf=1, Denorm=1, FP=1, 32 bits */
#define AME_DTYPE_BF16            0x20300110U  /* Exp=8, RNE=0, Inf=1, Denorm=1, FP=1, 16 bits */
#define AME_DTYPE_FP64            0x2C300140U  /* Exp=11, RNE=0, Inf=1, Denorm=1, FP=1, 64 bits */

#define AME_DTYPE_UNINITIALIZED   0x00000000U

/* Datatype inspection helper macros */
#define AME_DTYPE_IS_CUSTOM(d)    (((d) >> 31) & 1U)
#define AME_DTYPE_IS_FP(d)        (!AME_DTYPE_IS_CUSTOM(d) && (((d) >> 8) & 1U))
#define AME_DTYPE_IS_INT(d)       (!AME_DTYPE_IS_CUSTOM(d) && !(((d) >> 8) & 1U))
#define AME_DTYPE_SIZE_BITS(d)    ((d) & 0xFFU)
#define AME_DTYPE_SIZE_BYTES(d)   ((AME_DTYPE_SIZE_BITS(d) + 7U) / 8U)
#define AME_DTYPE_INT_SIGNED(d)   (((d) >> 30) & 1U)
#define AME_DTYPE_INT_SAT(d)      (((d) >> 29) & 1U)

/******************************************************************************
 * Status & Control CSR Field Definitions (Ztt v0.6 Chapter 3 & 5)
 ******************************************************************************/
/* mstatus.MS / vsstatus.MS state */
typedef enum {
    AME_MS_OFF     = 0,
    AME_MS_INITIAL = 1,
    AME_MS_CLEAN   = 2,
    AME_MS_DIRTY   = 3
} ame_ms_state_t;

/* ameown CSR bits */
#define AME_OWN_OWNED             (1U << 0)

/* amestatus CSR bits */
#define AME_STATUS_UN             (1U << 0)  /* Unsupported-operation sticky flag */

/* amefflags CSR bits (Standard RISC-V FP flags) */
#define AME_FFLAGS_NX             (1U << 0)  /* Inexact */
#define AME_FFLAGS_UF             (1U << 1)  /* Underflow */
#define AME_FFLAGS_OF             (1U << 2)  /* Overflow */
#define AME_FFLAGS_DZ             (1U << 3)  /* Divide by zero */
#define AME_FFLAGS_NV             (1U << 4)  /* Invalid operation */

/* amexsat CSR bits */
#define AME_XSAT_SAT              (1U << 0)  /* Fixed-point saturation flag */

/* ame.acquire return codes */
#define AME_ACQ_SUCCESS           (1U << 0)
#define AME_ACQ_STATUS_GRANTED    (0x00U << 1)
#define AME_ACQ_STATUS_BUSY       (0x01U << 1)
#define AME_ACQ_STATUS_DISABLED   (0x03U << 1)
#define AME_ACQ_STATUS_BAD_DESC   (0x04U << 1)

/* Simulator error / status codes */
typedef enum {
    AME_OK                 =  0,
    AME_ERR_ILLEGAL_INSN   = -1,
    AME_ERR_UNSUPPORTED    = -2,
    AME_ERR_MISALIGNED     = -3,
    AME_ERR_NOT_OWNED      = -4,
    AME_ERR_INVALID_REG    = -5
} ame_err_t;

/******************************************************************************
 * Architectural State (Ztt v0.6 Chapter 2, 3, 5)
 ******************************************************************************/
/* Counters of successfully executed instructions and data traffic */
typedef struct {
    uint64_t mls_rm;
    uint64_t mss_rm;
    uint64_t mmulacc_2d;
    uint64_t mmov_m_a;
    uint64_t mmov_a_m;
    uint64_t bytes_loaded;  /* Bytes read from memory by mls.rm */
    uint64_t bytes_stored;  /* Bytes written to memory by mss.rm */
    uint64_t macs;          /* Multiply-accumulate operations performed by mmulacc.2d */
} ame_stats_t;

typedef struct {
    /* Matrix data registers M0..M[AME_NUM_M_REGS-1] */
    uint8_t M[AME_NUM_M_REGS][AME_M_REG_SIZE_BYTES];

    /* Accumulator data registers Acc0..Acc[AME_NUM_ACC_REGS-1] */
    uint8_t Acc[AME_NUM_ACC_REGS][AME_ACC_REG_SIZE_BYTES];

    /* Datatype metadata registers */
    uint32_t Md[AME_NUM_M_REGS];
    uint32_t Ad[AME_NUM_ACC_REGS];

    /* Control and Status Registers */
    uint32_t amenlen;   /* Read-only: N = sqrt(AME_NELEM) */
    uint32_t ameudsz;   /* Read-only: AME_UNIT_DATATYPE_SIZE */
    uint32_t amestype;  /* Scalar X operand datatype descriptor */
    uint32_t ameown;    /* Bit 0: OWNED */
    uint32_t amefflags; /* Accrued floating-point exception flags */
    uint32_t amexsat;   /* Fixed-point saturation sticky flag */
    uint32_t amestatus; /* Execution status (UN bit 0) */

    /* Hart processor state */
    ame_ms_state_t ms_status; /* mstatus.MS field */

    /* Instruction and traffic counters (not architectural state) */
    ame_stats_t stats;
} ame_state_t;

/******************************************************************************
 * API / Instruction Functions (Ztt v0.6 Chapter 5 & 6)
 ******************************************************************************/

/* Statistics */
void ame_stats_reset(ame_state_t *ame);
void ame_stats_report(const ame_state_t *ame, FILE *out);

/* State management */
void ame_init(ame_state_t *ame);
void ame_reset(ame_state_t *ame);
void ame_set_ms_state(ame_state_t *ame, ame_ms_state_t ms);

/* Helper queries */
uint32_t ame_get_group_size(uint32_t dtype);
uint32_t ame_get_pack_factor(uint32_t dtype);
bool ame_is_dtype_supported(uint32_t dtype);
int ame_dtype_element_size_bytes(uint32_t dtype);

/* 5.3 Backend management */
ame_err_t ame_acquire(ame_state_t *ame, uint32_t *rd, uint32_t rs1_desc);
ame_err_t ame_release(ame_state_t *ame);

/* 6.119 & 6.8 Datatype configuration */
ame_err_t ame_msettyp(ame_state_t *ame, uint32_t md, uint32_t rs1_val);
ame_err_t ame_asettyp(ame_state_t *ame, uint32_t ad, uint32_t rs1_val);
ame_err_t ame_mgettyp(const ame_state_t *ame, uint32_t *rd, uint32_t md);
ame_err_t ame_agettyp(const ame_state_t *ame, uint32_t *rd, uint32_t ad);

/* 6.141 & 6.142 Zeroing */
ame_err_t ame_mzero_2d_acc(ame_state_t *ame, uint32_t acc);
ame_err_t ame_mzero_2d_m(ame_state_t *ame, uint32_t md);

/* 6.50 & 6.130 Row-Major Memory Transfer */
ame_err_t ame_mls_rm(ame_state_t *ame, uint32_t md, const void *base_addr);
ame_err_t ame_mss_rm(ame_state_t *ame, uint32_t ms, void *base_addr);

/* 6.72 Matrix Multiply-Accumulate */
ame_err_t ame_mmulacc_2d(ame_state_t *ame, uint32_t acc, uint32_t ms1, uint32_t ms2);

/* 6.59 & 6.60 Accumulator / Matrix Moves */
ame_err_t ame_mmov_m_a(ame_state_t *ame, uint32_t md, uint32_t acc);
ame_err_t ame_mmov_a_m(ame_state_t *ame, uint32_t acc, uint32_t ms);

#ifdef __cplusplus
}
#endif

#endif /* ZTT_AME_H */
