#include "ztt_ame.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/******************************************************************************
 * Helper Functions & Validation
 ******************************************************************************/

static inline ame_err_t check_active(const ame_state_t *ame) {
    if (!ame) return AME_ERR_ILLEGAL_INSN;
    if (ame->ms_status == AME_MS_OFF) {
        return AME_ERR_ILLEGAL_INSN;
    }
    if (!(ame->ameown & AME_OWN_OWNED)) {
        return AME_ERR_NOT_OWNED;
    }
    return AME_OK;
}

uint32_t ame_get_group_size(uint32_t dtype) {
    uint32_t sz_bits = AME_DTYPE_SIZE_BITS(dtype);
    if (sz_bits == 0) return 1;
    if (sz_bits <= AME_UNIT_DATATYPE_SIZE) return 1;
    return (sz_bits + AME_UNIT_DATATYPE_SIZE - 1) / AME_UNIT_DATATYPE_SIZE;
}

uint32_t ame_get_pack_factor(uint32_t dtype) {
    uint32_t sz_bits = AME_DTYPE_SIZE_BITS(dtype);
    if (sz_bits == 0) return 1;
    if (sz_bits < AME_UNIT_DATATYPE_SIZE) {
        return AME_UNIT_DATATYPE_SIZE / sz_bits;
    }
    return 1;
}

int ame_dtype_element_size_bytes(uint32_t dtype) {
    return (AME_DTYPE_SIZE_BITS(dtype) + 7) / 8;
}

bool ame_is_dtype_supported(uint32_t dtype) {
    switch (dtype) {
        case AME_DTYPE_INT8:
        case AME_DTYPE_UINT8:
        case AME_DTYPE_INT16:
        case AME_DTYPE_UINT16:
        case AME_DTYPE_INT32:
        case AME_DTYPE_UINT32:
        case AME_DTYPE_FP16:
        case AME_DTYPE_FP32:
            return true;
        default:
            return false;
    }
}

/******************************************************************************
 * Hardware Initialization & Control
 ******************************************************************************/

void ame_init(ame_state_t *ame) {
    if (!ame) return;
    memset(ame, 0, sizeof(*ame));
    ame->amenlen = AME_N;
    ame->ameudsz = AME_UNIT_DATATYPE_SIZE;
    ame->ms_status = AME_MS_INITIAL;
    ame->ameown = 0;
    ame->amestype = AME_DTYPE_UNINITIALIZED;
    ame->amefflags = 0;
    ame->amexsat = 0;
    ame->amestatus = 0;
}

void ame_reset(ame_state_t *ame) {
    if (!ame) return;
    memset(ame->M, 0, sizeof(ame->M));
    memset(ame->Acc, 0, sizeof(ame->Acc));
    memset(ame->Md, 0, sizeof(ame->Md));
    memset(ame->Ad, 0, sizeof(ame->Ad));
    ame->amefflags = 0;
    ame->amexsat = 0;
    ame->amestatus = 0;
}

void ame_set_ms_state(ame_state_t *ame, ame_ms_state_t ms) {
    if (!ame) return;
    if ((ame->ameown & AME_OWN_OWNED) && ms == AME_MS_OFF) {
        /* Section 5.2: Cannot turn Off while owned */
        return;
    }
    ame->ms_status = ms;
}

/******************************************************************************
 * 5.3 Backend Management Instructions
 ******************************************************************************/

ame_err_t ame_acquire(ame_state_t *ame, uint32_t *rd, uint32_t rs1_desc) {
    if (!ame) return AME_ERR_ILLEGAL_INSN;
    if (ame->ms_status == AME_MS_OFF) {
        return AME_ERR_ILLEGAL_INSN;
    }

    /* Validate reserved descriptor bits [31:6] */
    if ((rs1_desc >> 6) != 0) {
        if (rd) *rd = AME_ACQ_STATUS_BAD_DESC;
        return AME_ERR_ILLEGAL_INSN;
    }

    /* Assign ownership */
    ame->ameown |= AME_OWN_OWNED;
    ame->ms_status = AME_MS_DIRTY;

    if (rd) {
        *rd = AME_ACQ_SUCCESS | AME_ACQ_STATUS_GRANTED;
    }
    return AME_OK;
}

ame_err_t ame_release(ame_state_t *ame) {
    if (!ame) return AME_ERR_ILLEGAL_INSN;
    if (ame->ms_status == AME_MS_OFF) {
        return AME_ERR_ILLEGAL_INSN;
    }
    ame->ameown &= ~AME_OWN_OWNED;
    return AME_OK;
}

/******************************************************************************
 * 6.119 & 6.8 Datatype Configuration Instructions
 ******************************************************************************/

ame_err_t ame_msettyp(ame_state_t *ame, uint32_t md, uint32_t rs1_val) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (md >= AME_NUM_M_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t group_size = ame_get_group_size(rs1_val);
    /* Group alignment check */
    if ((md % group_size != 0) || (md + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    ame->Md[md] = rs1_val;

    /* Raw-bit clear for entire group (Section 6.119) */
    for (uint32_t g = 0; g < group_size; ++g) {
        memset(ame->M[md + g], 0, AME_M_REG_SIZE_BYTES);
    }

    return AME_OK;
}

ame_err_t ame_asettyp(ame_state_t *ame, uint32_t ad, uint32_t rs1_val) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (ad >= AME_NUM_ACC_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    ame->Ad[ad] = rs1_val;

    /* Raw-bit clear for accumulator (Section 6.8) */
    memset(ame->Acc[ad], 0, AME_ACC_REG_SIZE_BYTES);

    return AME_OK;
}

ame_err_t ame_mgettyp(const ame_state_t *ame, uint32_t *rd, uint32_t md) {
    if (!ame || !rd) return AME_ERR_ILLEGAL_INSN;
    if (ame->ms_status == AME_MS_OFF) return AME_ERR_ILLEGAL_INSN;
    if (md >= AME_NUM_M_REGS) return AME_ERR_ILLEGAL_INSN;

    *rd = ame->Md[md];
    return AME_OK;
}

ame_err_t ame_agettyp(const ame_state_t *ame, uint32_t *rd, uint32_t ad) {
    if (!ame || !rd) return AME_ERR_ILLEGAL_INSN;
    if (ame->ms_status == AME_MS_OFF) return AME_ERR_ILLEGAL_INSN;
    if (ad >= AME_NUM_ACC_REGS) return AME_ERR_ILLEGAL_INSN;

    *rd = ame->Ad[ad];
    return AME_OK;
}

/******************************************************************************
 * 6.141 & 6.142 Zeroing Instructions
 ******************************************************************************/

ame_err_t ame_mzero_2d_acc(ame_state_t *ame, uint32_t acc) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (acc >= AME_NUM_ACC_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    /* Semantic zero for supported types (all binary zeros for standard int/float) */
    memset(ame->Acc[acc], 0, AME_ACC_REG_SIZE_BYTES);
    return AME_OK;
}

ame_err_t ame_mzero_2d_m(ame_state_t *ame, uint32_t md) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (md >= AME_NUM_M_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t group_size = ame_get_group_size(ame->Md[md]);
    if ((md % group_size != 0) || (md + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    for (uint32_t g = 0; g < group_size; ++g) {
        memset(ame->M[md + g], 0, AME_M_REG_SIZE_BYTES);
    }
    return AME_OK;
}

/******************************************************************************
 * 6.50 & 6.130 Row-Major Memory Transfer Instructions
 ******************************************************************************/

ame_err_t ame_mls_rm(ame_state_t *ame, uint32_t md, const void *base_addr) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (md >= AME_NUM_M_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t dtype = ame->Md[md];
    uint32_t group_size = ame_get_group_size(dtype);
    if ((md % group_size != 0) || (md + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    int elem_bytes = ame_dtype_element_size_bytes(dtype);
    if (elem_bytes == 0) elem_bytes = 1;

    /* Base address natural alignment check (Section 2.6) */
    if (((uintptr_t)base_addr % (uintptr_t)elem_bytes) != 0) {
        return AME_ERR_MISALIGNED;
    }

    size_t total_bytes = (size_t)AME_NELEM * elem_bytes;
    memcpy(ame->M[md], base_addr, total_bytes);
    return AME_OK;
}

ame_err_t ame_mss_rm(const ame_state_t *ame, uint32_t ms, void *base_addr) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (ms >= AME_NUM_M_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t dtype = ame->Md[ms];
    uint32_t group_size = ame_get_group_size(dtype);
    if ((ms % group_size != 0) || (ms + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    int elem_bytes = ame_dtype_element_size_bytes(dtype);
    if (elem_bytes == 0) elem_bytes = 1;

    /* Base address natural alignment check */
    if (((uintptr_t)base_addr % (uintptr_t)elem_bytes) != 0) {
        return AME_ERR_MISALIGNED;
    }

    size_t total_bytes = (size_t)AME_NELEM * elem_bytes;
    memcpy(base_addr, ame->M[ms], total_bytes);
    return AME_OK;
}

/******************************************************************************
 * 6.59 & 6.60 Accumulator / Matrix Moves
 ******************************************************************************/

ame_err_t ame_mmov_m_a(ame_state_t *ame, uint32_t md, uint32_t acc) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (md >= AME_NUM_M_REGS || acc >= AME_NUM_ACC_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t d_m = ame->Md[md];
    uint32_t d_acc = ame->Ad[acc];

    /* Section 6.59: Datatype must match strictly */
    if (d_m != d_acc) {
        ame->amestatus |= AME_STATUS_UN;
        return AME_ERR_UNSUPPORTED;
    }

    uint32_t group_size = ame_get_group_size(d_m);
    if ((md % group_size != 0) || (md + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    int elem_bytes = ame_dtype_element_size_bytes(d_m);
    size_t total_bytes = (size_t)AME_NELEM * elem_bytes;

    memcpy(ame->M[md], ame->Acc[acc], total_bytes);
    return AME_OK;
}

ame_err_t ame_mmov_a_m(ame_state_t *ame, uint32_t acc, uint32_t ms) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (ms >= AME_NUM_M_REGS || acc >= AME_NUM_ACC_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t d_m = ame->Md[ms];
    uint32_t d_acc = ame->Ad[acc];

    if (d_m != d_acc) {
        ame->amestatus |= AME_STATUS_UN;
        return AME_ERR_UNSUPPORTED;
    }

    uint32_t group_size = ame_get_group_size(d_m);
    if ((ms % group_size != 0) || (ms + group_size > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    int elem_bytes = ame_dtype_element_size_bytes(d_m);
    size_t total_bytes = (size_t)AME_NELEM * elem_bytes;

    memcpy(ame->Acc[acc], ame->M[ms], total_bytes);
    return AME_OK;
}

/******************************************************************************
 * 6.72 Matrix Multiply-Accumulate: C = C + A * B
 ******************************************************************************/

ame_err_t ame_mmulacc_2d(ame_state_t *ame, uint32_t acc, uint32_t ms1, uint32_t ms2) {
    ame_err_t err = check_active(ame);
    if (err != AME_OK) return err;

    if (acc >= AME_NUM_ACC_REGS || ms1 >= AME_NUM_M_REGS || ms2 >= AME_NUM_M_REGS) {
        return AME_ERR_ILLEGAL_INSN;
    }

    uint32_t d_acc = ame->Ad[acc];
    uint32_t d_a = ame->Md[ms1];
    uint32_t d_b = ame->Md[ms2];

    uint32_t gs_a = ame_get_group_size(d_a);
    uint32_t gs_b = ame_get_group_size(d_b);
    if ((ms1 % gs_a != 0) || (ms1 + gs_a > AME_NUM_M_REGS) ||
        (ms2 % gs_b != 0) || (ms2 + gs_b > AME_NUM_M_REGS)) {
        return AME_ERR_ILLEGAL_INSN;
    }

    const int N = AME_N;

    /* Tuple FP32 x FP32 -> FP32 */
    if (d_acc == AME_DTYPE_FP32 && d_a == AME_DTYPE_FP32 && d_b == AME_DTYPE_FP32) {
        const float *A = (const float *)ame->M[ms1];
        const float *B = (const float *)ame->M[ms2];
        float *C = (float *)ame->Acc[acc];

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] += sum;
            }
        }
        return AME_OK;
    }

    /* Tuple Int32 x Int32 -> Int32 (Signed) */
    if (d_acc == AME_DTYPE_INT32 && d_a == AME_DTYPE_INT32 && d_b == AME_DTYPE_INT32) {
        const int32_t *A = (const int32_t *)ame->M[ms1];
        const int32_t *B = (const int32_t *)ame->M[ms2];
        int32_t *C = (int32_t *)ame->Acc[acc];

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int64_t sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += (int64_t)A[i * N + k] * (int64_t)B[k * N + j];
                }
                C[i * N + j] += (int32_t)sum;
            }
        }
        return AME_OK;
    }

    /* Tuple Uint32 x Uint32 -> Uint32 */
    if (d_acc == AME_DTYPE_UINT32 && d_a == AME_DTYPE_UINT32 && d_b == AME_DTYPE_UINT32) {
        const uint32_t *A = (const uint32_t *)ame->M[ms1];
        const uint32_t *B = (const uint32_t *)ame->M[ms2];
        uint32_t *C = (uint32_t *)ame->Acc[acc];

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                uint64_t sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += (uint64_t)A[i * N + k] * (uint64_t)B[k * N + j];
                }
                C[i * N + j] += (uint32_t)sum;
            }
        }
        return AME_OK;
    }

    /* Unsupported tuple for 2D matrix multiply */
    ame->amestatus |= AME_STATUS_UN;
    return AME_ERR_UNSUPPORTED;
}
