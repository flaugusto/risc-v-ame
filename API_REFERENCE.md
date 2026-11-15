# RISC-V Attached Matrix Extension (AME) Simulator API Reference

## 1. Overview

This document provides a technical reference for the base C instruction emulation library of the **RISC-V Attached Matrix Extension (AME)** simulator.

**Specification Note**: This simulator is based on the RISC-V Attached Matrix Extension (Ztt) specification version 0.6. For the official specification document and release details, refer to the [GitHub Release v0.6](https://github.com/riscv/riscv-attached-matrix-extension/releases/tag/v0.6).

This simulator models the register banks, control/status registers (CSRs), datatype descriptors, memory transfers, and compute execution of the official AME specification. It is intended for kernel developers who want to write and evaluate custom matrix computation kernels (such as GEMM, convolutions, activations, or tensor contractions) using simulated AME hardware instructions before deploying to physical silicon or full-system emulators.

---

## 2. Compilation, Import, and Linking

The base simulator consists of two files:

- `ztt_ame.h`: Public header containing architectural constants, datatype descriptors, state definitions, register aliases, and instruction function prototypes.
- `ztt_ame.c`: C99 implementation of the AME state machine and instruction emulation.

### 2.1 Importing in C/C++ Code

To use the AME base instructions in your kernel source file, include `ztt_ame.h`:

```c
#include "ztt_ame.h"
```

The header contains `extern "C"` guards, making it fully compatible with C++ kernels.

### 2.2 Compilation

#### Method A: Direct Source Compilation

Compile your kernel source file alongside `ztt_ame.c`:

```bash
gcc -O3 -std=c99 my_kernel.c ztt_ame/ztt_ame.c -I./ztt_ame -lm -o my_kernel
```

#### Method B: Linking Against `libztt.a` Static Library

1. Build the static library inside `ztt_ame/`:
   ```bash
   cd ztt_ame && make libztt.a
   ```
2. Compile and link your kernel against `libztt.a`:
   ```bash
   gcc -O3 -std=c99 my_kernel.c -I./ztt_ame -L./ztt_ame -lztt -lm -o my_kernel
   ```

### 2.3 Compile-Time Hardware Parameter Overrides

The simulator hardware configuration can be customized at compile time via preprocessor definitions:

```bash
gcc -O3 -std=c99 -DAME_N=8 -DAME_NUM_M_REGS=32 -DAME_NUM_ACC_REGS=16 \
    my_kernel.c ztt_ame/ztt_ame.c -I./ztt_ame -lm -o my_kernel
```

| Preprocessor Macro       | Default | Valid Options                              | Description                                                                                  |
| :----------------------- | :------ | :----------------------------------------- | :------------------------------------------------------------------------------------------- |
| `AME_N`                  | `8`     | `4`, `8`, `16`, `32`, `64`, `128`          | Side dimension of square logical tile (power of 2).                                          |
| `AME_NELEM`              | `64`    | `16`, `64`, `256`, `1024`, `4096`, `16384` | Total element count per logical tile ($N \times N$, power of 4: $N^2$).                      |
| `AME_NUM_M_REGS`         | `32`    | `16`, `32`                                 | Total number of general matrix registers ($M_0 \dots M_{31}$).                               |
| `AME_NUM_ACC_REGS`       | `16`    | `1`, `2`, `4`, `8`, `16`                   | Total number of accumulator registers ($\text{Acc}_0 \dots \text{Acc}_{15}$).                |
| `AME_UNIT_DATATYPE_SIZE` | `32`    | `32` (power of 2)                          | Datatype size in bits where exactly one logical tile fits in one $M$ register.               |

---

## 3. Hardware Architecture & Concepts

### 3.1 Logical Squares ($N \times N$ Tiles)

The AME operates on 2D square matrix tiles called **logical squares**. For the default `AME_N = 8` (`AME_NELEM = 64`), each tile is $8 \times 8$ elements ($N = 8$).

### 3.2 Register Files

The hardware provides two separate register banks:

1. **Matrix Registers ($M_0 \dots M_{31}$)**:
   - Hold input matrices for arithmetic operations and destination tiles for elementwise operations.
   - Sized to `AME_M_REG_SIZE_BYTES` = `(AME_NELEM * AME_UNIT_DATATYPE_SIZE) / 8` bytes.
   - Register aliases `m0` through `m31` are provided for code readability.
2. **Accumulator Registers ($\text{Acc}_0 \dots \text{Acc}_{15}$)**:
   - Dedicated destinations for matrix multiply-accumulate (`ame_mmulacc_2d`) operations.
   - Sized to `AME_ACC_REG_SIZE_BYTES` = `(AME_NELEM * AME_MAX_DATATYPE_SIZE) / 8` bytes (supports up to 64-bit wide elements).
   - Register aliases `acc0` through `acc15` are provided.

### 3.3 Register Grouping Rule

When using datatypes wider than `AME_UNIT_DATATYPE_SIZE` (32 bits), the register grouping factor (`group_size`) is defined as:

$$
\text{group}_{\text{size}} = \max\left(1, \left\lceil \frac{E}{32} \right\rceil\right)
$$

- **Alignment Invariant**: The base matrix register index $md$ must be a multiple of `group_size` (`md % group_size == 0`). Violating this triggers `AME_ERR_ILLEGAL_INSN`.

### 3.4 Decoupled Datatype Descriptors ($Md$ and $Ad$)

In the AME architecture, instructions do not encode datatypes in opcode bits. Instead, every matrix register $M_i$ and accumulator register $\text{Acc}_j$ has an associated 32-bit metadata descriptor register ($Md_i$ and $Ad_j$) configured prior to computation using `ame_msettyp` or `ame_asettyp`.

---

## 4. Datatype Descriptors

A datatype descriptor is a 32-bit bitfield defined by Section 2.4 of the specification:

- **Bit 31**: `0` = Standard RISC-V type, `1` = Custom vendor type.
- **Bits 30:8**: Properties (signedness, saturation, rounding mode, exponent width).
- **Bits 7:0**: Element width in bits ($E$).

### 4.1 Predefined Datatype Constants

| Constant                  | Description                                           | Size (Bits) |
| :------------------------ | :---------------------------------------------------- | :---------- |
| `AME_DTYPE_INT8`          | Signed 8-bit integer (RNU rounding, non-saturating)   | 8           |
| `AME_DTYPE_UINT8`         | Unsigned 8-bit integer (RNU rounding, non-saturating) | 8           |
| `AME_DTYPE_INT16`         | Signed 16-bit integer                                 | 16          |
| `AME_DTYPE_UINT16`        | Unsigned 16-bit integer                               | 16          |
| `AME_DTYPE_INT32`         | Signed 32-bit integer                                 | 32          |
| `AME_DTYPE_UINT32`        | Unsigned 32-bit integer                               | 32          |
| `AME_DTYPE_INT64`         | Signed 64-bit integer                                 | 64          |
| `AME_DTYPE_UINT64`        | Unsigned 64-bit integer                               | 64          |
| `AME_DTYPE_FP16`          | IEEE 754 half-precision float (5 exponent bits)       | 16          |
| `AME_DTYPE_BF16`          | Brain floating point (8 exponent bits)                | 16          |
| `AME_DTYPE_FP32`          | IEEE 754 single-precision float (8 exponent bits)     | 32          |
| `AME_DTYPE_FP64`          | IEEE 754 double-precision float (11 exponent bits)    | 64          |
| `AME_DTYPE_UNINITIALIZED` | Uninitialized state (`0x00000000`)                    | 0           |

### 4.2 Inspection Macros

```c
AME_DTYPE_IS_CUSTOM(d)       /* Returns 1 if descriptor specifies a custom vendor type */
AME_DTYPE_IS_FP(d)           /* Returns 1 if descriptor specifies a standard floating-point type */
AME_DTYPE_IS_INT(d)          /* Returns 1 if descriptor specifies a standard integer type */
AME_DTYPE_SIZE_BITS(d)       /* Returns element size in bits */
AME_DTYPE_SIZE_BYTES(d)      /* Returns element size in bytes: (bits + 7) / 8 */
AME_DTYPE_INT_SIGNED(d)      /* Returns 1 if signed integer, 0 if unsigned */
AME_DTYPE_INT_SAT(d)         /* Returns 1 if saturation is enabled */
```

---

## 5. Return Codes and Status CSRs

All instruction emulation functions return an error status of type `ame_err_t`:

| Return Code            | Value | Description                                                                                                           |
| :--------------------- | :---- | :-------------------------------------------------------------------------------------------------------------------- |
| `AME_OK`               | `0`   | Instruction completed successfully.                                                                                   |
| `AME_ERR_ILLEGAL_INSN` | `-1`  | Invalid instruction format, out-of-range register index, violated group alignment, or execution when hardware is OFF. |
| `AME_ERR_UNSUPPORTED`  | `-2`  | Unsupported datatype combination (sets `amestatus.UN` sticky flag).                                                   |
| `AME_ERR_MISALIGNED`   | `-3`  | Memory base address is not aligned to the natural boundary of the element size.                                       |
| `AME_ERR_NOT_OWNED`    | `-4`  | Attempted execution when hart does not own the backend (`ameown.OWNED == 0`).                                         |
| `AME_ERR_INVALID_REG`  | `-5`  | Out-of-bounds register reference.                                                                                     |

---

## 6. Base Function Reference

### 6.1 State Lifecycle & Management

#### `ame_init`

```c
void ame_init(ame_state_t *ame);
```

- **Description**: Initializes an `ame_state_t` structure to architectural defaults. Sets `amenlen = AME_N`, `ameudsz = AME_UNIT_DATATYPE_SIZE`, clears all register banks, resets all CSRs to `0`, and sets `ms_status = AME_MS_INITIAL`.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance. Must not be `NULL`.
- **Return / Behavior**: Does not return a value. Must be invoked once before any other AME function.

#### `ame_reset`

```c
void ame_reset(ame_state_t *ame);
```

- **Description**: Resets registers and exception flags without altering backend ownership, architectural dimension parameters, or performance counters.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
- **Return / Behavior**: Clears `M`, `Acc`, `Md`, `Ad`, `amefflags`, `amexsat`, and `amestatus`.

#### `ame_set_ms_state`

```c
void ame_set_ms_state(ame_state_t *ame, ame_ms_state_t ms);
```

- **Description**: Updates the `mstatus.MS` (Matrix Status) architectural field (`AME_MS_OFF`, `AME_MS_INITIAL`, `AME_MS_CLEAN`, `AME_MS_DIRTY`).
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `ms`: Target state enum.
- **Return / Behavior**: If the hart currently owns the backend (`ameown & AME_OWN_OWNED`), setting `AME_MS_OFF` is prohibited and silently ignored according to Section 5.2.

---

### 6.2 Backend Ownership Instructions (Section 5.3)

#### `ame_acquire`

```c
ame_err_t ame_acquire(ame_state_t *ame, uint32_t *rd, uint32_t rs1_desc);
```

- **Description**: Emulates the `ame.acquire rd, rs1` instruction. Requests ownership of the matrix accelerator backend for the calling hart.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `rd`: Pointer to a 32-bit unsigned integer receiving the acquire status result. May be `NULL`.
  - `rs1_desc`: 32-bit backend descriptor. Bits `[31:6]` are reserved and must be `0`.
- **Return / Behavior**:
  - Returns `AME_OK` on success, sets `ameown |= AME_OWN_OWNED`, sets `ms_status = AME_MS_DIRTY`, and outputs `AME_ACQ_SUCCESS | AME_ACQ_STATUS_GRANTED` (`0x01`) into `*rd`.
  - Returns `AME_ERR_ILLEGAL_INSN` if `ms_status == AME_MS_OFF` or if reserved bits `[31:6]` are non-zero (writing `AME_ACQ_STATUS_BAD_DESC` to `*rd`).

#### `ame_release`

```c
ame_err_t ame_release(ame_state_t *ame);
```

- **Description**: Emulates the `ame.release` instruction. Releases backend ownership.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
- **Return / Behavior**: Clears bit 0 of `ameown` (`ameown &= ~AME_OWN_OWNED`). Subsequent instruction calls (except `ame_acquire`) will return `AME_ERR_NOT_OWNED` until `ame_acquire` is called again.

---

### 6.3 Datatype Configuration Instructions (Sections 6.119 & 6.8)

#### `ame_msettyp`

```c
ame_err_t ame_msettyp(ame_state_t *ame, uint32_t md, uint32_t rs1_val);
```

- **Description**: Emulates `msettyp md, rs1`. Configures the 32-bit datatype descriptor for matrix register `md` and its group.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `md`: Index of matrix register (`0` to `AME_NUM_M_REGS - 1`; aliases `m0`..`m31`).
  - `rs1_val`: 32-bit datatype descriptor value (e.g., `AME_DTYPE_FP32`).
- **Return / Behavior**:
  - Checks that `md` satisfies group alignment (`md % group_size == 0`) and does not exceed available registers.
  - Writes `rs1_val` to `Md[md]`.
  - Performs raw-bit clear (zeros all bytes) across all physical registers in the group `M[md .. md + group_size - 1]`.
  - Returns `AME_OK` on success; returns `AME_ERR_ILLEGAL_INSN` on out-of-range register or misaligned base index.

#### `ame_asettyp`

```c
ame_err_t ame_asettyp(ame_state_t *ame, uint32_t ad, uint32_t rs1_val);
```

- **Description**: Emulates `asettyp ad, rs1`. Configures the 32-bit datatype descriptor for accumulator register `ad`.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `ad`: Index of accumulator register (`0` to `AME_NUM_ACC_REGS - 1`; aliases `acc0`..`acc15`).
  - `rs1_val`: 32-bit datatype descriptor value.
- **Return / Behavior**:
  - Writes `rs1_val` to `Ad[ad]`.
  - Performs raw-bit clear (zeros all bytes) across `Acc[ad]`.
  - Returns `AME_OK` on success; returns `AME_ERR_ILLEGAL_INSN` if `ad >= AME_NUM_ACC_REGS`.

#### `ame_mgettyp` / `ame_agettyp`

```c
ame_err_t ame_mgettyp(const ame_state_t *ame, uint32_t *rd, uint32_t md);
ame_err_t ame_agettyp(const ame_state_t *ame, uint32_t *rd, uint32_t ad);
```

- **Description**: Emulate `mgettyp rd, md` and `agettyp rd, ad`. Query the current datatype descriptor assigned to matrix register `md` or accumulator `ad`.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `rd`: Destination pointer receiving the descriptor. Must not be `NULL`.
  - `md` / `ad`: Register index.
- **Return / Behavior**: Copies `Md[md]` or `Ad[ad]` into `*rd`. Returns `AME_OK`, or `AME_ERR_ILLEGAL_INSN` on invalid pointer or out-of-bounds register.

---

### 6.4 Register Zeroing Instructions (Sections 6.141 & 6.142)

#### `ame_mzero_2d_acc`

```c
ame_err_t ame_mzero_2d_acc(ame_state_t *ame, uint32_t acc);
```

- **Description**: Emulates `mzero.2d acc`. Resets the contents of accumulator register `acc` to semantic zero without modifying its datatype descriptor in `Ad[acc]`.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `acc`: Accumulator index (`acc0`..`acc15`).
- **Return / Behavior**: Fills `Acc[acc]` with zeros. Returns `AME_OK`, or `AME_ERR_ILLEGAL_INSN` if out-of-bounds.

#### `ame_mzero_2d_m`

```c
ame_err_t ame_mzero_2d_m(ame_state_t *ame, uint32_t md);
```

- **Description**: Emulates `mzero.2d md`. Resets matrix register `md` and all registers in its group to semantic zero.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `md`: Matrix register index (`m0`..`m31`).
- **Return / Behavior**: Zeros bytes in `M[md .. md + group_size - 1]`. Returns `AME_OK`, or `AME_ERR_ILLEGAL_INSN` on alignment error or out-of-range index.

---

### 6.5 Row-Major Memory Transfer Instructions (Sections 6.50 & 6.130)

#### `ame_mls_rm`

```c
ame_err_t ame_mls_rm(ame_state_t *ame, uint32_t md, const void *base_addr);
```

- **Description**: Emulates `mls.rm md, (rs1)`. Loads an $N \times N$ contiguous row-major tile from memory into matrix register `md`.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `md`: Destination matrix register index (`m0`..`m31`).
  - `base_addr`: Pointer to contiguous memory buffer holding $N \times N$ elements.
- **Return / Behavior**:
  - Calculates total transfer size: `AME_NELEM * element_size_bytes` bytes.
  - Verifies natural alignment: `(uintptr_t)base_addr % element_size_bytes == 0`. Returns `AME_ERR_MISALIGNED` if misaligned.
  - Copies data into `M[md]`.
  - Increments `stats.mls_rm` by 1 and `stats.bytes_loaded` by the byte count.
  - Returns `AME_OK`.

#### `ame_mss_rm`

```c
ame_err_t ame_mss_rm(ame_state_t *ame, uint32_t ms, void *base_addr);
```

- **Description**: Emulates `mss.rm ms, (rs1)`. Stores an $N \times N$ contiguous row-major tile from matrix register `ms` into memory.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `ms`: Source matrix register index (`m0`..`m31`).
  - `base_addr`: Destination memory buffer pointer.
- **Return / Behavior**:
  - Verifies natural alignment of `base_addr`.
  - Copies data from `M[ms]` to memory.
  - Increments `stats.mss_rm` by 1 and `stats.bytes_stored` by the byte count.
  - Returns `AME_OK`.

---

### 6.6 Strided Memory Transfer Instructions (Sections 6.51 & 6.131)

Strided instructions allow kernels to load/store an $N \times N$ tile directly to/from standard full-sized row-major matrices without allocating intermediate packed tile buffers.

#### `ame_mls_st`

```c
ame_err_t ame_mls_st(ame_state_t *ame, uint32_t md, const void *base_addr, size_t stride_bytes);
```

- **Description**: Emulates `mls.st md, (rs1), rs2`. Loads an $N \times N$ tile from a 2D matrix in memory with a given row stride.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `md`: Destination matrix register index (`m0`..`m31`).
  - `base_addr`: Memory address of the top-left element of the tile.
  - `stride_bytes`: Byte distance between consecutive rows in the source matrix (typically `leading_dimension * element_size_bytes`).
- **Return / Behavior**:
  - For each row $r \in [0, N-1]$, reads `N * element_size_bytes` bytes from:
    $$\text{base} + r \times \text{stride}$$
    and writes consecutively into row $r$ of matrix register `M[md]`.
  - Checks natural alignment of `base_addr`.
  - Increments `stats.mls_st` by 1 and `stats.bytes_loaded` by `AME_NELEM * element_size_bytes` bytes.
  - Returns `AME_OK`.

#### `ame_mss_st`

```c
ame_err_t ame_mss_st(ame_state_t *ame, uint32_t ms, void *base_addr, size_t stride_bytes);
```

- **Description**: Emulates `mss.st ms, (rs1), rs2`. Stores an $N \times N$ tile from matrix register `ms` into a 2D matrix in memory with a given row stride.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `ms`: Source matrix register index (`m0`..`m31`).
  - `base_addr`: Memory address of the top-left destination element.
  - `stride_bytes`: Byte distance between consecutive rows in destination memory.
- **Return / Behavior**:
  - For each row $r \in [0, N-1]$, writes row $r$ of `M[ms]` (`N * element_size_bytes` bytes) to:
    $$\text{base} + r \times \text{stride}$$
  - Checks natural alignment of `base_addr`.
  - Increments `stats.mss_st` by 1 and `stats.bytes_stored` by `AME_NELEM * element_size_bytes` bytes.
  - Returns `AME_OK`.

---

### 6.7 Matrix Multiply-Accumulate (Section 6.72)

#### `ame_mmulacc_2d`

```c
ame_err_t ame_mmulacc_2d(ame_state_t *ame, uint32_t acc, uint32_t ms1, uint32_t ms2);
```

- **Description**: Emulates `mmulacc.2d acc, ms1, ms2`. Computes the 2D matrix multiplication of tiles in `ms1` and `ms2`, adding the product into accumulator `acc`:
  $$\text{Acc}[\text{acc}] \leftarrow \text{Acc}[\text{acc}] + (M[\text{ms1}] \times M[\text{ms2}])$$
  where $M[\text{ms1}], M[\text{ms2}], \text{Acc}[\text{acc}]$ are all $N \times N$ square matrices.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `acc`: Destination and accumulation register index (`acc0`..`acc15`).
  - `ms1`: Left operand matrix register index (`m0`..`m31`).
  - `ms2`: Right operand matrix register index (`m0`..`m31`).
- **Supported Datatype Tuples**:
  - **FP32**: `Ad[acc] == AME_DTYPE_FP32`, `Md[ms1] == AME_DTYPE_FP32`, `Md[ms2] == AME_DTYPE_FP32`
  - **Int32 (Signed)**: `Ad[acc] == AME_DTYPE_INT32`, `Md[ms1] == AME_DTYPE_INT32`, `Md[ms2] == AME_DTYPE_INT32`
  - **UInt32 (Unsigned)**: `Ad[acc] == AME_DTYPE_UINT32`, `Md[ms1] == AME_DTYPE_UINT32`, `Md[ms2] == AME_DTYPE_UINT32`
- **Return / Behavior**:
  - Computes $\text{Acc}_{i, j} += \sum_{k=0}^{N-1} M1_{i, k} \times M2_{k, j}$.
  - Increments `stats.mmulacc_2d` by 1.
  - Increments `stats.macs` by $N^3$ ($N \times N \times N$ multiply-accumulate operations).
  - If the datatype tuple is unsupported, sets `ame->amestatus |= AME_STATUS_UN` and returns `AME_ERR_UNSUPPORTED`.
  - Returns `AME_OK` on success.

---

### 6.8 Register Transfer Instructions (Sections 6.59 & 6.60)

Accumulator registers cannot write directly to memory. To store computed results, transfer the tile to an $M$ register using `ame_mmov_m_a`, then store using `ame_mss_st` or `ame_mss_rm`.

#### `ame_mmov_m_a`

```c
ame_err_t ame_mmov_m_a(ame_state_t *ame, uint32_t md, uint32_t acc);
```

- **Description**: Emulates `mmov.m.a md, acc`. Copies an $N \times N$ tile from accumulator register `acc` to matrix register `md` (and its group).
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `md`: Destination matrix register index (`m0`..`m31`).
  - `acc`: Source accumulator register index (`acc0`..`acc15`).
- **Return / Behavior**:
  - Checks strict datatype equality: `Md[md] == Ad[acc]`. If types differ, sets `ame->amestatus |= AME_STATUS_UN` and returns `AME_ERR_UNSUPPORTED`.
  - Verifies group alignment of `md`.
  - Copies `AME_NELEM * element_size_bytes` bytes from `Acc[acc]` to `M[md]`.
  - Increments `stats.mmov_m_a` by 1.
  - Returns `AME_OK`.

#### `ame_mmov_a_m`

```c
ame_err_t ame_mmov_a_m(ame_state_t *ame, uint32_t acc, uint32_t ms);
```

- **Description**: Emulates `mmov.a.m acc, ms`. Copies an $N \times N$ tile from matrix register `ms` to accumulator register `acc`. Useful for initializing an accumulator with pre-existing $C$ matrix data before accumulation.
- **Parameters**:
  - `ame`: Pointer to the simulator state instance.
  - `acc`: Destination accumulator register index (`acc0`..`acc15`).
  - `ms`: Source matrix register index (`m0`..`m31`).
- **Return / Behavior**:
  - Checks strict datatype equality: `Ad[acc] == Md[ms]`. If types differ, sets `ame->amestatus |= AME_STATUS_UN` and returns `AME_ERR_UNSUPPORTED`.
  - Copies tile bytes from `M[ms]` to `Acc[acc]`.
  - Increments `stats.mmov_a_m` by 1.
  - Returns `AME_OK`.

---

### 6.9 Helper Query Functions

```c
uint32_t ame_get_group_size(uint32_t dtype);
```

Returns the register group size for the specified datatype descriptor ($\ge 1$).

```c
uint32_t ame_get_pack_factor(uint32_t dtype);
```

Returns the packing factor if element size is smaller than `AME_UNIT_DATATYPE_SIZE`.

```c
bool ame_is_dtype_supported(uint32_t dtype);
```

Returns `true` if `dtype` is supported by the simulator.

```c
int ame_dtype_element_size_bytes(uint32_t dtype);
```

Returns the size of one element in bytes (e.g., 4 for `AME_DTYPE_FP32`).

---

### 6.10 Profiling and Performance Statistics

**Note**: `ame->stats` is **not part of the official RISC-V AME specification**. It is a simulator-specific facility provided solely for performance monitoring, memory traffic analysis, and kernel profiling.

The simulator tracks instruction counts, data traffic, and arithmetic operations in `ame->stats`:

```c
typedef struct {
    uint64_t mls_rm;       /* Count of mls.rm instructions executed */
    uint64_t mss_rm;       /* Count of mss.rm instructions executed */
    uint64_t mls_st;       /* Count of mls.st instructions executed */
    uint64_t mss_st;       /* Count of mss.st instructions executed */
    uint64_t mmulacc_2d;   /* Count of mmulacc.2d instructions executed */
    uint64_t mmov_m_a;     /* Count of mmov.m.a instructions executed */
    uint64_t mmov_a_m;     /* Count of mmov.a.m instructions executed */
    uint64_t bytes_loaded; /* Total bytes read from memory into M registers */
    uint64_t bytes_stored; /* Total bytes written from M registers into memory */
    uint64_t macs;         /* Total MAC operations (mmulacc_2d * N^3) */
} ame_stats_t;
```

#### `ame_stats_reset`

```c
void ame_stats_reset(ame_state_t *ame);
```

Zeros out all counters in `ame->stats`.

#### `ame_stats_report`

```c
void ame_stats_report(const ame_state_t *ame, FILE *out);
```

Prints a formatted summary of instruction counts, total bytes loaded/stored, total MACs, arithmetic intensity (MACs/byte), and memory tile reuse to stream `out` (e.g. `stdout`).

---

## 7. Complete Example: Writing a Kernel with Base Instructions

Below is a self-contained, complete C example demonstrating how a kernel author imports `ztt_ame.h`, manages backend ownership, sets datatypes, loads strided tiles from host memory, executes 2D multiply-accumulate operations, and retrieves results.

```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "ztt_ame.h"

/* Compute C[N x N] += A[N x N] * B[N x N] using AME base instructions */
void single_tile_gemm_example(void) {
    const int N = AME_N; /* 8 by default */
    ame_state_t ame;

    /* 1. Initialize simulator state */
    ame_init(&ame);

    /* 2. Acquire backend ownership */
    uint32_t acq_result = 0;
    ame_err_t err = ame_acquire(&ame, &acq_result, 0);
    assert(err == AME_OK && (acq_result & AME_ACQ_SUCCESS));

    /* 3. Configure datatype descriptors for registers */
    /* m0: operand A tile, m1: operand B tile */
    ame_msettyp(&ame, m0, AME_DTYPE_FP32);
    ame_msettyp(&ame, m1, AME_DTYPE_FP32);
    /* acc0: accumulator tile */
    ame_asettyp(&ame, acc0, AME_DTYPE_FP32);

    /* 4. Prepare host matrix data */
    float A[N * N];
    float B[N * N];
    float C[N * N];

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    /* 5. Load memory tiles into matrix registers */
    /* For contiguous NxN arrays, row stride in bytes is N * sizeof(float) */
    size_t stride_bytes = (size_t)N * sizeof(float);
    err = ame_mls_st(&ame, m0, A, stride_bytes);
    assert(err == AME_OK);

    err = ame_mls_st(&ame, m1, B, stride_bytes);
    assert(err == AME_OK);

    /* 6. Perform matrix multiply-accumulate: acc0 += m0 x m1 */
    err = ame_mmulacc_2d(&ame, acc0, m0, m1);
    assert(err == AME_OK);

    /* 7. Move result from accumulator to matrix register m0 */
    err = ame_mmov_m_a(&ame, m0, acc0);
    assert(err == AME_OK);

    /* 8. Store result tile back to host memory C */
    err = ame_mss_st(&ame, m0, C, stride_bytes);
    assert(err == AME_OK);

    /* 9. Verify result: each element should be N * (1.0 * 2.0) = 2 * N */
    printf("Kernel executed successfully!\n");
    printf("Sample C[0, 0] = %.1f (Expected: %.1f)\n\n", C[0], 2.0f * (float)N);

    /* 10. Report execution statistics */
    ame_stats_report(&ame, stdout);

    /* 11. Release backend ownership */
    ame_release(&ame);
}

int main(void) {
    single_tile_gemm_example();
    return 0;
}
```

### To Compile and Run This Example:

```bash
gcc -O3 -std=c99 example_kernel.c ztt_ame/ztt_ame.c -I./ztt_ame -lm -o example_kernel
./example_kernel
```
