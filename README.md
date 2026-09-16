# Simulador RISC-V Attached Matrix Extension (AME - Ztt v0.6)

Este diretório contém a implementação em C do mini-simulador da extensão de matrizes acopladas do RISC-V (**Attached Matrix Extension - Ztt v0.6**), desenvolvida como expansão do projeto **QDMAES**.

O objetivo deste componente é modelar fielmente a arquitetura, o banco de registradores, os registradores de controle e status (CSRs), os descritores de tipo de dados e as instruções do núcleo computacional da especificação oficial **Ztt v0.6**, com foco inicial na execução de multiplicação geral de matrizes (**GEMM**).

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Parâmetros de Hardware (Capítulo 4)](#2-parâmetros-de-hardware-capítulo-4)
3. [Descritores de Tipos de Dados (Seção 2.4)](#3-descritores-de-tipos-de-dados-seção-24)
4. [Estado do Hardware e CSRs (Capítulos 3 e 5)](#4-estado-do-hardware-e-csrs-capítulos-3-e-5)
5. [Instruções e Funções da API](#5-instruções-e-funções-da-api)
   - [5.1 Gerenciamento de Posse e Backend](#51-gerenciamento-de-posse-e-backend)
   - [5.2 Configuração de Metadados e Tipos](#52-configuração-de-metadados-e-tipos)
   - [5.3 Instruções de Limpeza (Zeroing)](#53-instruções-de-limpeza-zeroing)
   - [5.4 Carga e Escrita em Memória (Row-Major)](#54-carga-e-escrita-em-memória-row-major)
   - [5.5 Multiplicação-Acumulação Matricial 2D](#55-multiplicação-acumulação-matricial-2d)
   - [5.6 Movimentação de Registradores](#56-movimentação-de-registradores)
6. [Microkernel GEMM Tiled](#6-microkernel-gemm-tiled)
7. [Como Compilar e Executar](#7-como-compilar-e-executar)
8. [Suíte de Testes Automatizada](#8-suíte-de-testes-automatizada)

---

## 1. Visão Geral da Arquitetura

Diferente do modelo legado em `v0_Gem5` (que utilizava uma ISA customizada com rotações circulares manuais de registradores), a extensão **Ztt v0.6** adota o conceito de **quadrados lógicos (*squares*)** de dimensão $N \times N$:

- As matrizes são particionadas em blocos bidimensionais $N \times N$, onde $N = \sqrt{\text{AME\_NELEM}}$.
- Existem dois arquivos principais de registradores:
  - **Registradores de Matriz ($M$)**: Armazenam operandos de entrada para as multiplicações e resultados de operações elementwise. Se o tipo for maior que o tamanho base da unidade (`AME_UNIT_DATATYPE_SIZE`), múltiplos registradores $M$ consecutivos são agrupados logicamente.
  - **Registradores Acumuladores ($Acc$)**: Destinados exclusivamente a acumular os produtos matriciais das instruções `mmulacc`. Cada acumulador é dimensionado para conter um quadrado lógico completo do maior tipo suportado.
- Os registradores **não embutem o tipo de dados na instrução assembly**. Em vez disso, cada registrador $M_i$ e $Acc_j$ possui um registrador de metadados correspondente ($Md_i$ e $Ad_j$) configurado previamente por instruções dedicadas (`msettyp` e `asettyp`).

```
          +-------------------------------------------------+
          |                   Memória                       |
          +-------------------------------------------------+
                       ^                        |
               mss.rm  |                        | mls.rm
                       |                        v
          +-------------------------------------------------+
          |           Registradores de Matriz (M)           |
          |       M0 .. M31 (Metadados: Md0 .. Md31)        |
          +-------------------------------------------------+
                 ^                      |            |
        mmov.m.a |                      | ms1        | ms2
                 |                      v            v
          +---------------+         +-----------------------+
          |  Acumuladores | <------ |      mmulacc.2d       |
          |  Acc0 .. Acc7 |  C +=   | C = C + (ms1 x ms2)   |
          | (Ad0 .. Ad7)  |         +-----------------------+
          +---------------+
```

---

## 2. Parâmetros de Hardware (Capítulo 4)

O comportamento do hardware é moldado pelas seguintes constantes (definidas em `ztt_ame.h`):

| Constante | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| `AME_NELEM` | `64` | Número fixo de elementos em um quadrado lógico. Deve ser uma potência de 4 ($16, 64, 256, 1024, 4096, 16384$). |
| `AME_N` | `8` | Dimensão linear da matriz quadrada: $N = \sqrt{\text{AME\_NELEM}}$. Deve ser uma potência de 2. |
| `AME_NUM_M_REGS` | `32` | Quantidade de registradores de matriz gerais ($M_0$ a $M_{31}$). A especificação permite 16 ou 32. |
| `AME_NUM_ACC_REGS`| `8` | Quantidade de acumuladores implementados ($Acc_0$ a $Acc_7$). A especificação permite 1, 2, 4, 8 ou 16. |
| `AME_UNIT_DATATYPE_SIZE` | `32` bits | Tamanho em bits no qual exatamente 1 quadrado lógico cabe em 1 registrador $M$. |
| `AME_M_REG_SIZE_BYTES` | `256` bytes | Tamanho físico em bytes de um registrador $M$: $\frac{\text{AME\_NELEM} \times \text{AME\_UNIT\_DATATYPE\_SIZE}}{8}$. |
| `AME_ACC_REG_SIZE_BYTES`| `512` bytes | Tamanho físico em bytes de um acumulador (para suportar até tipos de 64 bits por elemento). |

### Regra de Agrupamento (*Register Grouping*)
Para tipos de dados cujo tamanho em bits $E$ seja superior a `AME_UNIT_DATATYPE_SIZE`:
$$\text{group\_size} = \max\left(1, \left\lceil \frac{E}{\text{AME\_UNIT\_DATATYPE\_SIZE}} \right\rceil\right)$$
- **Alinhamento obrigatório**: O índice do registrador $M$ base **deve ser múltiplo de `group_size`**. Caso contrário, o hardware dispara uma exceção de instrução ilegal (`AME_ERR_ILLEGAL_INSN`).

---

## 3. Descritores de Tipos de Dados (Seção 2.4)

Cada tipo de dado na Ztt é representado por um descritor de **32 bits**:
- **Bit 31**: `0` = Tipo padrão RISC-V, `1` = Tipo customizado do fabricante.
- **Bits 30:8**: Propriedades específicas da classe (sinal, expoente, saturação, arredondamento).
- **Bits 7:0**: Largura do elemento em bits ($E$).

### 3.1 Inteiros Padrão (`Bit 8 = 0`)
- **Bit 30**: `0` = Não-sinalizado (*unsigned*), `1` = Sinalizado (*signed*).
- **Bit 29**: `1` = Saturação ativada, `0` = Overflow com wrap-around.
- **Bits 28:27**: Modo de arredondamento (`00`=RNU, `01`=RNE, `10`=RDN, `11`=ROD).
- **Bits 26:9**: Reservados (devem ser zero).

Constantes pré-definidas:
- `AME_DTYPE_INT8` (`0x40000008`): Inteiro sinalizado de 8 bits.
- `AME_DTYPE_UINT8` (`0x00000008`): Inteiro sem sinal de 8 bits.
- `AME_DTYPE_INT16` (`0x40000010`): Inteiro sinalizado de 16 bits.
- `AME_DTYPE_UINT16` (`0x00000010`): Inteiro sem sinal de 16 bits.
- `AME_DTYPE_INT32` (`0x40000020`): Inteiro sinalizado de 32 bits.
- `AME_DTYPE_UINT32` (`0x00000020`): Inteiro sem sinal de 32 bits.

### 3.2 Ponto Flutuante Padrão IEEE (`Bit 8 = 1`)
- **Bits 30:26**: Largura do expoente em bits (`exp_width`), ex: 5 para FP16, 8 para FP32/BF16.
- **Bits 25:22**: Modo de arredondamento IEEE (`0000` = RNE - Round to Nearest Even).
- **Bit 21**: `1` = Suporte a $\pm\infty$; `0` = Satura para min/max finito.
- **Bit 20**: `1` = Suporte a números subnormais/desnormalizados.
- **Bits 19:9**: Reservados (devem ser zero).

Constantes pré-definidas:
- `AME_DTYPE_FP32` (`0x20300120`): Float IEEE 754 de 32 bits (expoente 8 bits, mantissa 23 bits).
- `AME_DTYPE_FP16` (`0x14300110`): Half-precision IEEE 754 de 16 bits (expoente 5 bits).
- `AME_DTYPE_BF16` (`0x20300110`): Bfloat16 de 16 bits (expoente 8 bits, mantissa 7 bits).

---

## 4. Estado do Hardware e CSRs (Capítulos 3 e 5)

A estrutura `ame_state_t` encapsula todo o estado arquitetural do acelerador:

```c
typedef struct {
    uint8_t M[AME_NUM_M_REGS][AME_M_REG_SIZE_BYTES];     /* Registradores M0..M31 */
    uint8_t Acc[AME_NUM_ACC_REGS][AME_ACC_REG_SIZE_BYTES]; /* Acumuladores Acc0..Acc7 */
    uint32_t Md[AME_NUM_M_REGS];                          /* Metadados de tipo de M */
    uint32_t Ad[AME_NUM_ACC_REGS];                        /* Metadados de tipo de Acc */

    uint32_t amenlen;   /* [RO] Comprimento lateral do tile: N = sqrt(AME_NELEM) */
    uint32_t ameudsz;   /* [RO] AME_UNIT_DATATYPE_SIZE (32 bits) */
    uint32_t amestype;  /* Tipo de dados escalar para operandos X */
    uint32_t ameown;    /* Bit 0: OWNED (1 se o contexto atual detiver o backend) */
    uint32_t amefflags; /* Sticky flags de exceção FP (NV, DZ, OF, UF, NX) */
    uint32_t amexsat;   /* Sticky flag de saturação de inteiros (SAT) */
    uint32_t amestatus; /* Bit 0: UN (Operação ou tupla de tipos não suportada) */

    ame_ms_state_t ms_status; /* Estado do mstatus.MS (Off, Initial, Clean, Dirty) */
} ame_state_t;
```

### Regras de Posse e Exceções (Seção 5.2)
1. Se `ms_status == AME_MS_OFF`, qualquer instrução AME dispara exceção (`AME_ERR_ILLEGAL_INSN`).
2. Se `ms_status != Off`, mas `ameown.OWNED == 0`, qualquer instrução exceto `ame.acquire` e `ame.release` é rejeitada com erro de posse (`AME_ERR_NOT_OWNED`).
3. Se o hardware estiver adquirido (`ameown.OWNED == 1`), o software é impedido de desligar o acelerador (`ms_status = Off`) sem antes executar `ame.release`.

---

## 5. Instruções e Funções da API

Todas as instruções retornam `AME_OK` (`0`) em caso de sucesso, ou um código negativo de erro (`AME_ERR_*`).

### 5.1 Gerenciamento de Posse e Backend

#### `ame_err_t ame_acquire(ame_state_t *ame, uint32_t *rd, uint32_t rs1_desc)`
- **Mnemônico Ztt**: `ame.acquire rd, rs1`
- **Funcionamento**: Requisita a posse atômica do backend AME para o contexto atual.
  - Verifica se `ms_status != AME_MS_OFF`.
  - Valida o descritor de aquisição `rs1_desc` (bits reservados `[31:6]` devem ser zero).
  - Define `ameown.OWNED = 1` e transiciona o estado para `AME_MS_DIRTY`.
  - Escreve em `*rd` o valor `AME_ACQ_SUCCESS | AME_ACQ_STATUS_GRANTED` (`0x01`).

#### `ame_err_t ame_release(ame_state_t *ame)`
- **Mnemônico Ztt**: `ame.release`
- **Funcionamento**: Libera o backend AME, limpando o bit `ameown.OWNED = 0`.

---

### 5.2 Configuração de Metadados e Tipos

#### `ame_err_t ame_msettyp(ame_state_t *ame, uint32_t md, uint32_t rs1_val)`
- **Mnemônico Ztt**: `msettyp md, rs1`
- **Funcionamento**: Associa o descritor de tipo contido em `rs1_val` ao registrador de matriz `M[md]`.
  - Calcula o tamanho do grupo (`group_size`).
  - Valida se `md` é múltiplo de `group_size`. Se não for, falha com `AME_ERR_ILLEGAL_INSN`.
  - Escreve `Md[md] = rs1_val`.
  - **Limpeza física de bits brutos**: Zera todos os bytes de armazenamento físico dos registradores do grupo `M[md .. md+group_size-1]`.

#### `ame_err_t ame_asettyp(ame_state_t *ame, uint32_t ad, uint32_t rs1_val)`
- **Mnemônico Ztt**: `asettyp ad, rs1`
- **Funcionamento**: Associa o descritor de tipo contido em `rs1_val` ao acumulador `Acc[ad]`.
  - Escreve `Ad[ad] = rs1_val`.
  - **Limpeza física de bits brutos**: Zera todos os bytes de armazenamento físico do acumulador `Acc[ad]`.

#### `ame_err_t ame_mgettyp(const ame_state_t *ame, uint32_t *rd, uint32_t md)`
#### `ame_err_t ame_agettyp(const ame_state_t *ame, uint32_t *rd, uint32_t ad)`
- **Mnemônicos Ztt**: `mgettyp rd, md` / `agettyp rd, ad`
- **Funcionamento**: Lê o descritor de 32 bits configurado em `Md[md]` ou `Ad[ad]` e armazena em `*rd`.

---

### 5.3 Instruções de Limpeza (Zeroing)

#### `ame_err_t ame_mzero_2d_acc(ame_state_t *ame, uint32_t acc)`
- **Mnemônico Ztt**: `mzero.2d.acc acc`
- **Funcionamento**: Preenche todos os elementos do acumulador `Acc[acc]` com o zero semântico do seu tipo de dados (`+0.0f` para floats, `0` para inteiros). É utilizado antes do loop de redução do GEMM.

#### `ame_err_t ame_mzero_2d_m(ame_state_t *ame, uint32_t md)`
- **Mnemônico Ztt**: `mzero.2d.m md`
- **Funcionamento**: Preenche todos os elementos do registrador $M$ (ou grupo) com o zero semântico.

---

### 5.4 Carga e Escrita em Memória (Row-Major)

#### `ame_err_t ame_mls_rm(ame_state_t *ame, uint32_t md, const void *base_addr)`
- **Mnemônico Ztt**: `mls.rm md, rs1`
- **Funcionamento**: Carrega um quadrado $N \times N$ da memória contígua em formato Row-Major para o registrador `M[md]` (ou seu grupo de registradores).
  - Lê o tipo de dados configurado em `Md[md]`.
  - **Verificação de alinhamento natural**: Exige que o endereço de memória `base_addr` seja alinhado ao tamanho do elemento em bytes: `base_addr % elem_bytes == 0`. Se desalinhado, retorna `AME_ERR_MISALIGNED`.
  - Transfere exatamente $N \times N \times \text{elem\_bytes}$ bytes para o registrador.

#### `ame_err_t ame_mss_rm(const ame_state_t *ame, uint32_t ms, void *base_addr)`
- **Mnemônico Ztt**: `mss.rm ms, rs1`
- **Funcionamento**: Descarrega o quadrado $N \times N$ do registrador `M[ms]` (ou grupo) para a memória contígua em formato Row-Major. Também valida o alinhamento natural de endereço.

---

### 5.5 Multiplicação-Acumulação Matricial 2D

#### `ame_err_t ame_mmulacc_2d(ame_state_t *ame, uint32_t acc, uint32_t ms1, uint32_t ms2)`
- **Mnemônico Ztt**: `mmulacc.2d acc, ms1, ms2`
- **Funcionamento**: Realiza a multiplicação matricial bidimensional dos blocos $N \times N$ de $M[ms1]$ e $M[ms2]$ e **acumula** o resultado em `Acc[acc]`:
  $$C_{\text{acc}}[i, j] = C_{\text{acc}}[i, j] + \sum_{k=0}^{N-1} A_{ms1}[i, k] \times B_{ms2}[k, j]$$
  - Valida o alinhamento dos registradores $ms1$ e $ms2$ com seus respectivos `group_size`.
  - Inspeciona os descritores de tipo `Ad[acc]`, `Md[ms1]` e `Md[ms2]`.
  - **Tuplas suportadas na Fase 1**:
    1. **FP32**: `AME_DTYPE_FP32` $\times$ `AME_DTYPE_FP32` $\to$ `AME_DTYPE_FP32` (precisão simples IEEE 754).
    2. **Int32**: `AME_DTYPE_INT32` $\times$ `AME_DTYPE_INT32` $\to$ `AME_DTYPE_INT32` (inteiros sinalizados com acumulação em 64 bits truncada).
    3. **Uint32**: `AME_DTYPE_UINT32` $\times$ `AME_DTYPE_UINT32` $\to$ `AME_DTYPE_UINT32`.
  - **Tratamento de Incompatibilidade**: Se os tipos não forem suportados ou forem incompatíveis (ex: multiplicar FP32 com Int32), a instrução **não modifica** o acumulador, define a sticky flag `amestatus.UN = 1` e retorna `AME_ERR_UNSUPPORTED`.

---

### 5.6 Movimentação de Registradores

#### `ame_err_t ame_mmov_m_a(ame_state_t *ame, uint32_t md, uint32_t acc)`
- **Mnemônico Ztt**: `mmov.m.a md, acc`
- **Funcionamento**: Transfere o quadrado contido no acumulador `Acc[acc]` para o registrador de matriz `M[md]`.
  - **Regra estrita da Ztt v0.6**: O tipo configurado em `Ad[acc]` **deve ser idêntico** ao tipo em `Md[md]`. Se diferirem, a instrução não modifica `M[md]`, seta a flag sticky `amestatus.UN = 1` e retorna `AME_ERR_UNSUPPORTED`.
  - Transfere todos os $N \times N$ elementos para `M[md]` (ou seu grupo).

#### `ame_err_t ame_mmov_a_m(ame_state_t *ame, uint32_t acc, uint32_t ms)`
- **Mnemônico Ztt**: `mmov.a.m acc, ms`
- **Funcionamento**: Cópia inversa, transferindo do registrador $M[ms]$ para o acumulador `Acc[acc]`, com a mesma validação estrita de correspondência de tipos.

---

## 6. Microkernel GEMM Tiled

Os arquivos `ztt_gemm.h` e `ztt_gemm.c` implementam os kernels de alto nível:
- `ztt_sgemm(...)`: Multiplicação de matrizes em ponto flutuante FP32.
- `ztt_igemm(...)`: Multiplicação de matrizes em inteiros Int32.

### Algoritmo de Tiling ($M \times K \times P$)
Dadas as matrizes $A [M \times K]$, $B [K \times P]$ e $C [M \times P]$, a multiplicação $C += A \times B$ é particionada em blocos de tamanho fixo $N \times N$:

```c
for (int i = 0; i < M; i += N) {
    for (int j = 0; j < P; j += N) {
        // 1. Zera o acumulador de hardware para o tile C[i, j]
        ame_mzero_2d_acc(ame, 0);

        for (int k = 0; k < K; k += N) {
            // 2. Extrai e carrega os blocos A[i, k] e B[k, j] com mls.rm
            ame_mls_rm(ame, 0, tile_A); // carrega em m0
            ame_mls_rm(ame, 1, tile_B); // carrega em m1

            // 3. Executa a multiplicação-acumulação: acc0 += m0 * m1
            ame_mmulacc_2d(ame, 0, 0, 1);
        }

        // 4. Move o tile acumulado de volta para m0 e descarrega para a memória
        ame_mmov_m_a(ame, 0, 0);
        ame_mss_rm(ame, 0, tile_C);

        // 5. Soma o resultado ao bloco de destino C[i..i+N, j..j+N]
    }
}
```

- **Tratamento de Bordas**: Quando as dimensões $M, K, P$ não são múltiplos exatos de $N$, as funções realizam preenchimento com zeros (*zero-padding*) nas margens dos blocos temporários de entrada e acumulam apenas os elementos válidos na matriz de saída.

---

## 7. Como Compilar e Executar

O diretório conta com um `Makefile` automatizado.

### Compilação Padrão e Testes
```bash
cd ztt_ame
make test
```

### Configuração Dinâmica da Dimensão do Tile ($N$)
É possível compilar o simulador com qualquer dimensão válida para `AME_NELEM` via flag `CFLAGS`:

```bash
# Tile N = 4 (AME_NELEM = 16)
make clean && make test CFLAGS="-Wall -Wextra -O3 -std=c99 -DAME_NELEM=16"

# Tile N = 8 (AME_NELEM = 64) - Padrão
make clean && make test CFLAGS="-Wall -Wextra -O3 -std=c99 -DAME_NELEM=64"

# Tile N = 16 (AME_NELEM = 256)
make clean && make test CFLAGS="-Wall -Wextra -O3 -std=c99 -DAME_NELEM=256"

# Tile N = 128 (AME_NELEM = 16384) - Dimensão do exemplo Ztt v0.6
make clean && make test CFLAGS="-Wall -Wextra -O3 -std=c99 -DAME_NELEM=16384"
```

### Cross-Compilação para RISC-V Real
Caso possua a toolchain GNU para RISC-V instalada:
```bash
make riscv
```
Isto gerará o binário `test_ztt_riscv`.

---

## 8. Suíte de Testes Automatizada

O arquivo `test_ztt.c` contém testes unitários e de integração que validam cada aspecto da especificação:

1. **`test_ownership`**:
   - Confirma que instruções executadas antes de `ame_acquire` falham com `AME_ERR_NOT_OWNED`.
   - Valida a concessão de posse (`ameown.OWNED = 1`, `rd = GRANTED`).
   - Valida a revogação de posse após `ame.release`.
2. **`test_datatypes_and_zeroing`**:
   - Valida a escrita e leitura de metadados (`msettyp` / `mgettyp` / `asettyp` / `agettyp`).
   - Verifica se `msettyp` e `asettyp` realizam a limpeza física de bits brutos com zero em registradores previamente sujos com bytes `0xAA` e `0x55`.
3. **`test_memory_roundtrip`**:
   - Carrega um vetor ordenado em ponto flutuante na memória com `mls.rm` e descarrega com `mss.rm`, validando a integridade bit a bit.
4. **`test_type_mismatch`**:
   - Configura intencionalmente tipos divergentes entre $M$ e $Acc$ (ex: FP32 vs Int32).
   - Valida a rejeição da operação com `AME_ERR_UNSUPPORTED` e a ativação da flag sticky `amestatus.UN`.
5. **`test_single_tile_sgemm`**:
   - Executa `mmulacc.2d` em um quadrado isolado $N \times N$ (FP32) e compara contra a multiplicação de referência clássica em CPU.
6. **`test_single_tile_igemm`**:
   - Executa `mmulacc.2d` em um quadrado isolado $N \times N$ (Int32) com elementos positivos e negativos, comparando contra a referência de CPU.
7. **`test_full_sgemm`**:
   - Multiplicação multi-tile com matrizes arbitrárias de floats ($M=24, K=20, P=28$). Valida que o erro máximo contra a CPU é inferior a $10^{-4}$.
8. **`test_full_igemm`**:
   - Multiplicação multi-tile com matrizes arbitrárias de inteiros ($M=24, K=20, P=28$). Valida igualdade exata com a CPU.
