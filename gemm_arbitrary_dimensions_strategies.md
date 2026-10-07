# Estratégias para GEMM com Dimensões Arbitrárias no RISC-V AME (Ztt v0.6)

Este documento registra as opções arquiteturais e de software para suportar multiplicações de matrizes com dimensões arbitrárias ($M, K, P$ não múltiplos de $N = \sqrt{\text{AME\_NELEM}}$) utilizando a extensão matricial **RISC-V AME (Ztt v0.6)**.

---

## 1. Contexto do Problema

Os microkernels acelerados por hardware (`ztt_sgemm_strided` e `ztt_igemm_strided`) operam em blocos atômicos $N \times N$. Cargas e escritas de memória (`mls.st`, `mss.st`, `mls.rm`, `mss.rm`) transferem ladrilhos inteiros de tamanho fixo:
$$\text{AME\_NELEM} = N \times N$$

Quando $M$, $K$ ou $P$ não são múltiplos exatos de $N$, as bordas da matriz não preenchem um ladrilho completo. Três estratégias principais resolvem essa incompatibilidade no nível de biblioteca ou runtime.

---

## 2. Estratégias de Implementação

### Estratégia 1: Interior em Hardware + Limpeza de Bordas (Fringe Peeling)
*Padrão adotado em bibliotecas BLAS industriais (OpenBLAS, BLIS, CUTLASS, oneDNN) e sugerido na Seção 7.2 do Ztt v0.6.*

A matriz é particionada em:
1. **Núcleo Interior (Full Tiles)**:
   $$M_{\text{core}} = \lfloor M / N \rfloor \times N, \quad K_{\text{core}} = \lfloor K / N \rfloor \times N, \quad P_{\text{core}} = \lfloor P / N \rfloor \times N$$
   Executado diretamente no acelerador AME via `ztt_sgemm_strided` na velocidade máxima do hardware, sem qualquer overhead.
2. **Faixas de Borda (Fringes)**:
   - **Borda Direita**: $M_{\text{core}} \times K_{\text{core}} \times (P - P_{\text{core}})$
   - **Borda Inferior**: $(M - M_{\text{core}}) \times K_{\text{core}} \times P_{\text{core}}$
   - **Canto Inferior Direito**: $(M - M_{\text{core}}) \times \dots \times (P - P_{\text{core}})$
   - **Cauda da Redução $K$**: $M \times (K - K_{\text{core}}) \times P$

```text
       K_core        cauda
    +--------------+----+
    |              |    |
M_c |  Interior    |    |   Interior executado via ztt_sgemm_strided()
    |  (full tiles)|    |
    +--------------+----+
rem |  Ladrilhos   |Canto   Bordas e cantos tratados via limpeza
    +--------------+----+
```

#### Abordagens para o cálculo das bordas:
- **Opção 1A (Fallback Escalar / Vetorial RVV)**:
  O núcleo host RISC-V finaliza as bordas finas com laços escalares simples ou instruções vetoriais 1D (`vfmacc`). Para matrizes médias e grandes, o volume computacional das bordas é desprezível frente ao interior ($O(N \cdot M)$ vs. $O(M \cdot K \cdot P)$).
- **Opção 1B (Scratchpad Local para Tiles de Borda)**:
  Para os ladrilhos residuais na fronteira, copia-se o bloco parcial para um pequeno buffer temporário $N \times N$ alocado na pilha com preenchimento em zero, executa-se a instrução de multiplicação do AME, e escreve-se de volta em $C$ apenas os elementos válidos.

---

### Estratégia 2: Pré-Padding na Alocação (Allocation-Time Padding & Strided Views)
*Padrão comum em frameworks de Deep Learning (PyTorch, TensorFlow, XLA).*

Se o runtime gerencia a criação dos tensores:
1. Aloca-se a matriz em memória com dimensões alinhadas para cima em múltiplos de $N$:
   $$\text{lda} = \text{round\_up}(K, N), \quad \text{ldb} = \text{round\_up}(P, N), \quad \text{ldc} = \text{round\_up}(P, N)$$
2. Os elementos extras de padding são zerados na inicialização.
3. Chama-se diretamente o microkernel strided com as dimensões arredondadas:
   $$\text{ztt\_sgemm\_strided}(\dots, M_{\text{pad}}, K_{\text{pad}}, P_{\text{pad}}, \dots, \text{ldc}, \text{lda}, \text{ldb})$$
4. O código consumidor lê apenas a sub-visão lógica $M \times P$ de $C$.
- **Vantagem**: Zero cópias e zero desempacotamento em runtime.

---

### Estratégia 3: Predicação / Mascaramento de Ladrilhos em Hardware (Futuro)
*Presente em arquiteturas com suporte explícito a predicados matriciais (ex: ARM SME com SVL, NVIDIA TMA com out-of-bounds zeroing).*

- O registrador de configuração recebe `M_valid`, `K_valid` e `P_valid`.
- Ao carregar fora dos limites lógicos, o hardware insere `0.0f` automaticamente.
- Ao salvar em $C$, linhas e colunas excedentes são ignoradas sem disparar faltas de página.
- *Nota no Ztt v0.6*: A especificação base atual não define registradores de predicado 2D para `mls.st`/`mss.st`, sendo necessário o tratamento via software (Estratégias 1 ou 2).

---

## 3. Matriz Comparativa

| Estratégia | Complexidade de Código | Overhead de Memória | Overhead Computacional | Cenário Ideal |
| :--- | :--- | :--- | :--- | :--- |
| **1A (Interior AME + Borda CPU)** | Média | Nulo (0 bytes extras) | Desprezível ($< 2\%$ em matrizes reais) | Bibliotecas BLAS gerais (`cblas_gemm`) |
| **1B (Borda via Scratchpad)** | Baixa | Temporário na pilha ($3 \times N^2$ elementos) | Mínimo (apenas cópia da borda) | Microcontroladores / sistemas embarcados sem RVV |
| **2 (Padding na Alocação)** | Muito Baixa | Pequeno desperdício de memória nas margens | Zero cópias em runtime | Modelos de Machine Learning e compiladores de tensores |

---

## 4. Próximos Passos para Implementação

Quando for implementado o suporte geral:
1. Definir uma interface de alto nível `ztt_gemm_general(...)` em `ztt_gemm.h`.
2. Implementar a **Estratégia 1B** (scratchpad de pilha) ou **Estratégia 1A** (fallback escalar) para manter independência de alocadores externos.
3. Adicionar testes unitários dedicados em `test_ztt.c` cobrindo matrizes com dimensões primos e não-múltiplos (ex: $17 \times 23 \times 19$).
