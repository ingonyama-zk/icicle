# NTT - Number Theoretic Transform

## Overview

The Number Theoretic Transform (NTT) is a variant of the Fourier Transform used over finite fields, particularly those of integers modulo a prime number. NTT operates in a discrete domain and is used primarily in applications requiring modular arithmetic, such as cryptography and polynomial multiplication.

NTT is defined similarly to the Discrete Fourier Transform (DFT), but instead of using complex roots of unity, it uses roots of unity within a finite field. The definition hinges on the properties of the finite field, specifically the existence of a primitive root of unity of order $N$ (where $N$ is typically a power of 2), and the modulo operation is performed with respect to a specific prime number that supports these roots.

Formally, given a sequence of integers $a_0, a_1, ..., a_{N-1}$, the NTT of this sequence is another sequence of integers $A_0, A_1, ..., A_{N-1}$, computed as follows:

$$
A_k = \sum_{n=0}^{N-1} a_n \cdot \omega^{nk} \mod p
$$

where:

- $N$ is the size of the input sequence and is a power of 2,
- $p$ is a prime number such that $p = kN + 1$ for some integer $k$, ensuring that $p$ supports the existence of $N$th roots of unity,
- $\omega$ is a primitive $N$th root of unity modulo $p$, meaning $\omega^N \equiv 1 \mod p$ and no smaller positive power of $\omega$ is congruent to 1 modulo $p$,
- $k$ ranges from 0 to $N-1$, and it indexes the output sequence.

NTT is particularly useful because it enables efficient polynomial multiplication under modulo arithmetic, crucial for algorithms in cryptographic protocols and other areas requiring fast modular arithmetic operations.

There exists also INTT which is the inverse operation of NTT. INTT can take as input an output sequence of integers from an NTT and reconstruct the original sequence.

## C++ API

### Ordering

The `Ordering` enum defines how inputs and outputs are arranged for the NTT operation, offering flexibility in handling data according to different algorithmic needs or compatibility requirements. It primarily affects the sequencing of data points for the transform, which can influence both performance and the compatibility with certain algorithmic approaches. The available ordering options are:

- **`kNN` (Natural-Natural):** Both inputs and outputs are in their natural order. This is the simplest form of ordering, where data is processed in the sequence it is given, without any rearrangement.

- **`kNR` (Natural-Reversed):** Inputs are in natural order, while outputs are in bit-reversed order. This ordering is typically used in algorithms that benefit from having the output in a bit-reversed pattern.

- **`kRN` (Reversed-Natural):** Inputs are in bit-reversed order, and outputs are in natural order. This is often used with the Cooley-Tukey FFT algorithm.

- **`kRR` (Reversed-Reversed):** Both inputs and outputs are in bit-reversed order.

- **`kNM` (Natural-Mixed):** Inputs are provided in their natural order, while outputs are arranged in a digit-reversed (mixed) order. This ordering is good for mixed radix NTT operations, where the mixed or digit-reversed ordering of outputs is a generalization of the bit-reversal pattern seen in simpler, radix-2 cases.

- **`kMN` (Mixed-Natural):** Inputs are in a digit-reversed (mixed) order, while outputs are restored to their natural order. This ordering would primarily be used for mixed radix NTT

Choosing an algorithm is heavily dependent on your use case. For example Cooley-Tukey will often use `kRN` and Gentleman-Sande often uses `kNR`.

```cpp
enum class Ordering {
   kNN, /**< Inputs and outputs are in natural-order. */
   kNR, /**< Inputs are in natural-order and outputs are in bit-reversed-order. */
   kRN, /**< Inputs are in bit-reversed-order and outputs are in natural-order. */
   kRR, /**< Inputs and outputs are in bit-reversed-order. */
   kNM, /**< Inputs are in natural-order and outputs are in digit-reversed-order. */
   kMN  /**< Inputs are in digit-reversed-order and outputs are in natural-order. */
};
```

### `NTTConfig` Struct

The `NTTConfig` struct configures the NTT operation. It allows customization of parameters like the batch size, column batch computation, order of inputs and outputs etc.

```cpp
  template <typename S>
  struct NTTConfig {
    icicleStreamHandle stream; 
    S coset_gen;
    int batch_size;            
    bool columns_batch; 
    Ordering ordering;
    bool are_inputs_on_device;      
    bool are_outputs_on_device;     
    bool is_async;                  
    ConfigExtension* ext = nullptr; 
  };
```

#### Default configuration

You can obtain a default `NTTConfig` using:
```cpp
template <typename S>
static NTTConfig<S> default_ntt_config()
{
   NTTConfig<S> config = {
   nullptr,       // stream
   S::one(),      // coset_gen
   1,             // batch_size
   false,         // columns_batch
   Ordering::kNN, // ordering
   false,         // are_inputs_on_device
   false,         // are_outputs_on_device
   false,         // is_async
   };
   return config;
}
```

### NTT domain
Before computing an NTT, it is mandatory to initialize the roots of unity domain for computing the NTT.

:::note
NTT domain is constructed for a given size $2^N$ and can be used for any NTT of size smaller or equal to $2^N$. For example a domain of size 32 can be used to compute NTTs of size 2,4,8,16,32.
:::

```cpp
template <typename S>
eIcicleError ntt_init_domain(const S& primitive_root, const NTTInitDomainConfig& config);
```

:::note
Domain is constructed per device. When using multiple devices (e.g. GPUs), need to call it per device prior to calling ntt.
:::

To retrieve a root of unity from the domain:
```cpp
template <typename S> S get_root_of_unity(uint64_t max_size);
```

Finally, release the domain to free up device memory when not required:
```cpp
template <typename S> S get_root_of_unity(uint64_t max_size);
```

where

```cpp
struct NTTInitDomainConfig {
   icicleStreamHandle stream;      /**< Stream for asynchronous execution. */
   bool is_async;                  /**< True if operation is asynchronous. Default value is false. */
   ConfigExtension* ext = nullptr; /**< Backend-specific extensions. */
};

static NTTInitDomainConfig default_ntt_init_domain_config()
{
   NTTInitDomainConfig config = {
   nullptr, // stream
   false    // is_async
   };
   return config;
}
```

### `ntt` Function

the `ntt` function computes the NTT operation:

```cpp
template <typename S, typename E>
eIcicleError ntt(const E* input, int size, NTTDir dir, const NTTConfig<S>& config, E* output);

// Where NTTDir specific whether it is a forward or inverse transform
enum class NTTDir {
   kForward, /**< Perform forward NTT. */
   kInverse  /**< Perform inverse NTT (iNTT). */
};
```
### EC-NTT
[The ntt api](#ntt-function) works for ECNTT too, given correct types, for supported curves.

### Batch NTT

Batch NTT allows you to compute many NTTs with a single API call. Batch NTT can significantly reduce read/write times as well as computation overhead by executing multiple NTT operations in parallel. Batch mode may also offer better utilization of computational resources (memory and compute).

To compute a batch, set the `batch_size` and `columns_batch` fields of the config struct.

### Rust and Go bindings

- [Golang](../golang-bindings/ntt.md)
- [Rust](../rust-bindings/ntt.md)

### Example

The following example demonstartes how to use ntt and how pass custom configurations to the CUDA backend. This is discussed below.

```cpp
#include "icicle/backend/ntt_config.h"

// allocate and init input/output
int batch_size = /*...*/;
int log_ntt_size = /*...*/;
int ntt_size = 1 << log_ntt_size;
auto input = std::make_unique<scalar_t[]>(batch_size * ntt_size);
auto output = std::make_unique<scalar_t[]>(batch_size * ntt_size);
initialize_input(ntt_size, batch_size, input.get());

// Initialize NTT domain with fast twiddles (CUDA backend)
scalar_t basic_root = scalar_t::omega(log_ntt_size);
auto ntt_init_domain_cfg = default_ntt_init_domain_config();
ConfigExtension backend_cfg_ext;
backend_cfg_ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
ntt_init_domain_cfg.ext = &backend_cfg_ext;
ntt_init_domain(basic_root, ntt_init_domain_cfg);

// ntt configuration
NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
ConfigExtension ntt_cfg_ext;
config.batch_size = batch_size;

// Compute NTT with explicit selection of Mixed-Radix algorithm.
ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::MixedRadix);
config.ext = &ntt_cfg_ext;
ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get());
```

### CUDA backend NTT
This section describes the CUDA ntt implementation and how to use it.

Our CUDA NTT implementation supports two algorithms `radix-2` and `mixed-radix`.

### Radix 2

At its core, the Radix-2 NTT algorithm divides the problem into smaller sub-problems, leveraging the properties of "divide and conquer" to reduce the overall computational complexity. The algorithm operates on sequences whose lengths are powers of two.

1. **Input Preparation:**
   The input is a sequence of integers $a_0, a_1, \ldots, a_{N-1}, \text{ where } N$ is a power of two.

2. **Recursive Decomposition:**
   The algorithm recursively divides the input sequence into smaller sequences. At each step, it separates the sequence into even-indexed and odd-indexed elements, forming two subsequences that are then processed independently.

3. **Butterfly Operations:**
   The core computational element of the Radix-2 NTT is the "butterfly" operation, which combines pairs of elements from the sequences obtained in the decomposition step.

   Each butterfly operation involves multiplication by a "twiddle factor," which is a root of unity in the finite field, and addition or subtraction of the results, all performed modulo the prime modulus.

   $$
    X_k = (A_k + B_k \cdot W^k) \mod p
   $$

   $X_k$ - The output of the butterfly operation for the $k$-th element

   $A_k$ - an element from the even-indexed subset

   $B_k$ - an element from the odd-indexed subset

   $p$ - prime modulus

   $k$ - The index of the current operation within the butterfly or the transform stage

   The twiddle factors are precomputed to save runtime and improve performance.

4. **Bit-Reversal Permutation:**
   A final step involves rearranging the output sequence into the correct order. Due to the halving process in the decomposition steps, the elements of the transformed sequence are initially in a bit-reversed order. A bit-reversal permutation is applied to obtain the final sequence in natural order.

### Mixed Radix

The Mixed Radix NTT algorithm extends the concepts of the Radix-2 algorithm by allowing the decomposition of the input sequence based on various factors of its length. Specifically ICICLEs implementation splits the input into blocks of sizes 16, 32, or 64 compared to radix2 which is always splitting such that we end with NTT of size 2. This approach offers enhanced flexibility and efficiency, especially for input sizes that are composite numbers, by leveraging the "divide and conquer" strategy across multiple radices.

The NTT blocks in Mixed Radix are implemented more efficiently based on winograd NTT but also optimized memory and register usage is better compared to Radix-2.

Mixed Radix can reduce the number of stages required to compute for large inputs.

1. **Input Preparation:**
   The input to the Mixed Radix NTT is a sequence of integers $a_0, a_1, \ldots, a_{N-1}$, where $N$ is not strictly required to be a power of two. Instead, $N$ can be any composite number, ideally factorized into primes or powers of primes.

2. **Factorization and Decomposition:**
   Unlike the Radix-2 algorithm, which strictly divides the computational problem into halves, the Mixed Radix NTT algorithm implements a flexible decomposition approach which isn't limited to prime factorization.

   For example, an NTT of size 256 can be decomposed into two stages of $16 \times \text{NTT}_{16}$, leveraging a composite factorization strategy rather than decomposing into eight stages of $\text{NTT}_{2}$. This exemplifies the use of composite factors (in this case, $256 = 16 \times 16$) to apply smaller NTT transforms, optimizing computational efficiency by adapting the decomposition strategy to the specific structure of $N$.

3. **Butterfly Operations with Multiple Radices:**
   The Mixed Radix algorithm utilizes butterfly operations for various radix sizes. Each sub-transform involves specific butterfly operations characterized by multiplication with twiddle factors appropriate for the radix in question.

   The generalized butterfly operation for a radix-$r$ element can be expressed as:

   $$
   X_{k,r} = \sum_{j=0}^{r-1} (A_{j,k} \cdot W^{jk}) \mod p
   $$

   where:

   $X_{k,r}$ - is the output of the $radix-r$ butterfly operation for the $k-th$ set of inputs

   $A_{j,k}$ - represents the $j-th$ input element for the $k-th$ operation

   $W$ - is the twiddle factor

   $p$ - is the prime modulus

4. **Recombination and Reordering:**
   After applying the appropriate butterfly operations across all decomposition levels, the Mixed Radix algorithm recombines the results into a single output sequence. Due to the varied sizes of the sub-transforms, a more complex reordering process may be required compared to Radix-2. This involves digit-reversal permutations to ensure that the final output sequence is correctly ordered.

### Which algorithm should I choose ?

Both work only on inputs of power of 2 (e.g., 256, 512, 1024).

Radix 2 is faster for small NTTs. A small NTT would be around logN = 16 and batch size 1. Radix 2 won't necessarily perform better for smaller `logn` with larger batches.

Mixed radix on the other hand works better for larger NTTs with larger input sizes.

Performance really depends on logn size, batch size, ordering, inverse, coset, coeff-field and which GPU you are using.

For this reason we implemented our [heuristic auto-selection](https://github.com/ingonyama-zk/icicle/blob/main/icicle/src/ntt/ntt.cu#L573) which should choose the most efficient algorithm in most cases.

We still recommend you benchmark for your specific use case if you think a different configuration would yield better results.

To Explicitly choose the algorithm:

```cpp
#include "icicle/backend/ntt_config.h"

NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
ConfigExtension ntt_cfg_ext;
ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::MixedRadix);
config.ext = &ntt_cfg_ext;
ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get());
```


### Fast twiddles

When using the Mixed-radix algorithm, it is recommended to initialize the domain in "fast-twiddles" mode. This is essentially allocating the domain using extra memory but enables faster ntt.
To do so simply, pass this flag to the CUDA backend.

```cpp
#include "icicle/backend/ntt_config.h"

scalar_t basic_root = scalar_t::omega(log_ntt_size);
auto ntt_init_domain_cfg = default_ntt_init_domain_config();
ConfigExtension backend_cfg_ext;
backend_cfg_ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
ntt_init_domain_cfg.ext = &backend_cfg_ext;
ntt_init_domain(basic_root, ntt_init_domain_cfg);
```