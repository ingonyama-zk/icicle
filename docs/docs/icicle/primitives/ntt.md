# NTT - Number Theoretic Transform

The Number Theoretic Transform (NTT) is a variant of the Fourier Transform used over finite fields or rings, particularly those of integers modulo a prime number. NTT operates in a discrete domain and is used primarily in applications requiring modular arithmetic, such as cryptography and polynomial multiplication.

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

The NTT is particularly useful because it enables efficient polynomial multiplication under modulo arithmetic, crucial for algorithms in cryptographic protocols, and other areas requiring fast modular arithmetic operations. 

There exists also INTT which is the inverse operation of NTT. INTT can take as input an output sequence of integers from an NTT and reconstruct the original sequence.

# Using NTT

### Supported curves

NTT supports the following curves:

`bls12-377`, `bls12-381`, `bn-254`, `bw6-761`


### Examples

- [Rust API examples](https://github.com/ingonyama-zk/icicle/blob/d84ffd2679a4cb8f8d1ac2ad2897bc0b95f4eeeb/examples/rust/ntt/src/main.rs#L1)

- [C++ API examples](https://github.com/ingonyama-zk/icicle/blob/d84ffd2679a4cb8f8d1ac2ad2897bc0b95f4eeeb/examples/c%2B%2B/ntt/example.cu#L1)

## NTT API overview

```rust
pub fn ntt<F>(
    input: &HostOrDeviceSlice<F>,
    dir: NTTDir,
    cfg: &NTTConfig<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
```

`ntt:ntt` expects:

`input` - buffer to read the inputs of the NTT from. <br/>
`dir` - whether to compute forward or inverse NTT. <br/>
`cfg` - config used to specify extra arguments of the NTT. <br/>
`output` - buffer to write the NTT outputs into. Must be of the same  size as input.

The `input` and `output` buffers can be on device or on host. Being on host means that they will be transferred to device during runtime.

### NTT Config

```rust
pub struct NTTConfig<'a, S> {
    pub ctx: DeviceContext<'a>,
    pub coset_gen: S,
    pub batch_size: i32,
    pub ordering: Ordering,
    are_inputs_on_device: bool,    
    are_outputs_on_device: bool,
    pub is_async: bool,
    pub ntt_algorithm: NttAlgorithm,
}
```

The `NTTConfig` struct is a configuration object used to specify parameters for an NTT instance.

#### Fields

- **`ctx: DeviceContext<'a>`**: Specifies the device context, including the device ID and the stream ID.

- **`coset_gen: S`**: Defines the coset generator used for coset (i)NTTs. By default, this is set to `S::one()`, indicating that no coset is being used.

- **`batch_size: i32`**: Determines the number of NTTs to compute in a single batch. The default value is 1, meaning that operations are performed on individual inputs without batching. Batch processing can significantly improve performance by leveraging parallelism in GPU computations.

- **`ordering: Ordering`**: Controls the ordering of inputs and outputs for the NTT operation. This field can be used to specify decimation strategies (in time or in frequency) and the type of butterfly algorithm (Cooley-Tukey or Gentleman-Sande). The ordering is crucial for compatibility with various algorithmic approaches and can impact the efficiency of the NTT.

- **`are_inputs_on_device: bool`**: Indicates whether the input data has been preloaded on the device memory. If `false` inputs will be copied from host to device.

- **`are_outputs_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device. If the inputs and outputs are the same pointer NTT will be computed in place.

- **`is_async: bool`**: Specifies whether the NTT operation should be performed asynchronously. When set to `true`, the NTT function will not block the CPU, allowing other operations to proceed concurrently. Asynchronous execution requires careful synchronization to ensure data integrity and correctness.

- **`ntt_algorithm: NttAlgorithm`**: Can be one of `Auto`, `Radix2`, `MixedRadix`.
`Auto` will select `Radix 2` or `Mixed Radix` algorithm based on heuristics.
`Radix2` and `MixedRadix` will force the use of an algorithm regardless of the input size or other considerations. You should use one of these options when you know for sure that you want to 


#### Usage

Example initialization with default settings:

```rust
let default_config = NTTConfig::default();
```

Customizing the configuration:

```rust
let custom_config = NTTConfig {
    ctx: custom_device_context,
    coset_gen: my_coset_generator,
    batch_size: 10,
    ordering: Ordering::kRN,
    are_inputs_on_device: true,
    are_outputs_on_device: true,
    is_async: false,
    ntt_algorithm: NttAlgorithm::MixedRadix,
};
```

### Ordering

The `Ordering` enum defines how inputs and outputs are arranged for the NTT operation, offering flexibility in handling data according to different algorithmic needs or compatibility requirements. It primarily affects the sequencing of data points for the transform, which can influence both performance and the compatibility with certain algorithmic approaches. The available ordering options are:

- **`kNN` (Natural-Natural):** Both inputs and outputs are in their natural order. This is the simplest form of ordering, where data is processed in the sequence it is given, without any rearrangement.

- **`kNR` (Natural-Reversed):** Inputs are in natural order, while outputs are in bit-reversed order. This ordering is typically used in algorithms that benefit from having the output in a bit-reversed pattern.

- **`kRN` (Reversed-Natural):** Inputs are in bit-reversed order, and outputs are in natural order. This is often used with the Cooley-Tukey FFT algorithm.

- **`kRR` (Reversed-Reversed):** Both inputs and outputs are in bit-reversed order.

- **`kNM` (Natural-Mixed):** Inputs are provided in their natural order, while outputs are arranged in a digit-reversed (mixed) order. This ordering is good for mixed radix NTT operations, where the mixed or digit-reversed ordering of outputs is a generalization of the bit-reversal pattern seen in simpler, radix-2 cases.

- **`kMN` (Mixed-Natural):** Inputs are in a digit-reversed (mixed) order, while outputs are restored to their natural order. This ordering would primarily be used for mixed radix NTT

Choosing an algorithm is heavily dependent on your use case. For example Cooley-Tukey will often use `kRN` and Gentleman-Sande often uses `kNR`.

### Modes

NTT also supports two different modes `Batch NTT` and `Single NTT`

Batch NTT allows you to run many NTTs with a single API call, Single MSM will launch a single MSM computation.

You may toggle between single and batch NTT by simply configure `batch_size` to be larger then 1 in your `NTTConfig`.

```rust
let mut cfg = ntt::get_default_ntt_config::<ScalarField>();
cfg.batch_size = 10 // your ntt using this config will run in batch mode.
```

`batch_size=1` would keep our NTT in single NTT mode.

Deciding weather to use `batch NTT` vs `single NTT` is highly dependent on your application and use case.

**Single NTT Mode**

- Choose this mode when your application requires processing individual NTT operations in isolation.

**Batch NTT Mode**

- Batch NTT mode can significantly reduce read/write as well as computation overhead by executing multiple NTT operations in parallel.

- Batch mode may also offer better utilization of computational resources (memory and compute).

## Supported algorithms

Our NTT implementation supports two algorithms `radix-2` and `mixed-radix`.

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

The Mixed Radix NTT algorithm extends the concepts of the Radix-2 algorithm by allowing the decomposition of the input sequence based on various factors of its length. Specifically ICICLEs implementation splits the input into blocks of sizes 16,32,64 compared to radix2 which is always splitting such that we end with NTT of size 2. This approach offers enhanced flexibility and efficiency, especially for input sizes that are composite numbers, by leveraging the "divide and conquer" strategy across multiple radixes.

The NTT blocks in Mixed Radix are implemented more efficiently based on winograd NTT but also optimized memory and register usage is better compared to Radix-2.

Mixed Radix can reduce the number of stages required to compute for large inputs.

1. **Input Preparation:**
   The input to the Mixed Radix NTT is a sequence of integers $a_0, a_1, \ldots, a_{N-1}$, where $N$ is not strictly required to be a power of two. Instead, $N$ can be any composite number, ideally factorized into primes or powers of primes.

2. **Factorization and Decomposition:**
   Unlike the Radix-2 algorithm, which strictly divides the computational problem into halves, the Mixed Radix NTT algorithm implements a flexible decomposition approach which isn't limited to prime factorization. 
   
   For example, an NTT of size 256 can be decomposed into two stages of $16 \times \text{NTT}_{16}$, leveraging a composite factorization strategy rather than decomposing into eight stages of $\text{NTT}_{2}$. This exemplifies the use of composite factors (in this case, $256 = 16 \times 16$) to apply smaller NTT transforms, optimizing computational efficiency by adapting the decomposition strategy to the specific structure of $N$.

3. **Butterfly Operations with Multiple Radixes:**
   The Mixed Radix algorithm utilizes butterfly operations for various radix sizes. Each sub-transform involves specific butterfly operations characterized by multiplication with twiddle factors appropriate for the radix in question.

   The generalized butterfly operation for a radix-$r$ element can be expressed as:

   $$
   X_{k,r} = \sum_{j=0}^{r-1} (A_{j,k} \cdot W^{jk}) \mod p
   $$

   where $X_{k,r}$ is the output of the $radix-r$ butterfly operation for the $k-th$ set of inputs, $A_{j,k}$ represents the $j-th$ input element for the $k-th$ operation, $W$ is the twiddle factor, and $p$ is the prime modulus.

4. **Recombination and Reordering:**
   After applying the appropriate butterfly operations across all decomposition levels, the Mixed Radix algorithm recombines the results into a single output sequence. Due to the varied sizes of the sub-transforms, a more complex reordering process may be required compared to Radix-2. This involves digit-reversal permutations to ensure that the final output sequence is correctly ordered.

### Which algorithm should I choose ?

Radix 2 is faster for small NTTs. A small NTT would be around logN = 16 and batch size 1. Its also more suited for inputs which are power of 2 (e.g., 256, 512, 1024). Radix 2 won't necessarily perform better for smaller `logn` with larger batches.

Mixed radix on the other hand better for larger NTTs with larger input sizes which are not necessarily power of 2.

Performance really depends on logn size, batch size, ordering, inverse, coset, coeff-field and which GPU you are using.

For this reason we implemented our [heuristic auto-selection](https://github.com/ingonyama-zk/icicle/blob/774250926c00ffe84548bc7dd97aea5227afed7e/icicle/appUtils/ntt/ntt.cu#L474) which should choose the most efficient algorithm in most cases. 

We still recommend you benchmark for your specific use case if you think a different configuration would yield better results.
