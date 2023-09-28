#pragma once
#ifndef NTT_H
#define NTT_H

/**
 * @namespace ntt
 * Number Theoretic Transform, or NTT is a version of [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) where instead of real or 
 * complex numbers, inputs and outputs belong to certain finite groups or fields. NTT computes the values of a polynomial 
 * \f$ p(x) = p_0 + p_1 \cdot x + \dots + p_{n-1} \cdot x^{n-1} \f$ on special subfields called "roots of unity", or "twiddle factors":
 * \f[
 *  NTT(p) = \{ p(\omega^0), p(\omega^1), \dots, p(\omega^{n-1}) \}
 * \f]
 * Inverse NTT, or iNTT solves the inverse problem of computing coefficients of \f$ p(x) \f$ from evaluations 
 * \f$ \{ p(\omega^0), p(\omega^1), \dots, p(\omega^{n-1}) \} \f$. If not specified otherwise, \f$ n \f$ is a power of 2.
 */
namespace ntt {

/**
 * Generates twiddles \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$ from root of unity \f$ \omega \f$ and stores them on device.
 * @param d_twiddles Input empty array on device to which twiddles are to be written.
 * @param n_twiddles Number of twiddle \f$ n \f$ factors to generate.
 * @param omega Root of unity \f$ \omega \f$.
 * @param stream Stream to use.
 * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$.
 */
template <typename S>
void generate_twiddle_factors(S* d_twiddles, uint32_t n_twiddles, S omega, cudaStream_t stream);

/**
 * @enum Ordering
 * How to order inputs and outputs of the NTT:
 * - kNN: inputs and outputs are natural-order (example of natural ordering: \f$ \{a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7\} \f$).
 * - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: \f$ \{a_0, a_4, a_2, a_6, a_1, a_5, a_3, a_7\} \f$).
 * - kRN: inputs are bit-reversed-order and outputs are natural-order.
 * - kRR: inputs and outputs are bit-reversed-order.
 */
enum class Ordering { kNN, kNR, kRN, kRR };

/**
 * @enum Decimation
 * Decimation of the NTT algorithm:
 * - kDIT: decimation in time.
 * - kDIF: decimation in frequency.
 */
enum class Decimation { kDIT, kDIF };

/**
 * @enum Butterfly
 * [Butterfly](https://en.wikipedia.org/wiki/Butterfly_diagram) used in the NTT algorithm (i.e. what happens to each pair of inputs on every iteration):
 * - kCooleyTukey: Cooley-Tukey butterfly.
 * - kGentlemanSande: Gentleman-Sande butterfly.
 */
enum class Butterfly { kCooleyTukey, kGentlemanSande };

/**
 * @struct NTTConfig
 * Struct that encodes NTT parameters to be passed into the [ntt_internal](@ref ntt_internal) function.
 */
template <typename S>
struct NTTConfig {
    bool are_inputs_on_device;          /**< True if inputs/outputs are on device and false if they're on host. Default value: false. */
    Ordering ordering;                  /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value: `Ordering::kNN`. */
    Decimation decimation;              /**< Decimation of the algorithm, see [Decimation](@ref Decimation). Default value: `Decimation::kDIT`.
                                         *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
                                         *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to 
                                         *   `Decimation::kDIT` and if ordering is `Ordering::kNR` — to `Decimation::kDIF`. */
    Butterfly butterfly;                /**< Butterfly used by the NTT. See [Butterfly](@ref Butterfly). Default value: `Butterfly::kCooleyTukey`.
                                         *   __Note:__ this variable exists mainly for compatibility with codebases that use similar notation.
                                         *   If [ordering](@ref ordering) is `Ordering::kRN`, the value of this variable will be overridden to 
                                         *   `Butterfly::kCooleyTukey` and if ordering is `Ordering::kNR` — to `Butterfly::kGentlemanSande`. */
    bool is_coset;                      /**< If false, NTT is computed on a subfield given by [twiddles](@ref twiddles). If true, NTT is computed
                                         *   on a coset of [twiddles](@ref twiddles) given by [the coset generator](@ref coset_gen), so: 
                                         *   \f$ \{coset\_gen\cdot\omega^0, coset\_gen\cdot\omega^1, \dots, coset\_gen\cdot\omega^{n-1}\} \f$. Default value: false. */
    S* coset_gen;                       /**< The field element that generates a coset if [is_coset](@ref is_coset) is true. 
                                         *   Otherwise should be set to null. Default value: `null`. */
    S* twiddles;                        /**< "Twiddle factors", (or "domain", or "roots of unity") on which the NTT is evaluated. 
                                         *   This pointer is expected to live on device. The order is as follows:
                                         *   \f$ \{\omega^0=1, \omega^1, \dots, \omega^{n-1}\} \f$. If this pointer is `null`, twiddle factors
                                         *   are generated online using the default generator (TODO: link to twiddle gen here) and function
                                         *   [generate_twiddle_factors](@ref generate_twiddle_factors). Default value: `null`. */
    unsigned batch_size;                /**< The number of NTTs to compute. Default value: 1. */
    unsigned device_id;                 /**< Index of the GPU to run the NTT on. Default value: 0. */
    cudaStream_t stream;                /**< Stream to use. Default value: 0. */
};

/**
 * A function that computes NTT or iNTT in-place.
 * @param input Input that's mutated in-place by this function. Length of this array needs to be [size](@ref size) * [config.batch_size](@ref config.batch_size).
 * @param size NTT size \f$ n \f$. If a batch of NTTs (which all need to have the same size) is computed, this is the size of 1 NTT.
 * @param is_inverse If true, inverse NTT is computed, otherwise — regular forward NTT.
 * @param config [NTTConfig](@ref NTTConfig) used in this NTT.
 * @tparam E The type of inputs and outputs (i.e. coefficients \f$ \{p_i\} \f$ and values \f$ p(x) \f$). Must be a group.
 * @tparam S The type of "twiddle factors" \f$ \{ \omega^i \} \f$. Must be a field. Often (but not always) `S=E`. 
 */
template <typename E, typename S>
void ntt_internal(E* input, unsigned size, bool is_inverse, NTTConfig<S> config);

/**
 * A function that computes NTT by calling [ntt_internal](@ref ntt_internal) function with default [NTTConfig](@ref NTTConfig) values.
 */
template <typename E, typename S>
void ntt(E* input, unsigned size, bool is_inverse);

} // namespace ntt

#endif
