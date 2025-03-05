#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "icicle/program/returning_value_program.h"
#include "icicle/sumcheck/sumcheck_config.h"
#include "icicle/sumcheck/sumcheck_proof.h"
#include "icicle/sumcheck/sumcheck_transcript_config.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

template <typename F>
using CombineFunction = ReturningValueProgram<F>;

namespace icicle {
  /**
   * @brief Abstract base class for Sumcheck backend implementations.
   *
   * This backend handles the core logic for Sumcheck operations such as calculating the round polynomials
   * per round and building the Sumcheck proof for the verifier,
   * Derived classes will provide specific implementations for various devices (e.g., CPU, GPU).
   */
  template <typename F>
  class SumcheckBackend
  {
  public:
    /**
     * @brief Constructor for the SumcheckBackend class.
     */
    SumcheckBackend() {}

    virtual ~SumcheckBackend() = default;

    /**
     * @brief Calculate the sumcheck based on the inputs and retrieve the Sumcheck proof.
     * @param mle_polynomials a vector of MLE polynomials to process
     * @param mle_polynomial_size the size of each MLE polynomial
     * @param claimed_sum The total sum of the values of a multivariate polynomial f(x₁, x₂, ..., xₖ)
     * when evaluated over all possible Boolean input combinations
     * @param combine_function a program that define how to fold all MLS polynomials into the round polynomial.
     * @param transcript_config Configuration for encoding and hashing prover messages.
     * @param sumcheck_config Configuration for the Sumcheck operation.
     * @param sumcheck_proof Reference to the SumcheckProof object where all round polynomials will be stored.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_proof(
      const std::vector<F*>& mle_polynomials,
      const uint64_t mle_polynomial_size,
      const F& claimed_sum,
      const CombineFunction<F>& combine_function,
      SumcheckTranscriptConfig<F>&& transcript_config,
      const SumcheckConfig& sumcheck_config,
      SumcheckProof<F>& sumcheck_proof /*out*/) = 0;
  };

  /*************************** Backend Factory Registration ***************************/
  template <typename F>
  using SumcheckFactoryImpl =
    std::function<eIcicleError(const Device& device, std::shared_ptr<SumcheckBackend<F>>& backend /*OUT*/)>;

  /**
   * @brief Register a Sumcheck backend factory for a specific device type.
   *
   * @param deviceType String identifier for the device type.
   * @param impl Factory function that creates tSumcheckBackend.
   */
  void register_sumcheck_factory(const std::string& deviceType, SumcheckFactoryImpl<scalar_t> impl);

  /**
   * @brief Macro to register a Sumcheck backend factory.
   *
   * This macro registers a factory function for a specific backend by calling
   * `register_sumcheck_factory` at runtime.
   */
#define REGISTER_SUMCHECK_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_sumcheck) = []() -> bool {                                                                 \
      register_sumcheck_factory(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle