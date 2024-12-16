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
     *
     * @param claimed_sum The total sum of the values of a multivariate polynomial f(x₁, x₂, ..., xₖ)
     * when evaluated over all possible Boolean input combinations
     * @param transcript_config Configuration for encoding and hashing prover messages.
     */
    SumcheckBackend(const F& claimed_sum, SumcheckTranscriptConfig<F>&& transcript_config)
        : m_claimed_sum(claimed_sum), m_transcript_config(std::move(transcript_config))
    {
    }

    virtual ~SumcheckBackend() = default;

    /**
     * @brief Calculate the sumcheck based on the inputs and retrieve the Sumcheck proof.
     * @param input_polynomials a vector of MLE polynomials to process
     * @param combine_function a program that define how to fold all MLS polynomials into the round polynomial.
     * @param config Configuration for the Sumcheck operation.
     * @param sumcheck_proof Reference to the SumCheckProof object where all round polynomials will be stored.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_proof(
      const std::vector<std::vector<F>*>& input_polynomials,
      const CombineFunction<F>& combine_function,
      SumCheckConfig& config,
      SumCheckProof<F>& sumcheck_proof /*out*/) const = 0;

    /**
     * @brief Calculate alpha based on m_transcript_config and the round polynomial.
     * @param round_polynomial a vector of MLE polynomials evaluated at x=0,1,2...
     * @return alpha
     */
    virtual F get_alpha(std::vector<F>& round_polynomial) = 0;

    const F& get_claimed_sum() const { return m_claimed_sum; }

  protected:
    const F m_claimed_sum;                                 ///< Vector of hash functions for each layer.
    const SumcheckTranscriptConfig<F> m_transcript_config; ///< Size of each leaf element in bytes.
  };

  /*************************** Backend Factory Registration ***************************/
  template <typename F>
  using SumcheckFactoryImpl = std::function<eIcicleError(
    const Device& device,
    const F& claimed_sum,
    SumcheckTranscriptConfig<F>&& transcript_config,
    std::shared_ptr<SumcheckBackend<F>>& backend /*OUT*/)>;

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