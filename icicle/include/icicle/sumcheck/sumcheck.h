#pragma once

#include "icicle/sumcheck/sumcheck_transcript_config.h"

namespace icicle {

  /**
   * @brief Sumcheck protocol implementation for a given field type.
   *
   * This class encapsulates the Sumcheck protocol, including its transcript configuration.
   *
   * @tparam F The field type used in the Sumcheck protocol.
   */
  template <typename F>
  class Sumcheck
  {
  public:
    /**
     * @brief Constructs a Sumcheck instance with the given transcript configuration.
     * @param transcript_config The configuration for the Sumcheck transcript.
     */
    explicit Sumcheck(SumcheckTranscriptConfig<F>&& transcript_config)
        : m_transcript_config(std::move(transcript_config))
    {
    }

    // Add public methods for protocol operations, e.g., prove, verify.

  private:
    SumcheckTranscriptConfig<F> m_transcript_config; ///< Transcript configuration for the protocol.
  };

} // namespace icicle