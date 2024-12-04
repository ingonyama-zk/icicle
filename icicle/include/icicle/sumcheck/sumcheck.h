#pragma once

#include "icicle/sumcheck/sumcheck_transcript_config.h"

namespace icicle {

  template <typename F>
  struct Sumcheck {
  public:
    SumcheckTranscriptConfig<F> config;
  };

} // namespace icicle