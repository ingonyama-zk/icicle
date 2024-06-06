
#include "icicle/ecntt.h"
#include "icicle/dispatcher.h"

namespace icicle {
  ICICLE_DISPATCHER_INST(ECNttExtFieldDispatcher, ecntt, ECNttFieldImpl);

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, ecntt)(
    const projective_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, projective_t* output)
  {
    return ECNttExtFieldDispatcher::execute(input, size, dir, config, output);
  }

  template <>
  eIcicleError ntt(const projective_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, projective_t* output)
  {
    return CONCAT_EXPAND(FIELD, ecntt)(input, size, dir, config, output);
  }
} // namespace icicle