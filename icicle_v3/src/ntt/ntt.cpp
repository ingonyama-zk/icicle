#include "ntt/ntt.h"
#include "dispatcher.h"

using namespace icicle;

ICICLE_DISPATCHER_INST(NttDispatcher, ntt, NttImpl);

extern "C" eIcicleError
CONCAT_EXPAND(FIELD, ntt)(const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
{
  return NttDispatcher::execute(input, size, dir, config, output);
}