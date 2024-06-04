#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

extern "C" void CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

#ifdef EXT_FIELD
extern "C" void CONCAT_EXPAND(FIELD, extension_generate_scalars)(extension_t* scalars, int size)
{
  extension_t::rand_host_many(scalars, size);
}
#endif // EXT_FIELD
