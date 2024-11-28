#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

extern "C" void CONCAT_EXPAND(FIELD, generate_scalars)(scalar_t* scalars, int size)
{
  scalar_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(FIELD, sub)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void CONCAT_EXPAND(FIELD, add)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void CONCAT_EXPAND(FIELD, mul)(scalar_t* scalar1, scalar_t* scalar2, scalar_t* result)
{
  *result = *scalar1 * *scalar2;
}

#ifdef EXT_FIELD
extern "C" void CONCAT_EXPAND(FIELD, extension_generate_scalars)(extension_t* scalars, int size)
{
  extension_t::rand_host_many(scalars, size);
}

extern "C" void CONCAT_EXPAND(FIELD, sub)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 - *scalar2;
}

extern "C" void CONCAT_EXPAND(FIELD, add)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 + *scalar2;
}

extern "C" void CONCAT_EXPAND(FIELD, mul)(extension_t* scalar1, extension_t* scalar2, extension_t* result)
{
  *result = *scalar1 * *scalar2;
}
#endif // EXT_FIELD
