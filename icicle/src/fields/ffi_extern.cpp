#include "icicle/utils/utils.h"
#include "icicle/fields/field_config.h"
#include "icicle/fields/externs.h"

using namespace field_config;

ICICLE_DEFINE_FIELD_FFI_FUNCS(, scalar_t);

#ifdef EXT_FIELD
ICICLE_DEFINE_FIELD_FFI_FUNCS(_extension, extension_t);
#endif // EXT_FIELD

#ifdef RING
ICICLE_DEFINE_FIELD_FFI_FUNCS(_rns, scalar_rns_t);
#endif // RING