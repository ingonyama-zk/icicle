
#include "icicle/ecntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "ntt.template"

#include "icicle/curves/curve_config.h"

using namespace field_config;
using namespace icicle;

REGISTER_ECNTT_BACKEND("CPU", (cpu_ntt<scalar_t, projective_t>));
