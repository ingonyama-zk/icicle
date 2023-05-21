#ifndef _BLS12_381_POSEIDON
#define _BLS12_381_POSEIDON
#include <cuda.h>
#include "../../appUtils/poseidon/poseidon.cuh"
#include "curve_config.cuh"

template class Poseidon<BLS12_381::scalar_t>;
#endif