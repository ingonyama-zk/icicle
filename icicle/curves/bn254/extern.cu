#include "fields/bn254/extern.cu"

#include "bn254.cuh"
#define CURVE bn254
using namespace bn254;

#include "curves/mont.cu"
#include "curves/projective.cu"
#include "msm/extern.cu"