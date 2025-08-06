#include "utils.h"

PolyRing zero()
{
  PolyRing z;
  for (size_t i = 0; i < PolyRing::d; i++) {
    z.values[i] = Zq::zero();
  }
  return z;
}
