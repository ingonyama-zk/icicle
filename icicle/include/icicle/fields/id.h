#pragma once

#define BN254     1
#define BLS12_381 2
#define BLS12_377 3
#define BW6_761   4
#define GRUMPKIN  5

#define BABY_BEAR  1001
#define STARK_252  1002
#define M31        1003
#define KOALA_BEAR 1004

// Note: rings IDs are included here since most code is shared with fields.
// In the future we may refactor the code to separate fields and rings.
#include "icicle/rings/id.h"
