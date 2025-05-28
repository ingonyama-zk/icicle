#pragma once

#include "icicle/hash/hash.h"

namespace icicle {
  /**
   * @brief Creates a Skyscraper hash instance with the specified parameters.
   *
   * This function generates a Skyscraper hash with customizable parameters to suit various cryptographic
   * contexts and use cases. The parameters should match those required by the Skyscraper construction.
   *
   * @param n Extension degree for the field (GF(p^n)).
   * @param beta Modulus for extension field x^n + beta.
   * @param s Decomposition size (bits).
   * @return Hash An instance of the Skyscraper hash initialized with the specified parameters, ready
   *         for hashing operations.
   */
  Hash create_skyscraper_hash(unsigned n = 1, unsigned beta = 5, unsigned s = 8);

  struct Skyscraper {
    inline static Hash create(unsigned n = 1, unsigned beta = 5, unsigned s = 8)
    {
      return create_skyscraper_hash(n, beta, s);
    }
  };

} // namespace icicle 