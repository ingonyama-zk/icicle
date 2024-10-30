#pragma once

#include "icicle/hash/hash.h"

namespace icicle {
  /**
   * @brief Creates a Poseidon hash instance with the specified width and optional domain tag.
   *
   * This function generates a Poseidon hash with customizable parameters to suit various cryptographic
   * contexts and use cases. The width parameter (`t`) determines the number of elements in the state,
   * influencing the security level and output structure of the hash. The optional domain tag enables
   * support for domain separation, allowing you to isolate hash outputs across different contexts
   * or applications.
   *
   * @tparam S The type representing the state or field element used by the hash (e.g., a field element class).
   *
   * @param t The width of the Poseidon hash state, representing the number of elements in the hash state.
   *          Typical values are 3, 5, 9, or 12, which correspond to different security levels and
   *          constraints. Ensure that the selected `t` is compatible with the Poseidon implementation
   *          and use case.
   *
   * @param use_domain_tag A boolean flag indicating whether to include a domain tag in the hash.
   *                       Setting this to `true` enables domain separation, which isolates hash outputs
   *                       across different contexts, making it particularly useful in applications where
   *                       the same input data may need to be hashed in different scenarios.
   *
   * @return Hash An instance of the Poseidon hash initialized with the specified parameters, ready
   *         for hashing operations.
   */
  template <typename S>
  Hash create_poseidon_hash(unsigned t, bool use_domain_tag);

  // Poseidon struct providing a static interface to Poseidon-related operations.
  struct Poseidon {
    template <typename S>
    inline static Hash create(unsigned t, bool use_domain_tag)
    {
      return create_poseidon_hash<S>(t, use_domain_tag);
    }
  };

} // namespace icicle