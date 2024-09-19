#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  // Options for generating the on-device Poseidon constants.
  // This struct will hold parameters needed to initialize Poseidon constants with custom settings.
  // The fields will include:
  // - `arity`: The arity (branching factor) of the Poseidon hash.
  // - `alpha`: The exponent used in the S-box function.
  // - `nof_rounds`: The number of rounds (both full and partial) for the Poseidon hash.
  // - `mds_matrix`: The Maximum Distance Separable (MDS) matrix used for mixing the state.
  // The struct should be FFI (Foreign Function Interface) compatible, meaning it should use basic types or pointers
  // that can easily be shared between languages. The template parameter `S` represents the field type for which the
  // Poseidon constants are being initialized.
  template <typename S>
  struct PoseidonConstantsInitOptions {
    // TODO: Define the struct with fields such as arity, alpha, nof_rounds, mds_matrix, etc.
    // It must be compatible with FFI, so make sure to use only types like integers, arrays, and pointers.
  };

  // Function to generate and initialize Poseidon constants based on user-defined options.
  // This function allows the user to customize the initialization of Poseidon constants by providing their own
  // parameters. It is important to call this function per arity (branching factor) because the constants depend on the
  // arity. The template parameter `S` represents the field type (e.g., scalar field) for which the constants are being
  // initialized.
  template <typename S>
  eIcicleError poseidon_init_constants(const PoseidonConstantsInitOptions<S>* options);

  // Function to initialize Poseidon constants using default, precomputed values.
  // These constants are optimized and precomputed for the given field and arity.
  // The arity must be supported by the implementation (i.e., predefined sets of constants exist for the supported
  // arities). This function simplifies initialization when custom constants are not needed, and the user can rely on
  // default values.
  template <typename S>
  eIcicleError poseidon_init_default_constants();

  // Function to create a Poseidon hash object for a given arity.
  // This function returns a `Hash` object configured to use the Poseidon hash for the specified arity.
  // The arity controls the number of inputs the hash function can take (branching factor).
  template <typename S>
  Hash create_poseidon_hash(unsigned arity);

  // Poseidon struct providing a static interface to Poseidon-related operations.
  struct Poseidon {
    // Static method to create a Poseidon hash object.
    // This method provides a simple API for creating a Poseidon hash object, hiding the complexity of template
    // parameters from the user. It uses the specified `arity` to create the Poseidon hash.
    template <typename S>
    inline static Hash create(unsigned arity)
    {
      return create_poseidon_hash<S>(arity);
    }

    // Static method to initialize Poseidon constants based on user-defined options.
    // This method abstracts away the complexity of calling the `poseidon_init_constants` function directly,
    // providing a clean interface to initialize Poseidon constants.
    // The user provides a pointer to `PoseidonConstantsInitOptions` to customize the constants.
    template <typename S>
    inline static eIcicleError init_constants(const PoseidonConstantsInitOptions<S>* options)
    {
      return poseidon_init_constants<S>(options);
    }

    // Static method to initialize Poseidon constants with default values.
    // This provides a clean interface for initializing Poseidon with precomputed default constants for the given field
    // and arity. Useful when the user doesn't need to customize the constants and wants to use pre-optimized
    // parameters.
    template <typename S>
    inline static eIcicleError init_default_constants()
    {
      return poseidon_init_default_constants<S>();
    }
  };

} // namespace icicle