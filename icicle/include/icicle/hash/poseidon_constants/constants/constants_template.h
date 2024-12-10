#pragma once
#ifndef CURVE_POSEIDON_H
  #define CURVE_POSEIDON_H

namespace poseidon_constants_curve {
  /**
   * This inner namespace contains optimized constants for running Poseidon.
   * These constants were generated using an algorithm defined at
   * https://spec.filecoin.io/algorithms/crypto/poseidon/
   * The number in the name corresponds to the arity of hash function
   * Each array contains:
   * RoundConstants | MDSMatrix | Non-sparse matrix | Sparse matrices
   */

  int partial_rounds_3 = 0;

  int partial_rounds_5 = 0;

  int partial_rounds_9 = 0;

  int partial_rounds_12 = 0;

  unsigned char poseidon_constants_3[] = {0x00};

  unsigned char poseidon_constants_5[] = {0x00};

  unsigned char poseidon_constants_9[] = {0x00};

  unsigned char poseidon_constants_12[] = {0x00};
} // namespace poseidon_constants_curve
#endif