#pragma once
#ifndef POSEIDON2_KERNELS_H
#define POSEIDON2_KERNELS_H

#include "utils/utils.h"
#include "hash/hash.cuh"
#include "matrix/matrix.cuh"
#include "poseidon2/constants.cuh"
#include "gpu-utils/modifiers.cuh"

using matrix::Matrix;

namespace poseidon2 {
  DEVICE_INLINE unsigned int d_next_pow_of_two(unsigned int v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

  template <typename S>
  DEVICE_INLINE S sbox_el(S element, const int alpha)
  {
    S result2 = S::sqr(element);
    switch (alpha) {
    case 3:
      return result2 * element;
    case 5:
      return S::sqr(result2) * element;
    case 7:
      return S::sqr(result2) * result2 * element;
    case 11:
      return S::sqr(S::sqr(result2)) * result2 * element;
    }
  }

  template <typename S, int T>
  DEVICE_INLINE void sbox(S state[T], const int alpha)
  {
    for (int i = 0; i < T; i++) {
      state[i] = sbox_el(state[i], alpha);
    }
  }

  template <typename S, int T>
  DEVICE_INLINE void add_rc(S state[T], size_t rc_offset, const S* rc)
  {
    for (int i = 0; i < T; i++) {
      state[i] = state[i] + rc[rc_offset + i];
    }
  }

  template <typename S>
  __device__ void mds_light_4x4(S s[4])
  {
    S t0 = s[0] + s[1];
    S t1 = s[2] + s[3];
    S t2 = s[1] + s[1] + t1;
    S t3 = s[3] + s[3] + t0;
    S t4 = S::template mul_unsigned<4>(t1) + t3;
    S t5 = S::template mul_unsigned<4>(t0) + t2;
    s[0] = t3 + t5;
    s[1] = t5;
    s[2] = t2 + t4;
    s[3] = t4;
  }

  // Multiply a 4-element vector x by:
  // [ 2 3 1 1 ]
  // [ 1 2 3 1 ]
  // [ 1 1 2 3 ]
  // [ 3 1 1 2 ].
  // https://github.com/Plonky3/Plonky3/blob/main/poseidon2/src/matrix.rs#L36
  template <typename S>
  __device__ void mds_light_plonky_4x4(S s[4])
  {
    S t01 = s[0] + s[1];
    S t23 = s[2] + s[3];
    S t0123 = t01 + t23;
    S t01123 = t0123 + s[1];
    S t01233 = t0123 + s[3];
    s[3] = t01233 + S::template mul_unsigned<2>(s[0]);
    s[1] = t01123 + S::template mul_unsigned<2>(s[2]);
    s[0] = t01123 + t01;
    s[2] = t01233 + t23;
  }

  template <typename S, int T>
  __device__ void mds_light(S state[T], MdsType mds)
  {
    S sum;
    switch (T) {
    case 2:
      // Matrix circ(2, 1)
      // [2, 1]
      // [1, 2]
      sum = state[0] + state[1];
      state[0] = state[0] + sum;
      state[1] = state[1] + sum;
      break;
    case 3:
      // Matrix circ(2, 1, 1)
      // [2, 1, 1]
      // [1, 2, 1]
      // [1, 1, 2]
      sum = state[0] + state[1] + state[2];
      state[0] = state[0] + sum;
      state[1] = state[1] + sum;
      state[2] = state[2] + sum;
      break;
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
      for (int i = 0; i < T; i += 4) {
        switch (mds) {
        case MdsType::DEFAULT_MDS:
          mds_light_4x4(&state[i]);
          break;
        case MdsType::PLONKY:
          mds_light_plonky_4x4(&state[i]);
        }
      }

      S sums[4] = {state[0], state[1], state[2], state[3]};
      for (int i = 4; i < T; i += 4) {
        sums[0] = sums[0] + state[i];
        sums[1] = sums[1] + state[i + 1];
        sums[2] = sums[2] + state[i + 2];
        sums[3] = sums[3] + state[i + 3];
      }
      for (int i = 0; i < T; i++) {
        state[i] = state[i] + sums[i % 4];
      }
      break;
    }
  }

  template <typename S, int T>
  __device__ void internal_round(S state[T], size_t rc_offset, const Poseidon2Constants<S>& constants)
  {
    S element = state[0];
    element = element + constants.round_constants[rc_offset];
    element = sbox_el<S>(element, constants.alpha);

    S sum;
    switch (T) {
    case 2:
      // [2, 1]
      // [1, 3]
      sum = element + state[1];
      state[0] = element + sum;
      state[1] = S::template mul_unsigned<2>(state[1]) + sum;
      break;
    case 3:
      // [2, 1, 1]
      // [1, 2, 1]
      // [1, 1, 3]
      sum = element + state[1] + state[2];
      state[0] = element + sum;
      state[1] = state[1] + sum;
      state[2] = S::template mul_unsigned<2>(state[2]) + sum;
      break;
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
      typename S::Wide wide_sum = S::Wide::from_field(element);
      for (int i = 1; i < T; i++) {
        wide_sum = wide_sum + S::Wide::from_field(state[i]);
      }
      sum = S::reduce(wide_sum);
      switch (constants.diffusion) {
      case DiffusionStrategy::DEFAULT_DIFFUSION:
        state[0] = element * constants.internal_matrix_diag[0] + sum;
        for (int i = 1; i < T; i++) {
          state[i] = state[i] * constants.internal_matrix_diag[i] + sum;
        }
        break;
      case DiffusionStrategy::MONTGOMERY:
        state[0] = S::from_montgomery(element * constants.internal_matrix_diag[0] + sum);
        for (int i = 1; i < T; i++) {
          state[i] = S::from_montgomery(state[i] * constants.internal_matrix_diag[i] + sum);
        }
        break;
      }
    }
  }

  template <typename S, int T>
  __device__ void permute_state(S state[T], const Poseidon2Constants<S>& constants)
  {
    unsigned int rn;

    mds_light<S, T>(state, constants.mds_type);

    size_t rc_offset = 0;
    // External rounds
    for (rn = 0; rn < constants.external_rounds / 2; rn++) {
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }

    // Internal rounds
    for (; rn < constants.external_rounds / 2 + constants.internal_rounds; rn++) {
      internal_round<S, T>(state, rc_offset, constants);
      rc_offset++;
    }

    // External rounds
    for (; rn < constants.external_rounds + constants.internal_rounds; rn++) {
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }
  }

  template <typename S, int T>
  __global__ void permutation_kernel(
    const S* states, S* states_out, unsigned int number_of_states, const Poseidon2Constants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T];
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }

    permute_state<S, T>(state, constants);

    UNROLL
    for (int i = 0; i < T; i++) {
      states_out[idx * T + i] = state[i];
    }
  }

  template <typename S>
  __device__ void print_scalar_u32(S element)
  {
    printf("%d", element.limbs_storage.limbs[0]);
  }

  template <typename S, int T>
  __device__ void print_state(S state[T])
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf(
      "IDX: %d. State: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", idx, state[0], state[1],
      state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11], state[12],
      state[13], state[14], state[15]);
  }

  template <typename S, int T>
  __device__ void absorb_2d_state(
    const Matrix<S>* inputs,
    S state[T],
    unsigned int number_of_inputs,
    unsigned int rate,
    unsigned int row_idx,
    const Poseidon2Constants<S>& constants)
  {
    unsigned int index = 0;
    for (int i = 0; i < number_of_inputs; i++) {
      const Matrix<S> input = inputs[i];
      unsigned int row_len = input.width;
      for (int j = 0; j < row_len; j++) {
        state[index] = input.values[row_idx * row_len + j];
        index++;
        if (index == rate) {
          // print_state<S, T>(state);
          permute_state<S, T>(state, constants);

          index = 0;
        }
      }
    }

    if (index) { permute_state<S, T>(state, constants); }
  }

  template <typename S, int T>
  __global__ void absorb_2d_kernel(
    const Matrix<S>* inputs,
    S* states,
    unsigned int number_of_inputs,
    unsigned int rate,
    const Poseidon2Constants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= inputs[0].height) { return; }

    S state[T] = {0};

    absorb_2d_state<S, T>(inputs, state, number_of_inputs, rate, idx, constants);

    UNROLL
    for (int i = 0; i < T; i++) {
      states[idx * T + i] = state[i];
    }
  }

  template <typename S, int T>
  __global__ void compress_and_inject_kernel(
    const Matrix<S>* matrices_to_inject,
    unsigned int number_of_inputs,
    const S* prev_layer,
    S* next_layer,
    unsigned int rate,
    unsigned int digest_elements,
    const Poseidon2Constants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t number_of_rows = d_next_pow_of_two(matrices_to_inject[0].height);
    if (idx >= number_of_rows) { return; }

    size_t next_layer_len = matrices_to_inject[0].height;
    S state_to_compress[T] = {S::zero()};

    for (int i = 0; i < digest_elements * 2; i++) {
      state_to_compress[i] = prev_layer[idx * 2 * digest_elements + i];
    }
    permute_state<S, T>(state_to_compress, constants);

    S injected_state[T] = {S::zero()};
    if (idx < next_layer_len) {
      absorb_2d_state<S, T>(matrices_to_inject, injected_state, number_of_inputs, rate, idx, constants);

      for (int i = 0; i < digest_elements; i++) {
        injected_state[digest_elements + i] = injected_state[i];
        injected_state[i] = state_to_compress[i];
      }
    } else {
      for (int i = 0; i < digest_elements; i++) {
        injected_state[i] = state_to_compress[i];
      }
    }
    // print_state<S, T>(injected_state);
    permute_state<S, T>(injected_state, constants);
    // print_state<S, T>(injected_state);

    for (int i = 0; i < digest_elements; i++) {
      next_layer[idx * digest_elements + i] = injected_state[i];
    }
  }
} // namespace poseidon2

#endif