#include "poseidon/poseidon.cuh"
#include "gpu-utils/modifiers.cuh"

namespace poseidon2 {

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
      S result8 = S::sqr(S::sqr(result2));
      return result8 * result2 * element;
    }
  }

  template <typename S, int T>
  DEVICE_INLINE S sbox(S state[T], const int alpha)
  {
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = sbox_el(state[i], alpha);
    }
  }

  template <typename S, int T>
  DEVICE_INLINE S add_rc(S state[T], const unsigned int rn, const S* rc)
  {
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = state[i] + rc[rn * T + i];
    }
  }

  template <typename S>
  __device__ S mds_light_4x4(S s[4])
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
  __device__ S mds_light_plonky_4x4(S s[4])
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
  __device__ S mds_light(S state[T], MdsType mds)
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
      UNROLL
      for (int i = 0; i < T; i += 4) {
        switch (mds) {
        case MdsType::DEFAULT:
          mds_light_4x4(&state[i]);
          break;
        case MdsType::PLONKY:
          mds_light_plonky_4x4(&state[i]);
        }
      }

      S sums[4] = {state[0], state[1], state[2], state[3]};
      UNROLL
      for (int i = 4; i < T; i += 4) {
        sums[i] = sums[i] + state[i];
        sums[i + 1] = sums[i + 1] + state[i + 1];
        sums[i + 2] = sums[i + 2] + state[i + 2];
        sums[i + 3] = sums[i + 3] + state[i + 3];
      }

      UNROLL
      for (int i = 0; i < T; i++) {
        state[i] = state[i] + sums[i % 4];
      }
      break;
    }
  }

  template <typename S, int T>
  __device__ S internal_round(S state[T], size_t rc_offset, const Poseidon2Constants<S>& constants)
  {
    S element = state[0];
    element = element + constants.round_constants[rc_offset];
    element = sbox_el<S>(element, constants.alpha);

    S sum = element;
    switch (T) {
    case 2:
      // [2, 1]
      // [1, 3]
      sum = sum + state[1];
      state[0] = element + sum;
      state[1] = state[1] * state[1] + sum;
      break;
    case 3:
      // [2, 1, 1]
      // [1, 2, 1]
      // [1, 1, 3]
      sum = state[1] + state[2];
      state[0] = element + sum;
      state[1] = state[1] + sum;
      state[2] = state[2] * state[2];
      state[2] = state[2] + sum;
      break;
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
      UNROLL
      for (int i = 1; i < T; i++) {
        sum = sum + state[i];
      }

      state[0] = element * constants.internal_matrix_diag[0] + sum;
      UNROLL
      for (int i = 1; i < T; i++) {
        state[i] = state[i] * constants.internal_matrix_diag[i] + sum;
      }
      break;
    }
  }

  template <typename S, int T>
  __global__ void poseidon2_permutation_kernel(
    S* states,
    S* states_out,
    size_t number_of_states,
    const Poseidon2Constants<S> constants,
    const Poseidon2Config config)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T];
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }
    unsigned int rn;

    mds_light<S, T>(state, config.mds_type);

    // External rounds
    for (rn = 0; rn < constants.external_rounds / 2; rn++) {
      add_rc<S, T>(state, rn, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, config.mds_type);
    }

    // Internal rounds
    size_t rc_offset = rn * T;
    for (; rn < constants.external_rounds / 2 + constants.internal_rounds; rn++) {
      internal_round<S, T>(state, rc_offset, constants);
      rc_offset++;
    }

    // External rounds
    for (; rn < constants.external_rounds + constants.internal_rounds; rn++) {
      add_rc<S, T>(state, rn, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, config.mds_type);
    }

    UNROLL
    for (int i = 0; i < T; i++) {
      states_out[idx * T + i] = state[i];
    }
  }

  // These function is just doing copy from the states to the output
  template <typename S, int T>
  __global__ void get_hash_results(S* states, size_t number_of_states, int index, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    out[idx] = states[idx * T + index];
  }

  template <typename S, int T>
  __global__ void copy_recursive(S* state, size_t number_of_states, int index, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    state[(idx / (T - 1) * T) + (idx % (T - 1)) + 1] = out[idx];
  }
} // namespace poseidon2