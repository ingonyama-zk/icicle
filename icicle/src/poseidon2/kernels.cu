#include "poseidon/poseidon.cuh"
#include "gpu-utils/modifiers.cuh"

namespace poseidon2 {
  template <typename S>
  __device__ void print_scalar(S element)
  {
    printf(
      "%lu, %lu, %lu, %lu\n",
      (unsigned long)element.limbs_storage.limbs[0] + (((unsigned long)element.limbs_storage.limbs[1]) << 32),
      (unsigned long)element.limbs_storage.limbs[2] + (((unsigned long)element.limbs_storage.limbs[3]) << 32),
      (unsigned long)element.limbs_storage.limbs[4] + (((unsigned long)element.limbs_storage.limbs[5]) << 32),
      (unsigned long)element.limbs_storage.limbs[6] + (((unsigned long)element.limbs_storage.limbs[7]) << 32));
  }

  template <typename S>
  __device__ void print_scalar_u32(S element)
  {
    printf("%d", element.limbs_storage.limbs[0]);
  }

  template <typename S>
  __device__ void print_scalar_u64(S element)
  {
    printf(
      "%lu", (unsigned long)element.limbs_storage.limbs[0] + ((unsigned long)element.limbs_storage.limbs[1] << 32));
  }

  template <typename S>
  __device__ void print_scalar_hex(S element)
  {
    printf(
      "0x%08x%08x%08x%08x%08x%08x%08x%08x", element.limbs_storage.limbs[0], element.limbs_storage.limbs[1],
      element.limbs_storage.limbs[2], element.limbs_storage.limbs[3], element.limbs_storage.limbs[4],
      element.limbs_storage.limbs[5], element.limbs_storage.limbs[6], element.limbs_storage.limbs[7]);
  }

  template <typename S, int T>
  __device__ void print_state(S state[T])
  {
    for (int i = 0; i < T; i++) {
      print_scalar_u32(state[i]);
      printf(", ");
    }
    printf("\n");
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
  DEVICE_INLINE S add_rc(S state[T], size_t rc_offset, const S* rc)
  {
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = state[i] + rc[rc_offset + i];
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
        case MdsType::DEFAULT_MDS:
          mds_light_4x4(&state[i]);
          break;
        case MdsType::PLONKY:
          mds_light_plonky_4x4(&state[i]);
        }
      }
      // printf("First matmul\n");
      // print_state<S, T>(state);

      S sums[4] = {state[0], state[1], state[2], state[3]};
      UNROLL
      for (int i = 4; i < T; i += 4) {
        sums[0] = sums[0] + state[i];
        sums[1] = sums[1] + state[i + 1];
        sums[2] = sums[2] + state[i + 2];
        sums[3] = sums[3] + state[i + 3];
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

    S sum;
    switch (T) {
    case 2:
      // [2, 1]
      // [1, 3]
      sum = sum + element + state[1];
      state[0] = element + sum;
      state[1] = S::template mul_unsigned<2>(state[1]) + sum;
      break;
    case 3:
      // [2, 1, 1]
      // [1, 2, 1]
      // [1, 1, 3]
      sum = sum + element + state[1] + state[2];
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
      UNROLL
      for (int i = 1; i < T; i++) {
        wide_sum = wide_sum + S::Wide::from_field(state[i]);
      }
      sum = S::reduce(wide_sum);
      switch (constants.diffusion) {
      case DiffusionStrategy::DEFAULT_DIFFUSION:
        state[0] = element * constants.internal_matrix_diag[0] + sum;
        UNROLL
        for (int i = 1; i < T; i++) {
          state[i] = state[i] * constants.internal_matrix_diag[i] + sum;
        }
        break;
      case DiffusionStrategy::MONTGOMERY:
        state[0] = S::from_montgomery(element * constants.internal_matrix_diag[0] + sum);
        UNROLL
        for (int i = 1; i < T; i++) {
          state[i] = S::from_montgomery(state[i] * constants.internal_matrix_diag[i] + sum);
        }
        // print_state<S, T>(state);
        break;
        // typename S::Wide part_sum = S::Wide::from_field(S::zero());
        // UNROLL
        // for (int i = 1; i < T; i++) {
        //   part_sum = part_sum + S::Wide::from_field(S::to_montgomery(state[i]));
        // }
        // typename S::Wide state0_neg = S::Wide::from_field(S::neg(S::to_montgomery(element)));
        // printf("element = ");
        // print_scalar_u32(element);
        // printf("; full sum = ");
        // print_scalar_u64(wide_sum);
        // printf("; -state[0] = ");
        // print_scalar_u64(state0_neg);
        // printf("; state[0] = ");
        // print_scalar_u64(part_sum + state0_neg);
        // printf("; state[0]_reduce = ");
        // print_scalar_u32(S::reduce(part_sum + state0_neg));
        // printf("; state[0]_from_mont = ");
        // print_scalar_u32(S::from_montgomery(S::reduce(part_sum + state0_neg)));
        // printf("\n");

        // state[0] = S::reduce(wide_sum + state0_neg + state0_neg);
        // UNROLL
        // for (int i = 1; i < T; i++) {
        //   S si = sum +
        //   state[i] = S::from_montgomery(state[i] * constants.internal_matrix_diag[i] + sum);
        // }
        break;
      }
    }
  }

  template <typename S, int T>
  __global__ void
  poseidon2_permutation_kernel(S* states, S* states_out, size_t number_of_states, const Poseidon2Constants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T];
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }
    unsigned int rn;

    mds_light<S, T>(state, constants.mds_type);

    printf("LinearLayer\n");
    print_state<S, T>(state);

    // printf("RC\n");
    // for (int i = 0; i < 10; i++) {
    //   print_scalar_hex(constants.round_constants[i]);
    // }

    size_t rc_offset = 0;
    // External rounds
    for (rn = 0; rn < constants.external_rounds / 2; rn++) {
      // printf("External Round %d\n", rn);
      // print_state<S, T>(state);
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      // printf("External Round rc %d\n", rn);
      // print_state<S, T>(state);
      sbox<S, T>(state, constants.alpha);
      // printf("External Round sbox %d\n", rn);
      // print_state<S, T>(state);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }
    printf("External Rounds\n");
    print_state<S, T>(state);

    // Internal rounds
    for (; rn < constants.external_rounds / 2 + constants.internal_rounds; rn++) {
      // printf("\n");
      // printf("Internal Round %d\n", rn);
      // print_state<S, T>(state);
      // printf("\n");
      internal_round<S, T>(state, rc_offset, constants);
      rc_offset++;
    }
    printf("Internal Rounds\n");
    print_state<S, T>(state);

    // External rounds
    for (; rn < constants.external_rounds + constants.internal_rounds; rn++) {
      add_rc<S, T>(state, rc_offset, constants.round_constants);
      sbox<S, T>(state, constants.alpha);
      mds_light<S, T>(state, constants.mds_type);
      rc_offset += T;
    }
    printf("External Rounds\n");
    print_state<S, T>(state);

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