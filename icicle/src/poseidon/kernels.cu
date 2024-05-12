#include "../../include/poseidon/poseidon.cuh"
#include "../../include/gpu-utils/modifiers.cuh"

namespace poseidon {
  template <typename S, int T>
  void prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, bool aligned)
  {
    return;
  }

  template <typename S>
  S sbox_alpha_five(S element)
  {
    S result = S::sqr(element);
    result = S::sqr(result);
    return result * element;
  }

  template <typename S, int T>
  S vecs_mul_matrix(S element, S* matrix, int element_number, int vec_number, S* shared_states)
  {
    return element;
  }

  template <typename S, int T>
  S full_round(
    S element,
    size_t rc_offset,
    int local_state_number,
    int element_number,
    bool multiply_by_mds,
    bool add_pre_round_constants,
    bool skip_rc,
    S* shared_states,
    const PoseidonConstants<S>& constants)
  {
    return element;
  }

  template <typename S, int T>
    void full_rounds(
    S* states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConstants<S> constants)
  {
    return;
  }

  template <typename S, int T>
  S partial_round(S state[T], size_t rc_offset, int round_number, const PoseidonConstants<S>& constants)
  {
    S element = state[0];
    return element;
  }

  template <typename S, int T>
  void
  partial_rounds(S* states, size_t number_of_states, size_t rc_offset, const PoseidonConstants<S> constants)
  {
    return;
  }

  // These function is just doing copy from the states to the output
  template <typename S, int T>
  void get_hash_results(S* states, size_t number_of_states, S* out)
  {
    return;
  }

  template <typename S, int T>
  void copy_recursive(S* state, size_t number_of_states, S* out)
  {
    return;
  }
} // namespace poseidon