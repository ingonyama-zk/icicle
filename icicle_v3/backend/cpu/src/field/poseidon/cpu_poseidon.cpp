#include <iostream>
#include "icicle/hash.h"
#include "icicle/errors.h"

// #include "../../include/icicle/hash.h"
using namespace icicle;

template <typename S>
class Poseidon : public Hash {
 public:
  Poseidon(int element_nof_limbs, int input_nof_elements, int output_nof_elements)
      : Hash(element_nof_limbs, input_nof_elements, output_nof_elements) {
        PoseidonConstants<S> constants;
      }

  virtual eIcicleError hash_many(const limb_t *input_limbs, limb_t *output_limbs, unsigned int batch) const override {
    // This is just a placeholder, copy first element of each hash group.
    int input_pointer = 0;
    int output_pointer = 0;
    int output_size = this->element_nof_limbs * this->output_nof_elements;
    int input_size = this->element_nof_limbs * this->input_nof_elements;
    for (int batch_idx = 0; batch_idx<batch; batch_idx++)
    {
      std::copy(input_limbs + batch_idx*input_size, input_limbs + batch_idx*input_size + output_size, output_limbs + batch_idx*output_size);
    }
    return eIcicleError::SUCCESS;
  }
};





eIcicleError poseidon_cpu(const Device& device, Hash** hash, int element_nof_limbs, int input_nof_elements, int output_nof_elements)
{
    *hash = new Poseidon(element_nof_limbs, input_nof_elements, output_nof_elements);
    return eIcicleError::SUCCESS;
}

REGISTER_POSEIDON_BACKEND("CPU", poseidon_cpu);





#include "poseidon/poseidon.h"

namespace poseidon {
  // template <typename S, int T>
  // __global__ void prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, bool aligned)
  // {
  //   int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //   int state_number = idx / T;
  //   if (state_number >= number_of_states) { return; }
  //   int element_number = idx % T;

  //   S prepared_element;

  //   // Domain separation
  //   if (element_number == 0) {
  //     prepared_element = domain_tag;
  //   } else {
  //     if (aligned) {
  //       prepared_element = states[idx];
  //     } else {
  //       prepared_element = states[idx - 1];
  //     }
  //   }

  //   // We need __syncthreads here if the state is not aligned
  //   // because then we need to shift the vector [A, B, 0] -> [D, A, B]
  //   if (!aligned) { __syncthreads(); }

  //   // Store element in state
  //   states[idx] = prepared_element;
  // }


  /*********************************************** 
   
    * intrinsic functions:
  - sbox_alpha_five
  - matmul_inplace
  - matmul_sparse_inplace

  ************************************************/

  template <typename S>
  S sbox_alpha_five(S element)
  {
    S result = S::sqr(element);
    result = S::sqr(result);
    return result * element;
  }

  template <typename S, int T>
  void matmul_inplace(S *matrix , S *inout)
  /*
  matrix: 2D array of size T x T,
  inout: 1D array of type S and size T,
  */
  {
    S::Wide out_temp[T];
    S out[T];
    // TODO: this "wide add" optimization only works if field size reserved_bits > log2(T)
    // need to add an assertion
    for (int i = 0; i < T; i++) {
      out_temp[i] = S::mul_wide(in[0], matrix[i]);
      for (int j = 1; j < T; j++) {
        out_temp[i] = out_temp[i] + S::mul(in[j], matrix[j * T + i]);
      }
      out[i] = S::reduce(out_temp[i]);
    }
    // copy back to inout
    for (int i = 0; i < T; i++) {
      inout[i] = out[i];
    }
  }

  template <typename S, int T>
  void matmul_sparse_inplace(S *sparse_matrix , S *inout)
  /*
  sparse_matrix: 1D array of size 2*T - 1,
  given the 1D array of sparse matrix = [x1... x2T-1],
  the matrix interperted as:
  [[ x0, x1 ... xT-1],]
   [xT, 1, 0, ... 0],
   [xT+1, 0, 1, ... 0],
   ...
   [x2T-1, 0, 0, ... 1]]
  inout: 1D array of size T,
  */
  {
    S element = inout[0];
    // TODO: this wide optimization only works if field size reserved_bits > log2(T)
    // need to add an assertion
    typename S::Wide state_0_wide = S::mul_wide(in[0], sparse_matrix[0]);
    for (int i = 1; i < T; i++) {
      state_0_wide = state_0_wide + S::mul_wide(state[i], sparse_matrix[i]);
    }
    inout[0] = S::reduce(state_0_wide);
    for (int i = 1; i < T; i++) {
      inout[i] = inout[i] + (element * sparse_matrix[T + i - 1]);
    }
  }



/*********************************************** 
   
    * rounds functions:
  - full_round: single full round of poseidon
  - full_rounds: multiple full rounds of poseidon
  - partial_round: single partial round of poseidon
  - partial_rounds: multiple partial rounds of poseidon

  ************************************************/

  template <typename S, int T>
  void full_round(
    S *states,
    size_t rc_offset,
    bool multiply_by_mds,
    bool add_pre_round_constants,
    bool skip_rc,
    const PoseidonConstants<S>& constants)
  {
    // pre-round constants addition (states = states + rc)
    if (add_pre_round_constants) {
      for (int i = 0; i < T; i++) {
        states[i] = states[i] + constants.round_constants[rc_offset + i];
        rc_offset += 1;
      }
    }

    // s-box (states = states ^ 5)
    for (int i = 0; i < T; i++) {
        states[i] = sbox_alpha_five(states[i]);
    }

    // round constants (states = states + rc)
    if (!skip_rc){
      for (int i = 0; i < T; i++) {
          states[i] = states[i] + constants.round_constants[rc_offset];
          rc_offset += 1;
      }
    }

    // Multiply all the states by mds matrix (states = matrix * states)
    S* matrix = multiply_by_mds ? constants.mds_matrix : constants.non_sparse_matrix;
    matmul_inplace(matrix, states);
  }

  template <typename S, int T>
  void full_rounds(
    S* states, size_t rc_offset, bool first_half, const PoseidonConstants<S> constants)
  {
    bool add_pre_round_constants = first_half;
    for (int i = 0; i < constants.full_rounds_half; i++) {
      full_round<S, T>(
        states, rc_offset, 
        !first_half || (i < (constants.full_rounds_half - 1)), // multiply by mds, always except last full round of first half.
        add_pre_round_constants, 
        !first_half && (i == constants.full_rounds_half - 1), // skip rc, only on the last round of the second half.
        constants);
      if (add_pre_round_constants) {
        rc_offset += T; // pre round constants offset update
        add_pre_round_constants = false;
      }
      
    }
    
  }

  template <typename S, int T>
  partial_round(S state[T], size_t rc_offset, int round_number, const PoseidonConstants<S>& constants)
  {
    S element = state[0];

    // sbox
    element = sbox_alpha_five(element);

    // round constants
    element = element + constants.round_constants[rc_offset];
    
    // sparse mat mult
    S* sparse_matrix = &constants.sparse_matrices[(T * 2 - 1) * round_number];
    matmul_sparse_inplace(sparse_matrix, state);

  }

  template <typename S, int T>
  void partial_rounds(S* states, size_t rc_offset, const PoseidonConstants<S> constants)
  {
    for (int i = 0; i < constants.partial_rounds; i++) {
      partial_round<S, T>(states, rc_offset, i, constants);
      rc_offset++;
    }
  }


  /*********************************************** 
   
    // poseidon hash function

  ************************************************/

  // // These function is just doing copy from the states to the output
  // template <typename S, int T>
  // __global__ void get_hash_results(S* states, size_t number_of_states, S* out)
  // {
  //   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //   if (idx >= number_of_states) { return; }

  //   out[idx] = states[idx * T + 1];
  // }

  // template <typename S, int T>
  // __global__ void copy_recursive(S* state, size_t number_of_states, S* out)
  // {
  //   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //   if (idx >= number_of_states) { return; }

  //   state[(idx / (T - 1) * T) + (idx % (T - 1)) + 1] = out[idx];
  // }



template <typename S, int T>
cudaError_t poseidon_hash(
const S* inputs,
S* outputs,
const PoseidonConstants<S>& constants)
  {
    // prepare states
    outputs[0] = constants.domain_tag; // first state is domain tag
    for (int i = 0; i < T - 1; i++) {  // others is inputs
      outputs[i + 1] = inputs[i];
    }

    size_t rc_offset = 0;
    full_rounds<S, T>(outputs, rc_offset, true, constants);
    rc_offset += T * (constants.full_rounds_half + 1);

    partial_rounds<S, T>(outputs, rc_offset, constants);
    rc_offset += constants.partial_rounds;

    full_rounds<S, T>(outputs, rc_offset, false, constants);
  }

} // namespace poseidon


