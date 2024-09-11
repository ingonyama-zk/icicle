#include <iostream>

#include "examples_utils.h"
#include "icicle/runtime.h"
#include "backend/cpu/src/hash/poseidon/hash.h"     // DEBUG - checnge to final hash.h
#include "icicle/fields/field.h"
#include "icicle/api/bn254.h"
#include "run_single_hash.in_params.h"

#include "backend/cpu/src/hash/poseidon/cpu_poseidon.h"

using namespace bn254;

// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 # bn254

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  std::cout << "\nIcicle Examples: Poseidon hash (single hash and many hashes)" << std::endl;
  // Parameters fit the data in run_single_hash.in_params.h file.
  const unsigned int arity = 4;
  const unsigned int alpha = 5;
  const unsigned int nof_partial_rounds = 56;
  const unsigned int nof_upper_full_rounds = 4;
  const unsigned int nof_end_full_rounds = 4;

  std::cout << "Example parameters:" << std::endl;
  std::cout << "arity: " << arity << std::endl;
  std::cout << "alpha: " << alpha << std::endl;
  std::cout << "nof_partial_rounds: " << nof_partial_rounds << std::endl;
  std::cout << "nof_upper_full_rounds: " << nof_upper_full_rounds << std::endl;
  std::cout << "nof_end_full_rounds: " << nof_end_full_rounds << std::endl;

  std::cout << "\nGenerating input data for a single Poseidon hash" << std::endl;

  // const unsigned int nof_rounds_constants = arity * (nof_upper_full_rounds + nof_end_full_rounds) + nof_partial_rounds;
  // const unsigned int nof_mds_matrix_elements = arity * arity;
  // const unsigned int nof_sparse_matrix_elements = arity;
  // auto rounds_constants = std::make_unique<scalar_t[]>(nof_rounds_constants);
  // scalar_t::rand_host_many(rounds_constants.get(), nof_rounds_constants); 
  // auto full_rounds_matrix        = std::make_unique<scalar_t[]>(nof_mds_matrix_elements);
  // scalar_t::rand_host_many(full_rounds_matrix.get(), nof_mds_matrix_elements);
  // auto partial_rounds_matrix     = std::make_unique<scalar_t[]>(nof_mds_matrix_elements);
  // scalar_t::rand_host_many(partial_rounds_matrix.get(), nof_mds_matrix_elements);

  icicle::HashConfig config;

  // Check single poseidon hash.
  std::cout << "\nPoseidon single hash test." << std::endl;
  limb_t* single_hash_out_limbs = new limb_t[scalar_t::TLC];
  scalar_t* single_hash_out_fields  = (scalar_t*)(single_hash_out_limbs);
  icicle::Poseidon<scalar_t> poseidon(arity, alpha, nof_partial_rounds, nof_upper_full_rounds, nof_end_full_rounds, rounds_constants, mds_matrix, pre_matrix, sparse_matrices);
  auto start_time = std::chrono::high_resolution_clock::now();
  ICICLE_CHECK(poseidon.run_single_hash(pre_round_input_state, single_hash_out_limbs, config));
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Poseidon single hash elapsed time: " << duration.count() << " microseconds."  << std::endl;
  // Check result of a single poseidon hash.
  bool single_hash_test_passed = true;
  if (expected_single_hash_out == *single_hash_out_fields) {
    std::cout << "\nPoseidon single hash test passed!" << std::endl; 
  } else {
    std::cout << "\nERROR!!! Poseidon single hash test does not passed!!!" << std:: endl;    
    single_hash_test_passed = false;
  }
  std::cout << "\nSingle hash generated output: " << *single_hash_out_fields << std:: endl;
  std::cout << "Single hash expected output: " << expected_single_hash_out << std:: endl;
  delete[] single_hash_out_limbs;
  single_hash_out_limbs = nullptr;
  // Return with error code if test failed.
  if (!single_hash_test_passed) return 1;

  // Check multiple poseidon hash.
  std::cout << "\nPoseidon multiple hash test." << std::endl;
  int nof_hashes = 3;  
  limb_t* multiple_hash_in_limbs = new limb_t[nof_hashes * scalar_t::TLC * arity];
  limb_t* multiple_hash_out_limbs = new limb_t[nof_hashes * scalar_t::TLC];
  // Prepare multiple hash input.
  for (int hash_idx=0; hash_idx<nof_hashes; hash_idx++) {
    for (int i=0; i<(scalar_t::TLC * arity); i++) {
      multiple_hash_in_limbs[hash_idx * scalar_t::TLC * arity + i] = pre_round_input_state[i];
    }
  }
  // Perform multiple hashes.
  start_time = std::chrono::high_resolution_clock::now();
  ICICLE_CHECK(poseidon.run_multiple_hash(multiple_hash_in_limbs, multiple_hash_out_limbs, nof_hashes, config));
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Poseidon " << nof_hashes <<" hashes elapsed time: " << duration.count() << " microseconds." << std::endl;
  // Check result of multiple poseidon hashes.
  scalar_t* multiple_hash_out_fields  = (scalar_t*)(multiple_hash_out_limbs);  
  scalar_t* multiple_hash_out_fields_for_check = multiple_hash_out_fields;    // Use multiple_hash_out_fields_for_check for check and multiple_hash_out_fields to keep original value in order to print the results later.
  bool multiple_hash_test_passed = true;
  for (int hash_idx=0; hash_idx<nof_hashes; hash_idx++) {
    if (expected_single_hash_out != *multiple_hash_out_fields_for_check++) {
      multiple_hash_test_passed = false;
    }
  }
  if (multiple_hash_test_passed) {
    std::cout << "\nPoseidon multiple hash test passed!" << std::endl;
  } else {
    std::cout << "\nERROR!!! Poseidon multiple hash test does not passed!!!" << std:: endl;
  }
  multiple_hash_out_fields_for_check = multiple_hash_out_fields;
  for (int hash_idx=0; hash_idx<nof_hashes; hash_idx++) {
    std::cout << "Multiple hash generated output: hash " << hash_idx << ", " << *multiple_hash_out_fields_for_check++ << std:: endl;
    std::cout << "Multiple hash generated output: hash " << hash_idx << ", " << *multiple_hash_out_fields++ << std:: endl;
    std::cout << "Multiple hash expected output: hash " << hash_idx << ", " << expected_single_hash_out << std:: endl;
  }

  delete[] multiple_hash_in_limbs;
  multiple_hash_in_limbs = nullptr;
  delete[] multiple_hash_out_limbs;
  multiple_hash_out_limbs = nullptr;

  if (!multiple_hash_test_passed) return 1;
  return 0;
}
