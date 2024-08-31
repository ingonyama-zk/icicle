#include "icicle/api/bn254.h"

#include <iostream>

#include "examples_utils.h"
#include "icicle/runtime.h"
#include "/home/administrator/users/danny/github/icicle/icicle_v3/backend/cpu/src/hash/poseidon/hash.h"

#include "/home/administrator/users/danny/github/icicle/icicle_v3/include/icicle/fields/field.h"

#include "/home/administrator/users/danny/github/icicle/icicle_v3/backend/cpu/src/hash/poseidon/cpu_poseidon.h"

using namespace bn254;

// void initialize_input(const unsigned int arity, const unsigned int alpha, const unsigned int nof_total_rounds, const unsigned int nof_full_rounds_half,
//   T* full_rounds_constants, T* partial_rounds_constants, T* full_round_matrix, T* partial_round_matrix);
// int validate_output(const unsigned int arity, const unsigned int alpha, const unsigned int nof_total_rounds, const unsigned int nof_full_rounds_half,
//   scalar_t* full_rounds_constants, scalar_t* partial_rounds_constants, scalar_t* full_round_matrix, scalar_t* partial_round_matrix);

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  std::cout << "\nIcicle Examples: Poseidon hash (single hash and many hashes)" << std::endl;
  const unsigned int arity = 12;
  const unsigned int alpha = 5;
  const unsigned int nof_total_rounds = 20;
  const unsigned int nof_upper_full_rounds = 1;
  const unsigned int nof_end_full_rounds = 1;

  std::cout << "Example parameters:" << std::endl;
  std::cout << "arity: " << arity << std::endl;
  std::cout << "alpha: " << alpha << std::endl;
  std::cout << "nof_total_rounds: " << nof_total_rounds << std::endl;
  std::cout << "nof_upper_full_rounds: " << nof_upper_full_rounds << std::endl;
  std::cout << "nof_end_full_rounds: " << nof_end_full_rounds << std::endl;

  std::cout << "\nGenerating input data for a single Poseidon hash" << std::endl;
  // const unsigned int nof_full_rounds_constants = arity * (nof_upper_full_rounds + nof_end_full_rounds);
  // auto full_rounds_constants = std::make_unique<scalar_t[]>(nof_full_rounds_constants);
  // scalar_t::rand_host_many(full_rounds_constants.get(), nof_full_rounds_constants); 
  // const unsigned int nof_partial_rounds_constants = arity * (nof_total_rounds - (nof_upper_full_rounds + nof_end_full_rounds));
  // auto partial_rounds_constants = std::make_unique<scalar_t[]>(nof_partial_rounds_constants);
  // scalar_t::rand_host_many(partial_rounds_constants.get(), nof_partial_rounds_constants);
  // const unsigned int nof_mds_elements = arity * arity;
  // auto full_rounds_matrix        = std::make_unique<scalar_t[]>(nof_mds_elements);
  // scalar_t::rand_host_many(full_rounds_matrix.get(), nof_mds_elements);
  // auto partial_rounds_matrix     = std::make_unique<scalar_t[]>(nof_mds_elements);
  // scalar_t::rand_host_many(partial_rounds_matrix.get(), nof_mds_elements);

  const unsigned int nof_full_rounds_constants = arity * (nof_upper_full_rounds + nof_end_full_rounds);
  scalar_t* full_rounds_constants = (scalar_t*)malloc(nof_full_rounds_constants * sizeof(scalar_t));
  scalar_t::rand_host_many(full_rounds_constants, nof_full_rounds_constants); 
  const unsigned int nof_partial_rounds_constants = arity * (nof_total_rounds - (nof_upper_full_rounds + nof_end_full_rounds));
  scalar_t* partial_rounds_constants = (scalar_t*)malloc(nof_partial_rounds_constants * sizeof(scalar_t));
  scalar_t::rand_host_many(partial_rounds_constants, nof_partial_rounds_constants);
  const unsigned int nof_mds_elements = arity * arity;
  scalar_t* full_rounds_matrix = (scalar_t*)malloc(nof_mds_elements * sizeof(scalar_t));
  scalar_t::rand_host_many(full_rounds_matrix, nof_mds_elements);
  scalar_t* partial_rounds_matrix = (scalar_t*)malloc(nof_mds_elements * sizeof(scalar_t));
  scalar_t::rand_host_many(partial_rounds_matrix, nof_mds_elements);

  // icicle::Poseidon<scalar_t> poseidon_single_hash(12, 5, 50, 1, 1, full_rounds_constants.get(), partial_rounds_constants.get(), full_rounds_matrix.get(), partial_rounds_matrix.get());
  icicle::Poseidon<scalar_t> poseidon_single_hash(12, 5, 50, 1, 1, full_rounds_constants, partial_rounds_constants, full_rounds_matrix, partial_rounds_matrix);  

  int nof_limbs_in_scalar = scalar_t::TLC;
  limb_t* single_hash_in  = (limb_t*)malloc(nof_limbs_in_scalar * arity * sizeof(limb_t));
  limb_t* single_hash_out = (limb_t*)malloc(nof_limbs_in_scalar * arity * sizeof(limb_t));
  single_hash_in[0] = 1;
  single_hash_in[1] = 2;
  single_hash_in[7] = 0x90000008;

  // initialize_input(arity, alpha, nof_total_rounds, nof_full_rounds, full_rounds_constants, partial_rounds_constants, partial_round_matrix);

  icicle::HashConfig config;

  ICICLE_CHECK(poseidon_single_hash.run_single_hash(single_hash_in, single_hash_out, config));

  // S* in_field_op = (S*)input_limbs[i];

  // imb_t* input_limbs = (limb_t*)

  // eIcicleError run_single_hash (const limb_t* input_limbs, limb_t* output_limbs, config);


  // // Initialize NTT domain
  // std::cout << "\nInit NTT domain" << std::endl;
  // scalar_t basic_root = scalar_t::omega(log_ntt_size /*NTT_LOG_SIZscalar_t*/);
  // auto ntt_init_domain_cfg = default_ntt_init_domain_config();
  // ConfigExtension backend_cfg_ext;
  // backend_cfg_ext.set(
  //   CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true); // optionally construct fast_twiddles for CUDA backend
  // ntt_init_domain_cfg.ext = &backend_cfg_ext;
  // ICICLE_CHECK(bn254_ntt_init_domain(&basic_root, ntt_init_domain_cfg));

  // // ntt configuration
  // NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
  // ConfigExtension ntt_cfg_ext;
  // config.ext = &ntt_cfg_ext;
  // config.batch_size = batch_size;

  // // warmup
  // ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));

  // // NTT radix-2 alg
  // std::cout << "\nRunning NTT radix-2 alg with on-host data" << std::endl;
  // ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::Radix2);
  // START_TIMER(Radix2);
  // ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));
  // END_TIMER(Radix2, "Radix2 NTT");

  // std::cout << "Validating output" << std::endl;
  // validate_output(ntt_size, batch_size, output.get());

  // // NTT mixed-radix alg
  // std::cout << "\nRunning NTT mixed-radix alg with on-host data" << std::endl;
  // ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::MixedRadix);
  // START_TIMER(MixedRadix);
  // ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));
  // END_TIMER(MixedRadix, "MixedRadix NTT");

  // std::cout << "Validating output" << std::endl;
  // validate_output(ntt_size, batch_size, output.get());

  return 0;
}

// void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, scalar_t* elements)
// {



//   // Lowest Harmonics
//   for (unsigned i = 0; i < ntt_size; i = i + 1) {
//     elements[i] = scalar_t::one();
//   }
//   // Highest Harmonics
//   for (unsigned i = 1 * ntt_size; i < 2 * ntt_size; i = i + 2) {
//     elements[i] = scalar_t::one();
//     elements[i + 1] = scalar_t::neg(scalar_t::one());
//   }
// }

// int validate_output(const unsigned int arity, const unsigned int alpha, const unsigned int nof_total_rounds, const unsigned int nof_full_rounds_half,
//   scalar_t* full_rounds_constants, scalar_t* partial_rounds_constants, scalar_t* full_round_matrix, scalar_t* partial_round_matrix)
// {
//   int nof_errors = 0;
  // scalar_t amplitude = scalar_t::from((uint32_t)ntt_size);
  // // Lowest Harmonics
  // if (elements[0] != amplitude) {
  //   ++nof_errors;
  //   std::cout << "Error in lowest harmonicscalar_t 0! " << std::endl;
  // } else {
  //   std::cout << "Validated lowest harmonics" << std::endl;
  // }
  // // Highest Harmonics
  // if (elements[1 * ntt_size + ntt_size / 2] != amplitude) {
  //   ++nof_errors;
  //   std::cout << "Error in highest harmonics! " << std::endl;
  // } else {
  //   std::cout << "Validated highest harmonics" << std::endl;
  // }
//   return nof_errors;
// }
