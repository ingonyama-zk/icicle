#include <iostream>

#include "icicle/runtime.h"

#include "icicle/api/bn254.h"
using namespace bn254;

#include "examples_utils.h"
#include "icicle/backend/ntt_config.h"

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, scalar_t* elements);
int validate_output(const unsigned ntt_size, const unsigned nof_ntts, scalar_t* elements);

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  std::cout << "\nIcicle Examples: Number Theoretical Transform (NTT)" << std::endl;
  const unsigned log_ntt_size = 20;
  const unsigned ntt_size = 1 << log_ntt_size;
  const unsigned batch_size = 2;

  std::cout << "Example parameters:" << std::endl;
  std::cout << "NTT size: " << ntt_size << std::endl;
  std::cout << "batch size: " << batch_size << std::endl;

  std::cout << "\nGenerating input data for lowest and highest harmonics" << std::endl;
  auto input = std::make_unique<scalar_t[]>(batch_size * ntt_size);
  auto output = std::make_unique<scalar_t[]>(batch_size * ntt_size);
  initialize_input(ntt_size, batch_size, input.get());

  // Initialize NTT domain
  std::cout << "\nInit NTT domain" << std::endl;
  scalar_t basic_root = scalar_t::omega(log_ntt_size /*NTT_LOG_SIZscalar_t*/);
  auto ntt_init_domain_cfg = default_ntt_init_domain_config();
  ConfigExtension backend_cfg_ext;
  backend_cfg_ext.set(
    CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true); // optionally construct fast_twiddles for CUDA backend
  ntt_init_domain_cfg.ext = &backend_cfg_ext;
  ICICLE_CHECK(bn254_ntt_init_domain(&basic_root, ntt_init_domain_cfg));

  // ntt configuration
  NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
  ConfigExtension ntt_cfg_ext;
  config.ext = &ntt_cfg_ext;
  config.batch_size = batch_size;

  // warmup
  ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));

  // NTT radix-2 alg
  std::cout << "\nRunning NTT radix-2 alg with on-host data" << std::endl;
  ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::Radix2);
  START_TIMER(Radix2);
  ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));
  END_TIMER(Radix2, "Radix2 NTT");

  std::cout << "Validating output" << std::endl;
  validate_output(ntt_size, batch_size, output.get());

  // NTT mixed-radix alg
  std::cout << "\nRunning NTT mixed-radix alg with on-host data" << std::endl;
  ntt_cfg_ext.set(CudaBackendConfig::CUDA_NTT_ALGORITHM, CudaBackendConfig::NttAlgorithm::MixedRadix);
  START_TIMER(MixedRadix);
  ICICLE_CHECK(bn254_ntt(input.get(), ntt_size, NTTDir::kForward, config, output.get()));
  END_TIMER(MixedRadix, "MixedRadix NTT");

  std::cout << "Validating output" << std::endl;
  validate_output(ntt_size, batch_size, output.get());

  return 0;
}

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, scalar_t* elements)
{
  // Lowest Harmonics
  for (unsigned i = 0; i < ntt_size; i = i + 1) {
    elements[i] = scalar_t::one();
  }
  // Highest Harmonics
  for (unsigned i = 1 * ntt_size; i < 2 * ntt_size; i = i + 2) {
    elements[i] = scalar_t::one();
    elements[i + 1] = scalar_t::neg(scalar_t::one());
  }
}

int validate_output(const unsigned ntt_size, const unsigned nof_ntts, scalar_t* elements)
{
  int nof_errors = 0;
  scalar_t amplitude = scalar_t::from((uint32_t)ntt_size);
  // Lowest Harmonics
  if (elements[0] != amplitude) {
    ++nof_errors;
    std::cout << "Error in lowest harmonicscalar_t 0! " << std::endl;
  } else {
    std::cout << "Validated lowest harmonics" << std::endl;
  }
  // Highest Harmonics
  if (elements[1 * ntt_size + ntt_size / 2] != amplitude) {
    ++nof_errors;
    std::cout << "Error in highest harmonics! " << std::endl;
  } else {
    std::cout << "Validated highest harmonics" << std::endl;
  }
  return nof_errors;
}