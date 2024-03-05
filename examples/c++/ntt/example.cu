#include <chrono>
#include <iostream>

// select the curve
#define CURVE_ID 1
// include NTT template
#include "appUtils/ntt/ntt.cu"
#include "appUtils/ntt/kernel_ntt.cu"
using namespace curve_config;
using namespace ntt;

// Operate on scalars
typedef scalar_t S;
typedef projective_t E;

void print_elements(const unsigned n, E* elements)
{
  for (unsigned i = 0; i < n; i++) {
    std::cout << i << ": " << elements[i] << std::endl;
  }
}

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  // Lowest Harmonics
  for (unsigned i = 0; i < ntt_size; i = i + 1) {
    elements[i] = E::zero();
  }
  // print_elements(ntt_size, elements );
  // Highest Harmonics
  for (unsigned i = 1 * ntt_size; i < 2 * ntt_size; i = i + 2) {
    elements[i] = E::zero();
    elements[i + 1] = E::neg(E::zero());
  }
  // print_elements(ntt_size, &elements[1*ntt_size] );
}

int validate_output(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  int nof_errors = 0;
  E amplitude = E::from_affine(affine_t::zero());
  // std::cout << "Amplitude: " << amplitude << std::endl;
  // Lowest Harmonics
  if (elements[0] != amplitude) {
    ++nof_errors;
    std::cout << "Error in lowest harmonics 0! " << std::endl;
    // print_elements(ntt_size, elements );
  } else {
    std::cout << "Validated lowest harmonics" << std::endl;
  }
  // Highest Harmonics
  if (elements[1 * ntt_size + ntt_size / 2] != amplitude) {
    ++nof_errors;
    std::cout << "Error in highest harmonics! " << std::endl;
    // print_elements(ntt_size, &elements[1*ntt_size] );
  } else {
    std::cout << "Validated highest harmonics" << std::endl;
  }
  return nof_errors;
}

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());


int main(int argc, char* argv[])
{
#if defined(ECNTT_DEFINED)
  std::cout << "Icicle Examples: Number Theoretical Transform (NTT)" << std::endl;
  std::cout << "Example parameters" << std::endl;
  const unsigned log_ntt_size = 8;
  std::cout << "Log2(NTT size): " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;
  std::cout << "NTT size: " << ntt_size << std::endl;
  const unsigned nof_ntts = 2;
  std::cout << "Number of NTTs: " << nof_ntts << std::endl;
  const unsigned batch_size = nof_ntts * ntt_size;

  std::cout << "Generating input data for lowest and highest harmonics" << std::endl;
  E* input;
  input = (E*)malloc(sizeof(E) * batch_size);
  initialize_input(ntt_size, nof_ntts, input);
  E* output;
  output = (E*)malloc(sizeof(E) * batch_size);

  std::cout << "Running NTT with on-host data" << std::endl;
  // Create a device context
  auto ctx = device_context::get_default_device_context();
  const S basic_root = S::omega(log_ntt_size /*NTT_LOG_SIZE*/);
  InitDomain(basic_root, ctx);
  // Create an NTTConfig instance
  NTTConfig<S> config = DefaultNTTConfig<S>();
  config.ntt_algorithm = NttAlgorithm::MixedRadix; 
  config.batch_size = nof_ntts;
  START_TIMER(MixedRadix);
  cudaError_t err = ntt::ECNTT_(input, ntt_size, NTTDir::kForward, config, output);
  END_TIMER(MixedRadix, "MixedRadix NTT");
  
  std::cout << "Validating output" << std::endl;
  validate_output(ntt_size, nof_ntts, output);

  config.ntt_algorithm = NttAlgorithm::Radix2; 
  START_TIMER(Radix2);
  err = ECNTT_(input, ntt_size, NTTDir::kForward, config, output);
  END_TIMER(Radix2, "Radix2 NTT");

  std::cout << "Validating output" << std::endl;
  validate_output(ntt_size, nof_ntts, output);

  std::cout << "Cleaning-up memory" << std::endl;
  free(input);
  free(output);
#endif
  return 0;
}
