#include <chrono>
#include <iostream>

// select the curve
#define CURVE_ID 1
// include NTT template
#include "appUtils/ntt/ntt.cu"
using namespace curve_config;

// Operate on scalars
typedef scalar_t S;
typedef scalar_t E;

void print_elements(const unsigned n, E * elements ) {
  for (unsigned i = 0; i < n; i++) {
    std::cout << i << ": " << elements[i] << std::endl;   
  }
}

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
  // Lowest Harmonics
  for (unsigned i = 0; i < ntt_size; i=i+1) {
    elements[i] = E::one();
  }
  // print_elements(ntt_size, elements );
  // Highest Harmonics
  for (unsigned i = 1*ntt_size; i < 2*ntt_size; i=i+2) {
    elements[i] =  E::one();
    elements[i+1] = E::neg(scalar_t::one());
  }
  // print_elements(ntt_size, &elements[1*ntt_size] );
}

int validate_output(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  int nof_errors = 0;
  E amplitude = E::from((uint32_t) ntt_size);
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
  if (elements[1*ntt_size+ntt_size/2] != amplitude) {
    ++nof_errors;
    std::cout << "Error in highest harmonics! " << std::endl;
    // print_elements(ntt_size, &elements[1*ntt_size] );
  } else {
    std::cout << "Validated highest harmonics" << std::endl;
  }
  return nof_errors;
}

int main(int argc, char* argv[])
{
  std::cout << "Icicle Examples: Number Theoretical Transform (NTT)" << std::endl;
  std::cout << "Example parameters" << std::endl;
  const unsigned log_ntt_size = 20;
  std::cout << "Log2(NTT size): " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;
  std::cout << "NTT size: " << ntt_size << std::endl;
  const unsigned nof_ntts = 2;
  std::cout << "Number of NTTs: " << nof_ntts << std::endl;
  const unsigned batch_size = nof_ntts * ntt_size;
  
  std::cout << "Generating input data for lowest and highest harmonics" << std::endl;
  E* input;
  input = (E*) malloc(sizeof(E) * batch_size);
  initialize_input(ntt_size, nof_ntts, input );
  E* output;
  output = (E*) malloc(sizeof(E) * batch_size);
  
  std::cout << "Running NTT with on-host data" << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Create a device context
  auto ctx = device_context::get_default_device_context();
  // the next line is valid only for CURVE_ID 1 (will add support for other curves soon)
  S rou = S{ {0x53337857, 0x53422da9, 0xdbed349f, 0xac616632, 0x6d1e303, 0x27508aba, 0xa0ed063, 0x26125da1} };
  ntt::InitDomain(rou, ctx);
  // Create an NTTConfig instance
  ntt::NTTConfig<S> config=ntt::DefaultNTTConfig<S>();
  config.batch_size = nof_ntts;
  config.ctx.stream = stream;
  auto begin0 = std::chrono::high_resolution_clock::now();
  cudaError_t err = ntt::NTT<S, E>(input, ntt_size, ntt::NTTDir::kForward, config, output);
  auto end0 = std::chrono::high_resolution_clock::now();
  auto elapsed0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - begin0);
  printf("On-device runtime: %.3f seconds\n", elapsed0.count() * 1e-9);
  validate_output(ntt_size, nof_ntts, output );
  cudaStreamDestroy(stream);
  free(input);
  free(output);
  return 0;
}
