#include <chrono>
#include <iostream>

<<<<<<< HEAD
=======
// include NTT template
#include "icicle/appUtils/ntt/ntt.cuh"

>>>>>>> dev
// select the curve
#define CURVE_ID 1
// include NTT template
#include "icicle/appUtils/ntt/ntt.cu"
using namespace curve_config;

// Operate on scalars
typedef scalar_t S;
typedef scalar_t E;

<<<<<<< HEAD
void print_elements(const unsigned n, E * elements ) {
  for (unsigned i = 0; i < n; i++) {
    std::cout << i << ": " << elements[i] << std::endl;   
=======
scalar_t smult(const unsigned n, scalar_t s)
{
  scalar_t r = scalar_t::zero();
  for (unsigned i = 0; i < n; i++) {
    r = r + s;
>>>>>>> dev
  }
}

<<<<<<< HEAD
void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
  // Lowest Harmonics
  for (unsigned i = 0; i < ntt_size; i=i+1) {
    elements[i] = scalar_t::one();
  }
  // print_elements(ntt_size, elements );
  // Highest Harmonics
  for (unsigned i = 1*ntt_size; i < 2*ntt_size; i=i+2) {
    elements[i] =  scalar_t::one();
    elements[i+1] = scalar_t::neg(scalar_t::one());
=======
void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  // Harmonics 0
  for (unsigned i = 0; i < ntt_size; i = i + 1) {
    elements[i] = scalar_t::one();
  }
  // Harmonics 1
  for (unsigned i = 1 * ntt_size; i < 2 * ntt_size; i = i + 2) {
    elements[i] = scalar_t::one();
    elements[i + 1] = scalar_t::neg(scalar_t::one());
>>>>>>> dev
  }
  // print_elements(ntt_size, &elements[1*ntt_size] );
}

int validate_output(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  int nof_errors = 0;
<<<<<<< HEAD
  E amplitude = scalar_t::from((uint32_t) ntt_size);
=======
  E amplitude = smult(ntt_size, scalar_t::one());
>>>>>>> dev
  // std::cout << "Amplitude: " << amplitude << std::endl;
  // Lowest Harmonics
  if (elements[0] != amplitude) {
    ++nof_errors;
    std::cout << "Error in lowest harmonics 0! " << std::endl;
    // print_elements(ntt_size, elements );
  } else {
    std::cout << "Validated lowest harmonics" << std::endl;
  }
<<<<<<< HEAD
  // Highest Harmonics 
  if (elements[1*ntt_size+ntt_size/2] != amplitude) {
    ++nof_errors;
    std::cout << "Error in highest harmonics! " << std::endl;
    // print_elements(ntt_size, &elements[1*ntt_size] );
=======
  // Harmonics 1
  if (elements[ntt_size + 1] != amplitude) {
    ++nof_errors;
    std::cout << "Error in harmonics 1: " << elements[ntt_size + 1] << std::endl;
>>>>>>> dev
  } else {
    std::cout << "Validated highest harmonics" << std::endl;
  }
<<<<<<< HEAD
=======
  // for (unsigned i = 0; i < nof_ntts * ntt_size; i++) {
  //   std::cout << elements[i] << std::endl;
  // }
>>>>>>> dev
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
<<<<<<< HEAD
  
  std::cout << "Generating input data for loest and highest harmonics" << std::endl;
  E* input;
  input = (scalar_t*) malloc(sizeof(E) * batch_size);
  initialize_input(ntt_size, nof_ntts, input );
  E* output;
  output = (scalar_t*) malloc(sizeof(E) * batch_size);
  
  std::cout << "Running NTT with on-host data" << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Create a device context
  auto ctx = device_context::get_default_device_context();
  // the next line is valid only for CURVE_ID 1 (will add support for other curves soon)
  scalar_t rou = scalar_t{ {0x53337857, 0x53422da9, 0xdbed349f, 0xac616632, 0x6d1e303, 0x27508aba, 0xa0ed063, 0x26125da1} };
  ntt::InitDomain(rou, ctx);
  // Create an NTTConfig instance
  ntt::NTTConfig<S> config=ntt::GetDefaultNTTConfig();
  config.batch_size = nof_ntts;
  config.ctx.stream = stream;
=======

  std::cout << "Generating input data for harmonics 0,1" << std::endl;
  E* elements;
  elements = (scalar_t*)malloc(sizeof(E) * batch_size);
  initialize_input(ntt_size, nof_ntts, elements);

  std::cout << "Running easy-to-use NTT" << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool inverse = false;
>>>>>>> dev
  auto begin0 = std::chrono::high_resolution_clock::now();
  cudaError_t err = ntt::NTT<S, E>(input, ntt_size, ntt::NTTDir::kForward, config, output);
  auto end0 = std::chrono::high_resolution_clock::now();
  auto elapsed0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - begin0);
  printf("On-device runtime: %.3f seconds\n", elapsed0.count() * 1e-9);
<<<<<<< HEAD
  validate_output(ntt_size, nof_ntts, output );
=======
  validate_output(ntt_size, nof_ntts, elements);
  cudaStreamSynchronize(stream);

  std::cout << "Running not that easy-to-use but fast NTT" << std::endl;

  uint32_t n_twiddles = ntt_size; // n_twiddles is set to 4096 as BLS12_381::scalar_t::omega() is of that order.
  // represent transform matrix using twiddle factors
  scalar_t* d_twiddles;
  d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega(log_ntt_size), stream); // Sscalar
  scalar_t* d_elements;                                                                       // Element

  cudaMallocAsync(&d_elements, sizeof(scalar_t) * batch_size, stream);
  initialize_input(ntt_size, nof_ntts, elements);
  cudaMemcpyAsync(d_elements, elements, sizeof(scalar_t) * batch_size, cudaMemcpyHostToDevice, stream);
  S* _null = nullptr;
  auto begin1 = std::chrono::high_resolution_clock::now();
  cudaStreamSynchronize(stream);
  ntt_inplace_batch_template(d_elements, d_twiddles, ntt_size, nof_ntts, inverse, false, _null, stream, false);
  cudaStreamSynchronize(stream);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
  printf("Runtime: %.3e seconds\n", elapsed1.count() * 1e-9);

  cudaMemcpyAsync(elements, d_elements, sizeof(E) * batch_size, cudaMemcpyDeviceToHost, stream);
  validate_output(ntt_size, nof_ntts, elements);
  cudaFreeAsync(d_elements, stream);
  cudaFreeAsync(d_twiddles, stream);

>>>>>>> dev
  cudaStreamDestroy(stream);
  free(input);
  free(output);
  return 0;
}
