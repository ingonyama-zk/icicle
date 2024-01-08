#include <chrono>
#include <iostream>

// include NTT template
#include "icicle/appUtils/ntt/ntt.cuh"

// select the curve
#include "icicle/curves/bls12_381/curve_config.cuh"
using namespace BLS12_381;

// Operate on scalars
typedef scalar_t S;
typedef scalar_t E;

scalar_t smult(const unsigned n, scalar_t s)
{
  scalar_t r = scalar_t::zero();
  for (unsigned i = 0; i < n; i++) {
    r = r + s;
  }
  return r;
}

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
  }
}

int validate_output(const unsigned ntt_size, const unsigned nof_ntts, E* elements)
{
  int nof_errors = 0;
  E amplitude = smult(ntt_size, scalar_t::one());
  // std::cout << "Amplitude: " << amplitude << std::endl;
  // Harmonics 0
  if (elements[0] != amplitude) {
    ++nof_errors;
    std::cout << "Error in harmonics 0: " << elements[0] << std::endl;
  } else {
    std::cout << "Validated harmonics 0" << std::endl;
  }
  // Harmonics 1
  if (elements[ntt_size + 1] != amplitude) {
    ++nof_errors;
    std::cout << "Error in harmonics 1: " << elements[ntt_size + 1] << std::endl;
  } else {
    std::cout << "Validated harmonics 1" << std::endl;
  }
  // for (unsigned i = 0; i < nof_ntts * ntt_size; i++) {
  //   std::cout << elements[i] << std::endl;
  // }
  return nof_errors;
}

int main(int argc, char* argv[])
{
  std::cout << "Icicle Examples: Number Theoretical Transform (NTT)" << std::endl;
  std::cout << "Example parameters" << std::endl;
  const unsigned log_ntt_size = 26;
  std::cout << "Log2(NTT size): " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;
  std::cout << "NTT size: " << ntt_size << std::endl;
  const unsigned nof_ntts = 2;
  std::cout << "Number of NTTs: " << nof_ntts << std::endl;
  const unsigned batch_size = nof_ntts * ntt_size;

  std::cout << "Generating input data for harmonics 0,1" << std::endl;
  E* elements;
  elements = (scalar_t*)malloc(sizeof(E) * batch_size);
  initialize_input(ntt_size, nof_ntts, elements);

  std::cout << "Running easy-to-use NTT" << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool inverse = false;
  auto begin0 = std::chrono::high_resolution_clock::now();
  ntt_end2end_batch_template<scalar_t, scalar_t>(elements, batch_size, ntt_size, inverse, stream);
  auto end0 = std::chrono::high_resolution_clock::now();
  auto elapsed0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - begin0);
  printf("On-device runtime: %.3f seconds\n", elapsed0.count() * 1e-9);
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

  cudaStreamDestroy(stream);

  free(elements);
  return 0;
}
