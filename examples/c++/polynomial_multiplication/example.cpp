#include <iostream>
#include <vector>
#include <memory>

#include "icicle/runtime.h"
#include "icicle/api/bn254.h"
using namespace bn254;

#include "examples_utils.h"

void random_samples(scalar_t* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? scalar_t::rand_host() : res[i - 1000];
}

void incremental_values(scalar_t* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + scalar_t::one() * scalar_t::omega(4) : scalar_t::zero();
  }
}

// calcaulting polynomial multiplication A*B via NTT,pointwise-multiplication and INTT
// (1) allocate A,B on HOST. Randomize first half, zero second half
// (2) allocate A,B,Res on device
// (3) calc NTT for A and for B from host to device
// (4) multiply d_polyRes = NttAGpu * NttBGpu (pointwise)
// (5) INTT d_polyRes inplace

int main(int argc, char** argv)
{
  try_load_and_set_backend_device(argc, argv);

  int NTT_LOG_SIZE = 20;
  int NTT_SIZE = 1 << NTT_LOG_SIZE;

  // init domain
  scalar_t basic_root = scalar_t::omega(NTT_LOG_SIZE);
  auto config = default_ntt_init_domain_config();
  bn254_ntt_init_domain(&basic_root, &config);

  // (1) cpu allocation
  auto polyA = std::make_unique<scalar_t[]>(NTT_SIZE);
  auto polyB = std::make_unique<scalar_t[]>(NTT_SIZE);
  random_samples(polyA.get(), NTT_SIZE >> 1); // second half zeros
  random_samples(polyB.get(), NTT_SIZE >> 1); // second half zeros

  scalar_t *d_polyA, *d_polyB, *d_polyRes;

  DeviceProperties device_props;
  ICICLE_CHECK(icicle_get_device_properties(device_props));

  auto benchmark = [&](bool print) {
    // (2) device input allocation. If device does not share memory with host, copy inputs explicitly and
    ICICLE_CHECK(icicle_malloc((void**)&d_polyA, sizeof(scalar_t) * NTT_SIZE));
    ICICLE_CHECK(icicle_malloc((void**)&d_polyB, sizeof(scalar_t) * NTT_SIZE));
    ICICLE_CHECK(icicle_malloc((void**)&d_polyRes, sizeof(scalar_t) * NTT_SIZE));

    // start recording
    START_TIMER(poly_multiply)

    // (3) NTT for A,B from host memory to device-memory
    auto ntt_config = default_ntt_config<scalar_t>();
    ntt_config.are_inputs_on_device = false;
    ntt_config.are_outputs_on_device = true;
    ntt_config.ordering = Ordering::kNM;
    ICICLE_CHECK(bn254_ntt(polyA.get(), NTT_SIZE, NTTDir::kForward, &ntt_config, d_polyA));
    ICICLE_CHECK(bn254_ntt(polyB.get(), NTT_SIZE, NTTDir::kForward, &ntt_config, d_polyB));

    // (4) multiply A,B
    VecOpsConfig config{
      nullptr,
      true,   // is_a_on_device
      true,   // is_b_on_device
      true,   // is_result_on_device
      false,  // is_async
      nullptr // ext
    };
    ICICLE_CHECK(bn254_vector_mul(d_polyA, d_polyB, NTT_SIZE, &config, d_polyRes));

    // (5) INTT (in place)
    ntt_config.are_inputs_on_device = true;
    ntt_config.are_outputs_on_device = true;
    ntt_config.ordering = Ordering::kMN;
    ICICLE_CHECK(bn254_ntt(d_polyRes, NTT_SIZE, NTTDir::kInverse, &ntt_config, d_polyRes));

    if (print) { END_TIMER(poly_multiply, "polynomial multiplication took"); }

    ICICLE_CHECK(icicle_free(d_polyA));
    ICICLE_CHECK(icicle_free(d_polyB));
    ICICLE_CHECK(icicle_free(d_polyRes));

    return eIcicleError::SUCCESS;
  };

  benchmark(false); // warmup
  benchmark(true);

  ICICLE_CHECK(bn254_ntt_release_domain());

  return 0;
}