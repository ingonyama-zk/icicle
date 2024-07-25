#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>

#include "icicle/runtime.h"
#include "icicle/api/bn254.h"
using namespace bn254;

#include "examples_utils.h"

void initialize_input(const unsigned ntt_size, const unsigned batch_size, scalar_t* elements)
{
  for (unsigned i = 0; i < ntt_size * batch_size; i++) {
    elements[i] = scalar_t::from(i + 1);
  }
}

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  // set these parameters to match the desired NTT size and batch size
  const unsigned log_ntt_size = 20;
  const unsigned batch_size = 16;

  scalar_t basic_root = scalar_t::omega(log_ntt_size);

  const unsigned ntt_size = 1 << log_ntt_size;
  std::cout << "log NTT size: " << log_ntt_size << std::endl;
  std::cout << "Batch size: " << batch_size << std::endl;

  // Create separate streams for overlapping data transfers and kernel execution.
  icicleStreamHandle stream_compute, stream_h2d, stream_d2h;
  ICICLE_CHECK(icicle_create_stream(&stream_compute));
  ICICLE_CHECK(icicle_create_stream(&stream_h2d));
  ICICLE_CHECK(icicle_create_stream(&stream_d2h));

  // Initialize NTT domain
  std::cout << "Init NTT domain" << std::endl;
  auto ntt_init_domain_cfg = default_ntt_init_domain_config();
  // set CUDA backend specific flag for init_domain
  ConfigExtension backend_cfg_ext;
  backend_cfg_ext.set("fast_twiddles", true);
  ntt_init_domain_cfg.ext = &backend_cfg_ext;
  ICICLE_CHECK(bn254_ntt_init_domain(&basic_root, ntt_init_domain_cfg));

  std::cout << "Concurrent Download, Upload, and Compute In-place NTT" << std::endl;
  int nof_blocks = 32;
  int block_size = ntt_size * batch_size / nof_blocks;
  std::cout << "Number of blocks: " << nof_blocks << ", block size: " << block_size << " Bytes" << std::endl;

  // on-host pinned data
  scalar_t* h_inp[2];
  scalar_t* h_out[2];
  for (int i = 0; i < 2; i++) {
    h_inp[i] = new scalar_t[ntt_size * batch_size];
    h_out[i] = new scalar_t[ntt_size * batch_size];
  }

  // on-device in-place data
  // we need two on-device vectors to overlap data transfers with NTT kernel execution
  scalar_t* d_vec[2];
  for (int i = 0; i < 2; i++) {
    ICICLE_CHECK(icicle_malloc((void**)&d_vec[i], sizeof(scalar_t) * ntt_size * batch_size));
  }

  // initialize input data
  initialize_input(ntt_size, batch_size, h_inp[0]);
  initialize_input(ntt_size, batch_size, h_inp[1]);

  // ntt configuration
  NTTConfig<scalar_t> config_compute = default_ntt_config<scalar_t>();  
  config_compute.batch_size = batch_size;
  config_compute.are_inputs_on_device = true;
  config_compute.are_outputs_on_device = true;
  config_compute.is_async = true;
  config_compute.stream = stream_compute;
  //  backend specific config extension
  ConfigExtension ntt_cfg_ext;
  ntt_cfg_ext.set("ntt_algorithm", 2); // mixed-radix
  config_compute.ext = &ntt_cfg_ext;

  for (int run = 0; run < 10; run++) {
    int vec_compute = run % 2;
    int vec_transfer = (run + 1) % 2;
    std::cout << "Run: " << run << std::endl;
    std::cout << "Compute Vector: " << vec_compute << std::endl;
    std::cout << "Transfer Vector: " << vec_transfer << std::endl;
    START_TIMER(inplace);
    bn254_ntt(d_vec[vec_compute], ntt_size, NTTDir::kForward, config_compute, d_vec[vec_compute]);
    // we have to delay upload to device relative to download from device by one block: preserve write after read
    for (int i = 0; i <= nof_blocks; i++) {
      if (i < nof_blocks) {
        // copy result back from device to host
        ICICLE_CHECK(icicle_copy_async(
          &h_out[vec_transfer][i * block_size], &d_vec[vec_transfer][i * block_size], sizeof(scalar_t) * block_size,
          stream_d2h));
      }
      if (i > 0) {
        // copy next input from host to device to alternate buffer
        ICICLE_CHECK(icicle_copy_async(
          &d_vec[vec_transfer][(i - 1) * block_size], &h_inp[vec_transfer][(i - 1) * block_size],
          sizeof(scalar_t) * block_size, stream_h2d));
      }
      // synchronize upload and download at the end of the block to ensure data integrity
      ICICLE_CHECK(icicle_stream_synchronize(stream_d2h));
      ICICLE_CHECK(icicle_stream_synchronize(stream_h2d));
    }
    // synchronize compute stream with the end of the computation
    ICICLE_CHECK(icicle_stream_synchronize(stream_compute));
    END_TIMER(inplace, "Concurrent In-Place  NTT");
  }

  // Clean-up
  for (int i = 0; i < 2; i++) {
    ICICLE_CHECK(icicle_free(d_vec[i]));
    delete[] (h_inp[i]);
    delete[] (h_out[i]);
  }
  ICICLE_CHECK(icicle_destroy_stream(stream_compute));
  ICICLE_CHECK(icicle_destroy_stream(stream_d2h));
  ICICLE_CHECK(icicle_destroy_stream(stream_h2d));
  return 0;
}
