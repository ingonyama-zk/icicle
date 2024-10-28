#include <icicle/api/bn254.h>
#include <icicle/backend/msm_config.h>
#include <icicle/device.h>
#include <icicle/errors.h>
#include <icicle/msm.h>
#include <icicle/runtime.h>

#include <filesystem>
#include <fstream>

using namespace bn254;

std::vector<char> read_file(const char *filename) {
  std::ifstream file(filename, std::ios::binary);
  const auto file_size = std::filesystem::file_size(filename);

  std::vector<char> bytes(file_size);

  file.read(bytes.data(), file_size);

  return bytes;
}

int main() {
  eIcicleError result = icicle_load_backend("/home/administrator/users/emir/icicle/icicle/build/backend/cuda", true);
  if (result != eIcicleError::SUCCESS) {
    std::cout << "backend load failed" << std::endl;
    return -1;
  }

  // auto scalars = read_file("./scalars_1");
  // auto points = read_file("./points_1");

  int batch_size = 1;
  unsigned msm_size = 1 << 10;
  int N = batch_size * msm_size;

  auto scalars = std::make_unique<scalar_t[]>(N);
  auto points = std::make_unique<affine_t[]>(N);

  scalar_t::rand_host_many(scalars.get(), N);
  projective_t::rand_host_many(points.get(), N);

  std::cout << "size: " << N << std::endl;

  MSMConfig config = default_msm_config();
  config.are_scalars_montgomery_form = true;
  config.are_points_montgomery_form = false;
  // config.c = 20;

  ConfigExtension ext;
  ext.set(CudaBackendConfig::CUDA_MSM_NOF_CHUNKS, 1);
  config.ext = &ext;

  {
    icicle::Device device = {"CPU", 0};

    result = icicle_set_device(device);
    if (result != eIcicleError::SUCCESS) {
      std::cout << "device set failed" << std::endl;
      return -1;
    }

    bn254::projective_t msm_result;
    result =
        msm((bn254::scalar_t *)scalars.get(), (bn254::affine_t *)points.get(),
            N, config, &msm_result);
    if (result != eIcicleError::SUCCESS) {
      std::cout << "msm failed" << std::endl;
      return -1;
    }

    bn254::affine_t apoint;
    bn254_to_affine(&msm_result, &apoint);

    // std::cout << msm_result << std::endl;
    std::cout << apoint << std::endl;
  }
  {
    icicle::Device device = {"CUDA", 0};

    result = icicle_set_device(device);
    if (result != eIcicleError::SUCCESS) {
      std::cout << "device set failed" << std::endl;
      return -1;
    }

    bn254::projective_t msm_result;
    result =
        msm((bn254::scalar_t *)scalars.get(), (bn254::affine_t *)points.get(),
            N, config, &msm_result);
    if (result != eIcicleError::SUCCESS) {
      std::cout << "msm failed" << std::endl;
      return -1;
    }

    bn254::affine_t apoint;
    bn254_to_affine(&msm_result, &apoint);

    // std::cout << msm_result << std::endl;
    std::cout << apoint << std::endl;
  }

  return 0;
}