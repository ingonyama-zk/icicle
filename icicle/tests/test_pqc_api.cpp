
#include <gtest/gtest.h>
#include <iostream>

#include "test_base.h"
#include "icicle/runtime.h"
#include "icicle/utils/rand_gen.h"
#include "icicle/pqc/ml_kem.h"

using namespace icicle;
using namespace icicle::pqc;
class PqcTest : public IcicleTestBase
{
public:
  std::vector<std::byte> random_entropy(size_t size)
  {
    std::vector<std::byte> buf(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 255);
    for (auto& b : buf) {
      b = static_cast<std::byte>(dist(gen));
    }
    return buf;
  }
};

TEST_F(PqcTest, MLkemTest)
{
  ICICLE_CHECK(icicle_set_device("CUDA-PQC"));
  // TODO: test all security categories on all devices, with batch
  using namespace ml_kem;
  constexpr SecurityCategory category = SecurityCategory::KYBER_512;

  const int batch_size = 1;

  // Allocate buffers
  auto entropy = random_entropy(batch_size * ENTROPY_BYTES);
  std::vector<std::byte> public_key(batch_size * Kyber512::PUBLIC_KEY_BYTES);
  std::vector<std::byte> secret_key(batch_size * Kyber512::SECRET_KEY_BYTES);
  std::vector<std::byte> ciphertext(batch_size * Kyber512::CIPHERTEXT_BYTES);
  std::vector<std::byte> shared_secret_enc(batch_size * Kyber512::SHARED_SECRET_BYTES);
  std::vector<std::byte> shared_secret_dec(batch_size * Kyber512::SHARED_SECRET_BYTES);

  // Config
  MlKemConfig config;
  config.batch_size = 1;

  // Key generation
  auto err = keygen(category, entropy.data(), config, public_key.data(), secret_key.data());
  // ASSERT_EQ(err, ICICLE_SUCCESS);

  // Encapsulation
  err = encapsulate(category, public_key.data(), config, ciphertext.data(), shared_secret_enc.data());
  // ASSERT_EQ(err, ICICLE_SUCCESS);

  // Decapsulation
  err = decapsulate(category, secret_key.data(), ciphertext.data(), config, shared_secret_dec.data());
  // ASSERT_EQ(err, ICICLE_SUCCESS);

  // Check equality
  // EXPECT_EQ(std::memcmp(shared_secret_enc.data(), shared_secret_dec.data(), Kyber512::SHARED_SECRET_BYTES), 0);
}

// TODO test with device memory

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}