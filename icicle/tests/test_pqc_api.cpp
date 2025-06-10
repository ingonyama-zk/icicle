#include <gtest/gtest.h>
#include <iostream>

#include "test_base.h"
#include "icicle/runtime.h"
#include "icicle/utils/rand_gen.h"
#include "icicle/pqc/ml_kem.h"
#include <cuda_runtime.h>

using namespace icicle;
using namespace icicle::pqc::ml_kem;

template <typename TypeParam>
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

protected:
  void SetUp() override { ICICLE_CHECK(icicle_set_device("CUDA-PQC")); }
};

typedef testing::Types<Kyber512Params, Kyber768Params, Kyber1024Params> MLkemTypes;
TYPED_TEST_SUITE(PqcTest, MLkemTypes);

TYPED_TEST(PqcTest, MLkemSharedSecretConsistencyTest)
{
  const int batch_size = 1 << 12;
  // Config
  MlKemConfig config;
  config.batch_size = batch_size;

  // Allocate buffers
  auto entropy = this->random_entropy(batch_size * ENTROPY_BYTES);
  std::vector<std::byte> public_key(batch_size * TypeParam::PUBLIC_KEY_BYTES);
  std::vector<std::byte> secret_key(batch_size * TypeParam::SECRET_KEY_BYTES);
  std::vector<std::byte> ciphertext(batch_size * TypeParam::CIPHERTEXT_BYTES);
  std::vector<std::byte> shared_secret_enc(batch_size * TypeParam::SHARED_SECRET_BYTES);
  std::vector<std::byte> shared_secret_dec(batch_size * TypeParam::SHARED_SECRET_BYTES);

  auto message = this->random_entropy(batch_size * MESSAGE_BYTES);

  // Key generation
  auto err = keygen<TypeParam>(entropy.data(), config, public_key.data(), secret_key.data());
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Encapsulation
  err = encapsulate<TypeParam>(message.data(), public_key.data(), config, ciphertext.data(), shared_secret_enc.data());
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Decapsulation
  err = decapsulate<TypeParam>(secret_key.data(), ciphertext.data(), config, shared_secret_dec.data());
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Check equality
  EXPECT_EQ(shared_secret_enc, shared_secret_dec);
}

TYPED_TEST(PqcTest, MLkemSharedSecretConsistencyTestOnDevice)
{
  const int batch_size = 1 << 12;
  // Config
  MlKemConfig config;
  config.batch_size = batch_size;
  config.entropy_on_device = true;
  config.public_keys_on_device = true;
  config.secret_keys_on_device = true;
  config.messages_on_device = true;
  config.ciphertexts_on_device = true;
  config.shared_secrets_on_device = true;
  config.is_async = true;

  // Set device
  icicle::Device dev = {"CUDA-PQC", 0};
  ICICLE_CHECK(icicle_set_device(dev));

  // Create stream
  icicleStreamHandle stream;
  ICICLE_CHECK(icicle_create_stream(&stream));
  config.stream = stream;

  // Allocate device buffers
  std::byte* d_entropy;
  ICICLE_CHECK(
    icicle_malloc_async(reinterpret_cast<void**>(&d_entropy), batch_size * ENTROPY_BYTES * sizeof(std::byte), stream));
  std::byte* d_public_key;
  ICICLE_CHECK(icicle_malloc_async(
    reinterpret_cast<void**>(&d_public_key), batch_size * TypeParam::PUBLIC_KEY_BYTES * sizeof(std::byte), stream));
  std::byte* d_secret_key;
  ICICLE_CHECK(icicle_malloc_async(
    reinterpret_cast<void**>(&d_secret_key), batch_size * TypeParam::SECRET_KEY_BYTES * sizeof(std::byte), stream));
  std::byte* d_ciphertext;
  ICICLE_CHECK(icicle_malloc_async(
    reinterpret_cast<void**>(&d_ciphertext), batch_size * TypeParam::CIPHERTEXT_BYTES * sizeof(std::byte), stream));
  std::byte* d_shared_secret_enc;
  ICICLE_CHECK(icicle_malloc_async(
    reinterpret_cast<void**>(&d_shared_secret_enc), batch_size * TypeParam::SHARED_SECRET_BYTES * sizeof(std::byte),
    stream));
  std::byte* d_shared_secret_dec;
  ICICLE_CHECK(icicle_malloc_async(
    reinterpret_cast<void**>(&d_shared_secret_dec), batch_size * TypeParam::SHARED_SECRET_BYTES * sizeof(std::byte),
    stream));

  // Generate random entropy and copy to device
  auto h_entropy = this->random_entropy(batch_size * ENTROPY_BYTES);
  ICICLE_CHECK(
    icicle_copy_to_device_async(d_entropy, h_entropy.data(), batch_size * ENTROPY_BYTES * sizeof(std::byte), stream));

  // Key generation
  auto err = keygen<TypeParam>(d_entropy, config, d_public_key, d_secret_key);
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Encapsulation
  auto h_message = this->random_entropy(batch_size * MESSAGE_BYTES);
  std::byte* d_message;
  ICICLE_CHECK(
    icicle_malloc_async(reinterpret_cast<void**>(&d_message), batch_size * MESSAGE_BYTES * sizeof(std::byte), stream));
  ICICLE_CHECK(
    icicle_copy_to_device_async(d_message, h_message.data(), batch_size * MESSAGE_BYTES * sizeof(std::byte), stream));
  err = encapsulate<TypeParam>(d_message, d_public_key, config, d_ciphertext, d_shared_secret_enc);
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Decapsulation
  err = decapsulate<TypeParam>(d_secret_key, d_ciphertext, config, d_shared_secret_dec);
  ASSERT_EQ(err, eIcicleError::SUCCESS);

  // Copy results back to host for comparison
  std::vector<std::byte> h_shared_secret_enc(batch_size * TypeParam::SHARED_SECRET_BYTES);
  std::vector<std::byte> h_shared_secret_dec(batch_size * TypeParam::SHARED_SECRET_BYTES);
  ICICLE_CHECK(icicle_copy_to_host_async(
    h_shared_secret_enc.data(), d_shared_secret_enc, batch_size * TypeParam::SHARED_SECRET_BYTES * sizeof(std::byte),
    stream));
  ICICLE_CHECK(icicle_copy_to_host_async(
    h_shared_secret_dec.data(), d_shared_secret_dec, batch_size * TypeParam::SHARED_SECRET_BYTES * sizeof(std::byte),
    stream));

  // Synchronize stream to ensure all operations are complete
  ICICLE_CHECK(icicle_stream_synchronize(stream));

  // Check equality
  EXPECT_EQ(h_shared_secret_enc, h_shared_secret_dec);

  // Free device memory
  ICICLE_CHECK(icicle_free_async(d_entropy, stream));
  ICICLE_CHECK(icicle_free_async(d_public_key, stream));
  ICICLE_CHECK(icicle_free_async(d_secret_key, stream));
  ICICLE_CHECK(icicle_free_async(d_ciphertext, stream));
  ICICLE_CHECK(icicle_free_async(d_shared_secret_enc, stream));
  ICICLE_CHECK(icicle_free_async(d_shared_secret_dec, stream));
  ICICLE_CHECK(icicle_free_async(d_message, stream));

  // Destroy stream
  ICICLE_CHECK(icicle_destroy_stream(stream));
}
