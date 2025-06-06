#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"
#include "icicle/hash/blake3.h"
#include "icicle/hash/poseidon2.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/fields/field.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "test_base.h"
#include "icicle/utils/rand_gen.h"
#include "icicle/merkle/merkle_proof_serializer.h"

using namespace icicle;

static bool VERBOSE = true;
static int ITERS = 1;

class HashApiTest : public IcicleTestBase
{
public:
  template <typename T>
  static void randomize(T* arr, uint64_t size)
  {
    // Fill the array with random values
    uint32_t* u32_arr = (uint32_t*)arr;
    for (int i = 0; i < (size * sizeof(T) / sizeof(uint32_t)); ++i) {
      u32_arr[i] = rand_uint_32b();
    }
  }

  static std::string voidPtrToHexString(const std::byte* byteData, size_t size)
  {
    std::ostringstream hexStream;
    for (size_t i = 0; i < size; ++i) {
      // Use fully qualified names for std::hex, std::setw, and std::setfill
      hexStream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byteData[i]);
    }

    return hexStream.str();
  }
};

TEST_F(HashApiTest, Keccak512)
{
  auto config = default_hash_config();

  const std::string input = "HelloWorld! FromKeccak512";
  const std::string expected_output = "b493094fc34b23cc868b170f68b767fcd5844f51640fdce7946958aba24336007637325d567ae456"
                                      "d4c981f144031a398f37122eb476fe75a67ab85974098e9a";
  const uint64_t output_size = 64;
  auto output = std::make_unique<std::byte[]>(output_size);

  // Create Keccak-512 hash object
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Keccak512 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    auto keccak512 = Keccak512::create();
    ICICLE_CHECK(keccak512.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output);
  }
}

TEST_F(HashApiTest, Blake2s)
{
  auto config = default_hash_config();

  const std::string input = "Hello world I am blake2s";
  const std::string expected_output = "291c4b3648438cc57d1e965ee52e5572e8dc4938bc960e22d6ebe3a280aea759";

  const uint64_t output_size = 32;
  auto output = std::make_unique<std::byte[]>(output_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Blake2s test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    auto blake2s = Blake2s::create();
    ICICLE_CHECK(blake2s.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output);
  }
}

TEST_F(HashApiTest, Blake3)
{
  // TODO: Add CUDA test, same as blake2s
  auto config = default_hash_config();

  const std::string input =
    "Hello world I am blake3. This is a semi-long C++ test with a lot of characters. "
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const std::string expected_output = "4b71f2c5cb7c26da2ba67cc742228e55b66c8b64b2b250e7ccce6f7f6d17c9ae";

  const uint64_t output_size = 32;
  auto output = std::make_unique<std::byte[]>(output_size);
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Blake2s test on device=" << device;
    ICICLE_CHECK(icicle_set_device("CPU"));

    auto blake3 = Blake3::create();
    ICICLE_CHECK(blake3.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output);
  }
}

TEST_F(HashApiTest, Keccak256Batch)
{
  auto config = default_hash_config();
  config.batch = 2;
  const std::string input = "0123456789abcdef"; // this is a batch of "01234567" and "89abcdef"
  const std::string expected_output_0 = "d529b8ccadec912a5c302a7a9ef53e70c144eea6043dcea534fdbbb2d042fc31";
  const std::string expected_output_1 = "58ed472a16d883f4dec9fc40438a59b017de9a7dbaa0bbc2cc9170e94eed2337";
  const uint64_t output_size = 32;
  auto output = std::make_unique<std::byte[]>(output_size * config.batch);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "Keccak256 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create Keccak-256 hash object and hash
    auto keccak256 = Keccak256::create();
    ICICLE_CHECK(keccak256.hash(input.data(), input.size() / config.batch, config, output.get()));
    // Convert the output do a hex string and compare to expected output string
    std::string output_as_str = voidPtrToHexString(output.get(), output_size);
    ASSERT_EQ(output_as_str, expected_output_0);
    output_as_str = voidPtrToHexString(output.get() + output_size, output_size);
    ASSERT_EQ(output_as_str, expected_output_1);
  }
}

TEST_F(HashApiTest, KeccakLarge)
{
  auto config = default_hash_config();
  config.batch = 1 << 13;
  const unsigned chunk_size = 1 << 11; // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto keccakCPU = Keccak256::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(keccakCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU Keccak large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto keccakMainDev = Keccak256::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(keccakMainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev Keccak large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(keccakMainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev Keccak large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, KeccakLargeUglySize)
{
  auto config = default_hash_config();
  config.batch = 1 << 12;
  const unsigned chunk_size = 11 + (1 << 11); // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto keccakCPU = Keccak256::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(keccakCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU Keccak large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto keccakMainDev = Keccak256::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(keccakMainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev Keccak large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(keccakMainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev Keccak large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Sha3Large)
{
  auto config = default_hash_config();
  config.batch = 1 << 13;
  const unsigned chunk_size = 1 << 11; // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto Sha3CPU = Sha3_256::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(Sha3CPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU Sha3 large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto Sha3MainDev = Sha3_256::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(Sha3MainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev Keccak large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  // ICICLE_CHECK(keccakMainDev.hash(d_input, chunk_size, config, d_output));
  ICICLE_CHECK(Sha3MainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev Sha3 large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Sha3LargeUgly)
{
  auto config = default_hash_config();
  config.batch = 1 << 12;
  const unsigned chunk_size = 11 + (1 << 11); // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto Sha3CPU = Sha3_256::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(Sha3CPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU Sha3 large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto Sha3MainDev = Sha3_256::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(Sha3MainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev Keccak large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  // ICICLE_CHECK(keccakMainDev.hash(d_input, chunk_size, config, d_output));
  ICICLE_CHECK(Sha3MainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev Sha3 large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Blake2sLarge)
{
  auto config = default_hash_config();
  config.batch = 1 << 13;
  const unsigned chunk_size = 1 << 11; // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto blake2sCPU = Blake2s::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(blake2sCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU blake2s large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto blake2sMainDev = Blake2s::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(blake2sMainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev blake2s large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(blake2sMainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev blake2s large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Blake2sLargeUgly)
{
  auto config = default_hash_config();
  config.batch = 1 << 12;
  const unsigned chunk_size = 11 + (1 << 11); // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto blake2sCPU = Blake2s::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(blake2sCPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU blake2s large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto blake2sMainDev = Blake2s::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(blake2sMainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev blake2s large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(blake2sMainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev blake2s large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Blake3Large)
{
  auto config = default_hash_config();
  config.batch = 1 << 13;
  ConfigExtension ext;
  ext.set(CpuBackendConfig::CPU_NOF_THREADS, 0); // 0 means autoselect
  config.ext = &ext;
  const unsigned chunk_size = 1 << 11; // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto blake3CPU = Blake3::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(blake3CPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU blake3 large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto blake3MainDev = Blake3::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(blake3MainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev blake3 large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(blake3MainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev blake3 large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, Blake3LargeUgly)
{
  auto config = default_hash_config();
  config.batch = 1 << 12;
  ConfigExtension ext;
  ext.set(CpuBackendConfig::CPU_NOF_THREADS, 0); // 0 means autoselect
  config.ext = &ext;
  const unsigned chunk_size = 11 + (1 << 11); // 2KB chunks
  const unsigned total_size = chunk_size * config.batch;
  auto input = std::make_unique<std::byte[]>(total_size);
  randomize((uint64_t*)input.get(), total_size / sizeof(uint64_t));

  const uint64_t output_size = 32;
  auto output_main = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_main_case_2 = std::make_unique<std::byte[]>(output_size * config.batch);
  auto output_ref = std::make_unique<std::byte[]>(output_size * config.batch);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::reference_device()));
  auto blake3CPU = Blake3::create();
  START_TIMER(cpu_timer);
  ICICLE_CHECK(blake3CPU.hash(input.get(), chunk_size, config, output_ref.get()));
  END_TIMER(cpu_timer, "CPU blake3 large time", true);

  ICICLE_CHECK(icicle_set_device(IcicleTestBase::main_device()));
  auto blake3MainDev = Blake3::create();

  // test with host memory
  START_TIMER(mainDev_timer);
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  ICICLE_CHECK(blake3MainDev.hash(input.get(), chunk_size, config, output_main.get()));
  END_TIMER(mainDev_timer, "MainDev blake3 large time (on host memory)", true);
  ASSERT_EQ(0, memcmp(output_main.get(), output_ref.get(), output_size * config.batch));

  // test with device memory
  std::byte *d_input = nullptr, *d_output = nullptr;
  ICICLE_CHECK(icicle_malloc((void**)&d_input, total_size));
  ICICLE_CHECK(icicle_malloc((void**)&d_output, output_size * config.batch));
  ICICLE_CHECK(icicle_copy(d_input, input.get(), total_size));
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(mainDev_timer_device_mem);
  ICICLE_CHECK(blake3MainDev.hash(d_input, chunk_size, config, d_output));
  END_TIMER(mainDev_timer_device_mem, "MainDev blake3 large time (on device memory)", true);
  ICICLE_CHECK(icicle_copy(output_main_case_2.get(), d_output, output_size * config.batch));
  ASSERT_EQ(0, memcmp(output_main_case_2.get(), output_ref.get(), output_size * config.batch));

  ICICLE_CHECK(icicle_free(d_input));
  ICICLE_CHECK(icicle_free(d_output));
}

TEST_F(HashApiTest, sha3)
{
  auto config = default_hash_config();

  const std::string input = "I am SHA3";
  const uint64_t output_size = 64;
  auto output = std::make_unique<std::byte[]>(output_size);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_DEBUG << "SHA3 test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    // sha3-256
    auto sha3_256 = Sha3_256::create();
    ICICLE_CHECK(sha3_256.hash(input.data(), input.size() / config.batch, config, output.get()));
    const std::string expected_output_sha_256 = "b45ee6bc2e599daf8ffd1fd952c32f58e6a7046300331b2321b927327a9affcf";
    std::string output_as_str = voidPtrToHexString(output.get(), 32);
    ASSERT_EQ(output_as_str, expected_output_sha_256);
    // sha3-512
    auto sha3_512 = Sha3_512::create();
    ICICLE_CHECK(sha3_512.hash(input.data(), input.size() / config.batch, config, output.get()));
    const std::string expected_output_sha_512 =
      "50b0cf05a243907301a10a1c14b4750a8fdbd1f8ef818624dff2f4e83901c9f8e8de84a2"
      "410d45c968b9307dfd9a4da58768e0d1f5594511b31b7274cfc04280";
    output_as_str = voidPtrToHexString(output.get(), 64);
    ASSERT_EQ(output_as_str, expected_output_sha_512);
  }
}

/****************************** Merkle **********************************/

/**
 * @brief Builds tree in a straight-forward single-threaded manner and compares the result with Icicle's calculation.
 * @param tree - Merkle tree to test (Already built).
 * @param input_size - Size of input in bytes.
 * @param leaf_size - Size of each leaf in the input below.
 * @param inputs - Input as a byte array.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @param config - Configuration of the merkle tree given above, to be used when building the reference.
 * @return True if the tree's calculations (icicle and test) match.
 */
bool is_valid_tree(
  const MerkleTree& tree,
  int input_size,
  int leaf_size,
  const std::byte* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config)
{
  std::vector<std::byte> input_vec(input_size);
  memcpy(input_vec.data(), inputs, input_size);

  int nof_hashes = 1;
  for (int i = hashes.size() - 2; i >= 0; i--) {
    nof_hashes *= hashes[i + 1].default_input_chunk_size() / hashes[i].output_size();
  }
  int tree_input_size = nof_hashes * hashes[0].default_input_chunk_size();

  ICICLE_ASSERT((config.padding_policy != PaddingPolicy::None) || (input_size == tree_input_size))
    << "Leaves size (" << (input_size / leaf_size) << ") is smaller than tree size (" << (tree_input_size / leaf_size)
    << ") while Padding policy is None\n";

  if (tree_input_size > input_size) {
    input_vec.resize(tree_input_size);
    if (config.padding_policy == PaddingPolicy::LastValue) {
      ICICLE_ASSERT(tree_input_size % leaf_size == 0)
        << "Leaf size (" << leaf_size << ") must divide tree size (" << tree_input_size << ")";
      std::vector<std::byte> last_leaf(leaf_size);
      memcpy(last_leaf.data(), inputs + input_size - leaf_size, leaf_size);
      int nof_leaves_in_padding = (tree_input_size - input_size) / leaf_size;
      for (int i = 0; i < nof_leaves_in_padding; i++) {
        memcpy(input_vec.data() + input_size + i * leaf_size, last_leaf.data(), leaf_size);
      }
    }
  }

  int max_layer_size_bytes = input_vec.size();
  int input_size_temp = input_vec.size();
  int output_size_temp = 1;

  for (auto& layer_hash : hashes) {
    output_size_temp = input_size_temp * layer_hash.output_size() / layer_hash.default_input_chunk_size();
    if (output_size_temp > max_layer_size_bytes) { max_layer_size_bytes = output_size_temp; }

    input_size_temp = output_size_temp;
  }

  input_size_temp = input_vec.size();
  int output_size = input_size * hashes[0].output_size() / hashes[0].default_input_chunk_size();
  auto layer_in = std::make_unique<std::byte[]>(max_layer_size_bytes);
  auto layer_out = std::make_unique<std::byte[]>(max_layer_size_bytes);
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  memcpy(layer_in.get(), input_vec.data(), input_size_temp);

  int side_inputs_offset = 0;
  int lidx = 0;
  for (auto& layer_hash : hashes) {
    output_size = input_size_temp * layer_hash.output_size() / layer_hash.default_input_chunk_size();
    const int nof_hashes = input_size_temp / layer_hash.default_input_chunk_size();

    auto config = default_hash_config();
    config.batch = nof_hashes;
    layer_hash.hash(layer_in.get(), layer_hash.default_input_chunk_size(), config, layer_out.get());

    // copy output outputs to inputs before moving to the next layer
    memcpy(layer_in.get(), layer_out.get(), output_size);
    input_size_temp = output_size;
  }

  // Compare computed root with the tree's root
  auto [root, root_size] = tree.get_merkle_root();
  for (int i = 0; i < root_size; i++) {
    if (root[i] != layer_out[i]) { return false; }
  }
  return true;
}

/**
 * @brief Wrapper to the non-template version of is_valid_tree above, allowing to insert different types of arrays as
 * inputs. Builds tree in a straight-forward single-threaded manner and compares the result with Icicle's calculation.
 * @param tree - Merkle tree to test  (Already built).
 * @param input_size - Size of input in bytes.
 * @param inputs - Input as a byte array.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @param config - - Configuration of the merkle tree given above, to be used when building the reference.
 * @return True if the tree's calculations (icicle and test) match.
 */
template <typename T>
bool is_valid_tree(
  const MerkleTree& tree,
  int nof_inputs,
  const T* inputs,
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config,
  unsigned explicit_leaf_size = 1)
{
  return is_valid_tree(
    tree, nof_inputs * sizeof(T), sizeof(T) * explicit_leaf_size, reinterpret_cast<const std::byte*>(inputs), hashes,
    config);
}

/**
 * @brief Function used by the HashApiTest to test the various Merkle trees defined in the tests below. Checks
 * validity of the tree construction, and correctness/incorrectness of valid/invalid proofs generated by the tree.
 * @param hashes - Vector of hashes of each layer in the tree above.
 * @param config - Merkle tree config (Mostly irrelevant for cpu tests).
 * @param output_store_min_layer - Store layer parameter for the Merkle tree builder.
 * @param nof_leaves - Size of the T leaves array.
 * @param leaves - Aforementioned leaves array.
 * @param explict_leaf_size_in_bytes - Optional. Size of each leaf element in case that leaves is given as a byte array.
 * @note Test will fail if this value isn't default (1) and T != std::byte
 * @param partial_leaves_size - size of input leaves in bytes in case not a whole number of leaves is given
 * @note Test will fail if this value is set (not 0) and T != std::byte, or if it is larger than leaves size.
 */
template <typename T>
void test_merkle_tree(
  const std::vector<Hash>& hashes,
  const MerkleTreeConfig& config,
  const int output_store_min_layer,
  int nof_leaves,
  const T* leaves,
  unsigned explict_leaf_size_in_bytes = 1,
  unsigned partial_leaves_size = 0)
{
  ASSERT_TRUE((explict_leaf_size_in_bytes == 1 || std::is_same<T, std::byte>::value))
    << "Explicit leaf size should only be given when the given leaves array is a bytes array.";

  const unsigned leaf_size = explict_leaf_size_in_bytes > 1 ? explict_leaf_size_in_bytes : sizeof(T);
  const unsigned leaves_size = nof_leaves * leaf_size;

  ASSERT_TRUE((partial_leaves_size == 0 || std::is_same<T, std::byte>::value) && (partial_leaves_size < leaves_size));

  T* device_leaves;
  if (config.is_leaves_on_device) {
    ICICLE_CHECK(icicle_malloc((void**)&device_leaves, leaves_size));
    ICICLE_CHECK(icicle_copy(device_leaves, leaves, leaves_size));
  }
  const T* leaves4tree = config.is_leaves_on_device ? device_leaves : leaves;

  auto prover_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);
  auto verifier_tree = MerkleTree::create(hashes, leaf_size);

  // assert that incorrect size fails
  if (config.padding_policy == PaddingPolicy::None) {
    ASSERT_NE(
      prover_tree.build(leaves4tree, nof_leaves * explict_leaf_size_in_bytes - 1, config), eIcicleError::SUCCESS);
    ASSERT_NE(
      prover_tree.build(leaves4tree, nof_leaves * explict_leaf_size_in_bytes + 1, config), eIcicleError::SUCCESS);
  }
  // build tree
  START_TIMER(MerkleTree_build)
  ICICLE_CHECK(prover_tree.build(
    leaves4tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, config));
  // END_TIMER(MerkleTree_build, "Merkle Tree build time", true)

  ASSERT_TRUE(is_valid_tree<T>(
    prover_tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, leaves, hashes,
    config, explict_leaf_size_in_bytes))
    << "Tree wasn't built correctly.";

  // Create wrong input leaves by taking the original input and swapping some leaves by random values
  auto wrong_leaves = std::make_unique<T[]>(nof_leaves * explict_leaf_size_in_bytes);
  memcpy(wrong_leaves.get(), leaves, nof_leaves * explict_leaf_size_in_bytes);
  const uint64_t nof_indices_modified = 5;
  unsigned int wrong_indices[nof_indices_modified];
  HashApiTest::randomize(wrong_indices, nof_indices_modified);
  for (int i = 0; i < nof_indices_modified; i++) {
    int wrong_byte_index = wrong_indices[i] % (partial_leaves_size ? partial_leaves_size : nof_leaves * leaf_size);

    uint8_t* wrong_leaves_byte_ptr = reinterpret_cast<uint8_t*>(wrong_leaves.get());

    uint8_t new_worng_val;
    do {
      new_worng_val = rand_uint_32b(0, UINT8_MAX);
    } while (new_worng_val == wrong_leaves_byte_ptr[wrong_byte_index]);

    wrong_leaves_byte_ptr[wrong_byte_index] = new_worng_val;

    int wrong_leaf_idx = wrong_byte_index / leaf_size;
    ICICLE_LOG_VERBOSE << "Wrong input is modified at leaf " << wrong_leaf_idx << " (modified at byte "
                       << wrong_byte_index % leaf_size << ")";
  }

  T* wrong_device_leaves;
  if (config.is_leaves_on_device) {
    ICICLE_CHECK(icicle_malloc((void**)&wrong_device_leaves, leaves_size));
    ICICLE_CHECK(icicle_copy(wrong_device_leaves, wrong_leaves.get(), leaves_size));
  }
  const T* wrong_leaves4tree = config.is_leaves_on_device ? wrong_device_leaves : wrong_leaves.get();

  // Test the paths at the random indices (Both that the original input is valid and the modified input isn't)
  for (int i = 0; i < nof_indices_modified; i++) {
    // int leaf_idx = (wrong_indices[i] % (nof_leaves * leaf_size)) / leaf_size;
    int leaf_idx =
      (wrong_indices[i] % (partial_leaves_size ? partial_leaves_size : nof_leaves * leaf_size)) / leaf_size;
    ICICLE_LOG_VERBOSE << "Checking proof of index " << leaf_idx << " (Byte idx "
                       << (wrong_indices[i] % (nof_leaves * leaf_size)) << ")";

    // get root and merkle-path for a leaf
    auto [root, root_size] = prover_tree.get_merkle_root();
    MerkleProof merkle_proof{};
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      leaves4tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, leaf_idx, false,
      config, merkle_proof));

    // Test valid proof
    bool verification_valid = false;
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid) << "Proof of valid inputs at index " << leaf_idx
                                    << " is invalid (And should be valid).";

    // Test invalid proof (By modifying random data in the leaves)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      wrong_leaves4tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, leaf_idx,
      false, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid) << "Proof of invalid inputs at index " << leaf_idx
                                     << " is valid (And should be invalid).";

    // Same for pruned proof
    verification_valid = false;
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      leaves4tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, leaf_idx, true,
      config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_TRUE(verification_valid) << "Pruned proof of valid inputs at index " << leaf_idx
                                    << " is invalid (And should be valid).";

    // Test invalid proof (By modifying random data in the leaves)
    verification_valid = true;
    ICICLE_CHECK(prover_tree.get_merkle_proof(
      wrong_leaves4tree, partial_leaves_size ? partial_leaves_size : nof_leaves * explict_leaf_size_in_bytes, leaf_idx,
      true, config, merkle_proof));
    ICICLE_CHECK(verifier_tree.verify(merkle_proof, verification_valid));
    ASSERT_FALSE(verification_valid) << "Pruned proof of invalid inputs at index " << leaf_idx
                                     << " is valid (And should be invalid).";

    // Serialize the proof
    size_t serialized_proof_size;
    ICICLE_CHECK(BinarySerializer<MerkleProof>::serialized_size(merkle_proof, serialized_proof_size));
    auto serialized_proof = std::vector<std::byte>(serialized_proof_size);
    ICICLE_CHECK(
      BinarySerializer<MerkleProof>::serialize(serialized_proof.data(), serialized_proof.size(), merkle_proof));
    // Deserialize the proof
    MerkleProof deserialized_proof;
    ICICLE_CHECK(
      BinarySerializer<MerkleProof>::deserialize(serialized_proof.data(), serialized_proof.size(), deserialized_proof));

    // Compare the original and deserialized proofs
    // Compare pruned
    ASSERT_EQ(merkle_proof.is_pruned(), deserialized_proof.is_pruned());

    // Compare paths
    auto [orig_path_ptr, orig_path_size] = merkle_proof.get_path();
    auto [deser_path_ptr, deser_path_size] = deserialized_proof.get_path();
    ASSERT_EQ(orig_path_size, deser_path_size);
    std::vector<std::byte> orig_path_vec(orig_path_ptr, orig_path_ptr + orig_path_size);
    std::vector<std::byte> deser_path_vec(deser_path_ptr, deser_path_ptr + deser_path_size);
    ASSERT_EQ(orig_path_vec, deser_path_vec);

    // Compare leaves
    auto [orig_leaf_ptr, orig_leaf_size, orig_leaf_idx] = merkle_proof.get_leaf();
    auto [deser_leaf_ptr, deser_leaf_size, deser_leaf_idx] = deserialized_proof.get_leaf();
    ASSERT_EQ(orig_leaf_size, deser_leaf_size);
    ASSERT_EQ(orig_leaf_idx, deser_leaf_idx);
    std::vector<std::byte> orig_leaf_vec(orig_leaf_ptr, orig_leaf_ptr + orig_leaf_size);
    std::vector<std::byte> deser_leaf_vec(deser_leaf_ptr, deser_leaf_ptr + deser_leaf_size);
    ASSERT_EQ(orig_leaf_vec, deser_leaf_vec);

    // Compare roots
    auto [orig_root_ptr, orig_root_size] = merkle_proof.get_root();
    auto [deser_root_ptr, deser_root_size] = deserialized_proof.get_root();
    ASSERT_EQ(orig_root_size, deser_root_size);
    std::vector<std::byte> orig_root_vec(orig_root_ptr, orig_root_ptr + orig_root_size);
    std::vector<std::byte> deser_root_vec(deser_root_ptr, deser_root_ptr + deser_root_size);
    ASSERT_EQ(orig_root_vec, deser_root_vec);
  }

  if (config.is_leaves_on_device) {
    ICICLE_CHECK(icicle_free(device_leaves));
    ICICLE_CHECK(icicle_free(wrong_device_leaves));
  }
}

TEST_F(HashApiTest, MerkleTreeZeroPadding)
{
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeZeroPadding test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    constexpr int leaf_size = 250;
    constexpr int nof_leaves = 100;
    const std::vector<int> test_cases_nof_input_leaves = {1,  8,  16, 17,
                                                          32, 70, 99, 100}; // those cases will be tested with padding
    constexpr int input_size = nof_leaves * leaf_size;
    std::byte leaves[input_size];
    randomize(leaves, input_size);

    // define the merkle tree
    auto layer0_hash = Blake2s::create(2 * leaf_size);
    auto layer1_hash = Blake2s::create(32);
    auto layer2_hash = Blake2s::create(5 * 32);
    auto layer3_hash = Blake2s::create(10 * 32);

    std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
    int output_store_min_layer = 0;

    auto config = default_merkle_tree_config();
    config.padding_policy = PaddingPolicy::ZeroPadding;

    // test various cases of missing leaves in input, requiring padding.
    for (auto nof_leaves_iter : test_cases_nof_input_leaves) {
      test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves_iter, leaves, leaf_size);
    }

    // Test tree with a fractional amount of leaves as input
    const unsigned partial_leaf_bytes = (rand() % (leaf_size - 1)) + 1;
    const unsigned nof_whole_leaves = rand() % nof_leaves;
    const unsigned partial_leaves_size_in_bytes = nof_whole_leaves * leaf_size + partial_leaf_bytes;
    ICICLE_LOG_VERBOSE << "Partial leaves byte size: " << partial_leaves_size_in_bytes;
    test_merkle_tree(
      hashes, config, output_store_min_layer, nof_leaves, leaves, leaf_size, partial_leaves_size_in_bytes);
  }
}

TEST_F(HashApiTest, MerkleTreeZeroPaddingLeavesOnDevice)
{
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeZeroPaddingLeavesOnDevice test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    constexpr int leaf_size = 320; // TODO: should be 250 after fix
    constexpr int nof_leaves = 100;
    const std::vector<int> test_cases_nof_input_leaves = {1,  8,  16, 17,
                                                          32, 70, 99, 100}; // those cases will be tested with padding
    constexpr int input_size = nof_leaves * leaf_size;
    std::byte leaves[input_size];
    randomize(leaves, input_size);

    // define the merkle tree
    auto layer0_hash = Keccak256::create(leaf_size); // TODO: should be 2 * leaf_size after fix
    auto layer1_hash = Keccak256::create(2 * 32);    // TODO: should be 32 after fix
    auto layer2_hash = Keccak256::create(5 * 32);
    auto layer3_hash = Keccak256::create(10 * 32);

    std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
    int output_store_min_layer = 0;

    auto config = default_merkle_tree_config();
    config.is_leaves_on_device = true;
    config.padding_policy = PaddingPolicy::ZeroPadding;

    // test various cases of missing leaves in input, requiring padding.
    for (auto nof_leaves_iter : test_cases_nof_input_leaves) {
      test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves_iter, leaves, leaf_size);
    }

    // Test tree with a fractional amount of leaves as input
    const unsigned partial_leaf_bytes = (rand() % (leaf_size - 1)) + 1;
    const unsigned nof_whole_leaves = rand() % nof_leaves;
    const unsigned partial_leaves_size_in_bytes = nof_whole_leaves * leaf_size + partial_leaf_bytes;
    ICICLE_LOG_VERBOSE << "Partial leaves byte size: " << partial_leaves_size_in_bytes;
    test_merkle_tree(
      hashes, config, output_store_min_layer, nof_leaves, leaves, leaf_size, partial_leaves_size_in_bytes);
  }
}

TEST_F(HashApiTest, MerkleTreeLastValuePadding)
{
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeLastValuePadding test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    constexpr int leaf_size = 320;
    constexpr int nof_leaves = 100;
    const std::vector<int> test_cases_nof_input_leaves = {1,  8,  16, 17,
                                                          32, 70, 99, 100}; // those cases will be tested with padding
    constexpr int input_size = nof_leaves * leaf_size;
    std::byte leaves[input_size];
    randomize(leaves, input_size);

    // define the merkle tree
    auto layer0_hash = Keccak256::create(leaf_size);
    auto layer1_hash = Keccak256::create(2 * 32);
    auto layer2_hash = Keccak256::create(5 * 32);
    auto layer3_hash = Keccak256::create(10 * 32);

    std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
    int output_store_min_layer = 0;

    auto config = default_merkle_tree_config();
    config.padding_policy = PaddingPolicy::LastValue;

    // test various cases of missing leaves in input, requiring padding.
    for (auto nof_leaves_iter : test_cases_nof_input_leaves) {
      test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves_iter, leaves, leaf_size);
    }

    // Test that leaf_size divides input size for this kind of padding
    auto prover_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);
    ASSERT_EQ(prover_tree.build(leaves, (leaf_size - 1) * nof_leaves, config), eIcicleError::INVALID_ARGUMENT);
  }
}

TEST_F(HashApiTest, MerkleTreeLastValuePaddingLeavesOnDevice)
{
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeLastValuePaddingLeavesOnDevice test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    constexpr int leaf_size = 320;
    constexpr int nof_leaves = 100;
    const std::vector<int> test_cases_nof_input_leaves = {1,  8,  16, 17,
                                                          32, 70, 99, 100}; // those cases will be tested with padding
    constexpr int input_size = nof_leaves * leaf_size;
    std::vector<std::byte> leaves(input_size);
    randomize(leaves.data(), input_size);

    std::byte* d_leaves;
    ICICLE_CHECK(icicle_malloc((void**)&d_leaves, input_size));
    ICICLE_CHECK(icicle_copy(d_leaves, leaves.data(), input_size));

    // define the merkle tree
    auto layer0_hash = Keccak256::create(leaf_size);
    auto layer1_hash = Keccak256::create(2 * 32);
    auto layer2_hash = Keccak256::create(5 * 32);
    auto layer3_hash = Keccak256::create(10 * 32);

    std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
    int output_store_min_layer = 0;

    auto config = default_merkle_tree_config();
    config.is_leaves_on_device = true;
    config.padding_policy = PaddingPolicy::LastValue;

    // test various cases of missing leaves in input, requiring padding.
    for (auto nof_leaves_iter : test_cases_nof_input_leaves) {
      test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves_iter, leaves.data(), leaf_size);
    }

    // Test that leaf_size divides input size for this kind of padding
    auto prover_tree = MerkleTree::create(hashes, leaf_size, output_store_min_layer);
    ASSERT_EQ(prover_tree.build(d_leaves, (leaf_size - 1) * nof_leaves, config), eIcicleError::INVALID_ARGUMENT);
    ICICLE_CHECK(icicle_free(d_leaves));
  }
}

TEST_F(HashApiTest, MerkleTreeMixMediumSize)
{
  const uint32_t leaf_size = sizeof(uint32_t);
  const uint32_t total_input_size = (1 << 20);
  const uint32_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<uint32_t[]>(nof_leaves);
  randomize(leaves.get(), nof_leaves);

  auto leaves_alternative = std::make_unique<uint32_t[]>(nof_leaves);
  randomize(leaves_alternative.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeMixMediumSize on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // define the merkle tree
    auto config = default_merkle_tree_config();

    auto layer0_hash = Keccak256::create(1 << 10); // hash every 1KB to 32B -> layer outputs 32KB (for 1MB input)
    auto layer1_hash = Blake2s::create(32 * 2);    // arity-2: 32KB -> 16KB
    auto layer2_hash = Sha3_512::create(32 * 4);   // arity-4: 16KB -> 8KB (note output is 64B per hash in this layer)
    auto layer3_hash = Blake2s::create(64 * 4);    // arity-4: 8KB -> 1KB
    auto layer4_hash = Keccak512::create(32 * 32); // arity-32: 1KB -> 64B

    const std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash, layer4_hash};

    const int output_store_min_layer = rand_uint_32b(0, hashes.size() - 1);
    ICICLE_LOG_DEBUG << "Min store layer:\t" << output_store_min_layer;

    test_merkle_tree(hashes, config, output_store_min_layer, nof_leaves, leaves.get());
  }
}

TEST_F(HashApiTest, MerkleTreeDevicePartialTree)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 8) * leaf_size;
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceCaps on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    auto config = default_merkle_tree_config();

    // Test with different values of output_store_min_layer
    test_merkle_tree<std::byte>(
      hashes, config, /*output_store_min_layer=*/0, nof_leaves, leaves.get(),
      /*explicit_leaf_size_in_bytes=*/leaf_size);
    test_merkle_tree<std::byte>(
      hashes, config, /*output_store_min_layer=*/2, nof_leaves, leaves.get(),
      /*explicit_leaf_size_in_bytes=*/leaf_size);
    test_merkle_tree<std::byte>(
      hashes, config, /*output_store_min_layer=*/4, nof_leaves, leaves.get(),
      /*explicit_leaf_size_in_bytes=*/leaf_size);
  }
}

TEST_F(HashApiTest, MerkleTreeLeavesOnDeviceTreeOnHost)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 3) * leaf_size;
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceBig on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));
    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    // Specify the config for the test function below
    auto config = default_merkle_tree_config();
    config.is_tree_on_device = false;
    config.is_leaves_on_device = true;

    test_merkle_tree<std::byte>(hashes, config, 0, nof_leaves, leaves.get(), /*explicit_leaf_size_in_bytes=*/leaf_size);
  }
}

TEST_F(HashApiTest, MerkleTreeLarge)
{
  const uint64_t leaf_size = 32;
  const uint64_t total_input_size = (1 << 28);
  const uint64_t nof_leaves = total_input_size / leaf_size;
  auto leaves = std::make_unique<std::byte[]>(total_input_size);
  randomize(leaves.get(), total_input_size);

  for (auto&& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceBig on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    // Create a Keccak256 hasher with an arity of 2: every 64B -> 32B
    auto layer_hash = Keccak256::create(32 * 2);
    // Calculate the tree height (log2 of the number of leaves for a binary tree)
    const int tree_height = static_cast<int>(std::log2(nof_leaves));
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(tree_height, layer_hash);

    // Specify the config for the test function below
    auto config = default_merkle_tree_config();
    config.is_leaves_on_device = true;
    auto prover_tree = MerkleTree::create(hashes, leaf_size);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size);

    test_merkle_tree<std::byte>(hashes, config, 0, nof_leaves, leaves.get(), /*explicit_leaf_size_in_bytes=*/leaf_size);
  }
}

#ifdef POSEIDON
// bn254 p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

  #include "icicle/fields/field_config.h"
  #include "icicle/hash/poseidon_constants/constants/bn254_poseidon.h"

using namespace field_config;
// using namespace poseidon_constants_bn254;

  #include "icicle/hash/poseidon.h"

TEST_F(HashApiTest, poseidon_12_single_hash)
{
  const unsigned t = 12;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
}

// TEST_F(HashApiTest, poseidon_3_single_hash_domain_tag)
// {
//   const unsigned  t                   = 2;
//   const unsigned  default_input_size      = 2;
//   const bool      use_domain_tag           = true;
//   scalar_t        domain_tag_value        = scalar_t::from(7);
//   const bool      use_all_zeroes_padding  = true;
//   auto            config                  = default_hash_config();

//   auto input = std::make_unique<scalar_t[]>(t);
//   scalar_t::rand_host_many(input.get(), t);

//   config.batch = 1;

//   auto run =
//     [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
//       Device dev = {dev_type, 0};
//       icicle_set_device(dev);

//       std::ostringstream oss;
//       oss << dev_type << " " << msg;

//       auto poseidon = Poseidon::create<scalar_t>(t);

//       START_TIMER(POSEIDON_sync)
//       for (int i = 0; i < iters; ++i) {
//         ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
//       }
//       END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
//     };

//   auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
//   auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

//   run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
//   run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

//   ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
// }

TEST_F(HashApiTest, poseidon_3_single_hash)
{
  const unsigned t = 3;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t);
  scalar_t::rand_host_many(input.get(), t);
  config.batch = 1;

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon_3_batch_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();

  config.batch = 1 << 10;
  config.batch = 6;
  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon_3_batch_with_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::rand_host();

  config.batch = 1 << 10;
  auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);
  scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon = Poseidon::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon.hash(input.get(), t - 1, config, out));
    }
    END_TIMER(POSEIDON_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon_tree)
{
  const uint64_t t = 9;
  const uint64_t nof_layers = 4;
  uint64_t nof_leaves = 1;
  for (int i = 0; i < nof_layers; i++) {
    nof_leaves *= t;
  }
  auto leaves = std::make_unique<scalar_t[]>(nof_leaves);
  const uint64_t leaf_size = sizeof(scalar_t);
  const uint64_t total_input_size = nof_leaves * leaf_size;

  scalar_t::rand_host_many(leaves.get(), nof_leaves);

  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "MerkleTreeDeviceBig on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    // Create relevant hash to compose the tree
    auto layer_hash = Poseidon::create<scalar_t>(t);
    // Create a vector of `Hash` objects, all initialized with the same `layer_hash`
    std::vector<Hash> hashes(nof_layers, layer_hash);

    // Specify the config for the test function below
    auto config = default_merkle_tree_config();
    config.is_leaves_on_device = true;
    auto prover_tree = MerkleTree::create(hashes, leaf_size);
    auto verifier_tree = MerkleTree::create(hashes, leaf_size);

    // Cast to bytes to conform with wrong leaves manipulation inside test_merkle_tree
    test_merkle_tree(hashes, config, 0, nof_leaves, leaves.get());
  }
}

#endif // POSEIDON

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

#ifdef POSEIDON2
  // bn254: p = 0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570
  #include "icicle/fields/field_config.h"
using namespace field_config;
TEST_F(HashApiTest, poseidon2_3_single_hasher)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  config.batch = 1;

  auto input = std::make_unique<scalar_t[]>(config.batch * t);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_single_hasher

// Sponge, chain of 2 hashers, no padding, no domain tag.
TEST_F(HashApiTest, poseidon2_4_sponge_2_hashers_without_dt)
{
  const unsigned t = 4;
  auto config = default_hash_config();
  int nof_hashers = 2;
  int nof_inputs = 1 + nof_hashers * (t - 1);

  auto input = std::make_unique<scalar_t[]>(nof_inputs);
  scalar_t::rand_host_many(input.get(), nof_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch); // config.batch = 1 here.
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  std::string main_device = IcicleTestBase::main_device();

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  // std::cout << "Output CPU  = " << output_cpu[0] << std::endl;
  // std::cout << "Output CUDA = " << output_mainDev[0] << std::endl;
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_4_sponge_2_hashers_without_dt

// Test used to generate expected result of any hasher ccording to the parameters inside.
TEST_F(HashApiTest, poseidon2_3_gen_hasher_expected_result_cpu_only)
{
  auto config = default_hash_config();
  config.batch = 1;

  const unsigned t = 4;
  auto input = std::make_unique<scalar_t[]>(t);
  // Set the inputs as needed.
  for (int i = 0; i < config.batch; i++) {
    for (int j = 0; j < t; j++) {
      input[i * t + j] = scalar_t::from(j + 4);
    }
  }
  for (int i = 0; i < config.batch * t; i++) {
    std::cout << "Input = " << input[i] << std::endl;
  }

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  }; //

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  for (int i = 0; i < config.batch; i++) {
    std::cout << "Output = " << output_cpu[i] << std::endl;
  }
} // poseidon2_3_gen_hasher_expected_result_cpu_only

TEST_F(HashApiTest, poseidon2_non_sponge_all_included_test)
{
  scalar_t domain_tag;
  std::unique_ptr<scalar_t[]> outputs[2]; // Used to compare CPU and CUDA results.

  for (auto t : {2, 3, 4, 8, 12, 16, 20, 24}) {
    if (scalar_t::TLC > 1 && t > 4) continue;
    auto config = default_hash_config();
    for (auto batch_size : {1, 16, 256}) {
      config.batch = batch_size;
      outputs[0] = std::make_unique<scalar_t[]>(batch_size);
      outputs[1] = std::make_unique<scalar_t[]>(batch_size);
      for (auto use_domain_tag : {false, true}) {
        std::unique_ptr<scalar_t[]> input;
        if (use_domain_tag) {
          input = std::make_unique<scalar_t[]>(config.batch * (t - 1));
          domain_tag = scalar_t::rand_host();
          scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);
        } // No domain tag.
        else {
          input = std::make_unique<scalar_t[]>(config.batch * t);
          scalar_t::rand_host_many(input.get(), t * config.batch);
        }
        for (const auto& device : s_registered_devices) {
          icicle_set_device(device);
          auto out = std::make_unique<scalar_t[]>(config.batch);
          std::ostringstream oss;
          oss << std::string(device) << " "
              << "poseidon2";
          if (use_domain_tag) {
            auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);
            START_TIMER(POSEIDON2_sync)
            for (int i = 0; i < ITERS; ++i) {
              ICICLE_CHECK(poseidon2.hash(input.get(), t - 1, config, out.get()));
            }
            END_TIMER(POSEIDON2_sync, oss.str().c_str(), VERBOSE);
          } else { // No domain tag.
            auto poseidon2 = Poseidon2::create<scalar_t>(t);
            START_TIMER(POSEIDON2_sync)
            for (int i = 0; i < ITERS; ++i) {
              ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out.get()));
            }
            END_TIMER(POSEIDON2_sync, oss.str().c_str(), VERBOSE);
          }
          if (std::string(device) == "CUDA") {
            outputs[0] = std::move(out);
          } else {
            outputs[1] = std::move(out);
          }
        } // for (const auto& device : s_registered_devices) {
        // for (int i=0; i<config.batch; i++) {
        //   std::cout << "CPU  outputs[1][" << i << "] = " << outputs[1][i] << std::endl;
        //   std::cout << "CUDA outputs[0][" << i << "] = " << outputs[0][i] << std::endl;
        // }
        ASSERT_EQ(0, memcmp(outputs[1].get(), outputs[0].get(), config.batch * sizeof(scalar_t)));
      } // for (auto use_domain_tag : {false, true}) {
    } // for (auto batch_size : {1, 16, 256}) {
  } // for (auto t : {2, 3, 4, 8, 12, 16, 20, 24}) {
} // poseidon2_non_sponge_inclusize_test

TEST_F(HashApiTest, poseidon2_sponge_all_included_test)
{
  scalar_t domain_tag;
  std::unique_ptr<scalar_t[]> outputs[2]; // Used to compare CPU and CUDA results.

  for (auto t : {2, 3, 4, 8, 12, 16, 20, 24}) {
    if (scalar_t::TLC > 1 && t > 4) continue;
    auto config = default_hash_config();
    for (auto batch_size : {1, 2, 16}) {
      config.batch = batch_size;
      outputs[0] = std::make_unique<scalar_t[]>(batch_size);
      outputs[1] = std::make_unique<scalar_t[]>(batch_size);
      for (auto nof_hashers : {1, 2, 16, 256}) {
        for (auto padding_size : {0, t > 2 ? t - 2 : 0}) { // If padding_size == 0 then there is no padding.
          for (auto use_domain_tag : {false, true}) {
            std::unique_ptr<scalar_t[]> input;
            if (use_domain_tag) {
              input = std::make_unique<scalar_t[]>(config.batch * (nof_hashers * (t - 1) - padding_size));
              domain_tag = scalar_t::rand_host();
              scalar_t::rand_host_many(input.get(), config.batch * (nof_hashers * (t - 1) - padding_size));
            } else {
              input = std::make_unique<scalar_t[]>(config.batch * (1 + nof_hashers * (t - 1) - padding_size));
              scalar_t::rand_host_many(input.get(), config.batch * (1 + nof_hashers * (t - 1) - padding_size));
            }
            for (const auto& device : s_registered_devices) {
              icicle_set_device(device);
              auto out = std::make_unique<scalar_t[]>(config.batch);
              std::ostringstream oss;
              oss << std::string(device) << " "
                  << "poseidon2";
              if (use_domain_tag) {
                auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);
                START_TIMER(POSEIDON2_sync)
                std::cout << "use_domain_tag, Device = " << device << std::endl;
                for (int i = 0; i < ITERS; ++i) {
                  ICICLE_CHECK(poseidon2.hash(input.get(), nof_hashers * (t - 1) - padding_size, config, out.get()));
                }
                END_TIMER(POSEIDON2_sync, oss.str().c_str(), VERBOSE);
              } // No domain tag.
              else {
                auto poseidon2 = Poseidon2::create<scalar_t>(t);
                START_TIMER(POSEIDON2_sync)
                std::cout << "Not use_domain_tag, Device = " << device << std::endl;
                for (int i = 0; i < ITERS; ++i) {
                  ICICLE_CHECK(
                    poseidon2.hash(input.get(), 1 + nof_hashers * (t - 1) - padding_size, config, out.get()));
                }
                END_TIMER(POSEIDON2_sync, oss.str().c_str(), VERBOSE);
              } // Domain tag.
              if (std::string(device) == "CUDA")
                outputs[0] = std::move(out);
              else
                outputs[1] = std::move(out);
            } // for (const auto& device : s_registered_devices) {
            // for (int i=0; i<config.batch; i++) {
            //   std::cout << "CPU  outputs[1][" << i << "] = " << outputs[1][i] << std::endl;
            //   std::cout << "CUDA outputs[0][" << i << "] = " << outputs[0][i] << std::endl;
            // }
            ASSERT_EQ(0, memcmp(outputs[1].get(), outputs[0].get(), config.batch * sizeof(scalar_t)));
          } // for (auto use_domain_tag : {false, true}) {
        } // for (auto use_padding : {false, true}) {
      } // for (auto nof_hashers : {1, 2, 16, 256}) {
    } // for (auto batch_size : {1, 16, 256}) {
  } // for (auto t : {2, 3, 4, 8, 12, 16, 20, 24}) {
} // poseidon2_sponge_all_included_test

// Sponge, chain of 2 hashers, no padding, with domain tag.
TEST_F(HashApiTest, poseidon2_3_sponge_2_hashers_with_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::rand_host();
  int nof_hashers = 2;
  int nof_valid_inputs = nof_hashers * (t - 1);

  auto input = std::make_unique<scalar_t[]>(nof_valid_inputs);
  scalar_t::rand_host_many(input.get(), nof_valid_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_valid_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch); // config.batch = 1 here.
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  // std::cout << "CPU output  = " << output_cpu[0] << std::endl;
  // std::cout << "CUDA output = " << output_mainDev[0] << std::endl;
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_sponge_2_hashers_with_dt

// Sponge, chain of 1 hasher, with padding (2 scalars 1,0), no domain tag.
TEST_F(HashApiTest, poseidon2_4_sponge_1_hasher_without_dt_with_padding)
{
  const unsigned t = 4;
  auto config = default_hash_config();
  int nof_hashers = 1;
  int nof_valid_inputs = 2;

  auto input = std::make_unique<scalar_t[]>(nof_valid_inputs);
  scalar_t::rand_host_many(input.get(), nof_valid_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);
    std::cout << "Device = " << dev << std::endl;
    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_valid_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch); // config.batch = 1 here.
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  // std::cout << "CPU output  = " << output_cpu[0] << std::endl;
  // std::cout << "CUDA output = " << output_mainDev[0] << std::endl;
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_4_sponge_1_hasher_without_dt_with_padding

// Sponge, chain of 1 hasher, with padding (2 scalars 1,0), no domain tag.
TEST_F(HashApiTest, poseidon2_3_sponge_1_hasher_without_dt_with_padding_debug)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  int nof_hashers = 1;
  int nof_valid_inputs = 2;

  auto input = std::make_unique<scalar_t[]>(nof_valid_inputs);
  scalar_t::rand_host_many(input.get(), nof_valid_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_valid_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch); // config.batch = 1 here.
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_sponge_1_hasher_without_dt_with_padding_debug

// Sponge, chain of 2 hashers, with padding (2 scalars 1,0), no domain tag.
TEST_F(HashApiTest, poseidon2_4_sponge_2_hashers_without_dt_with_padding)
{
  const unsigned t = 4;
  auto config = default_hash_config();
  int nof_hashers = 2;
  int nof_valid_inputs = 5;

  auto input = std::make_unique<scalar_t[]>(nof_valid_inputs);
  scalar_t::rand_host_many(input.get(), nof_valid_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_valid_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch); // config.batch = 1 here.
  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_4_sponge_2_hashers_without_dt_with_padding

// Test check single hash without domain tag.
TEST_F(HashApiTest, poseidon2_3_single_hash_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  config.batch = 1;

  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_single_hash_without_dt

TEST_F(HashApiTest, poseidon2_3_sponge_1K_hashers_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  int nof_hashers = 1 << 10;
  int nof_valid_inputs = 1 + nof_hashers * (t - 1);

  auto input = std::make_unique<scalar_t[]>(nof_valid_inputs);
  scalar_t::rand_host_many(input.get(), nof_valid_inputs);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), nof_valid_inputs, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(1);
  auto output_mainDev = std::make_unique<scalar_t[]>(1);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_sponge_1K_hashers_without_dt

TEST_F(HashApiTest, poseidon2_invalid_t)
{
  // Large fields do not support some t's.
  // This is testing that a correct error is returned for invalid t.
  const unsigned t = 20;
  auto config = default_hash_config();

  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  auto output = std::make_unique<scalar_t[]>(config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  const bool large_field = sizeof(scalar_t) > 8;

  for (const auto& device : s_registered_devices) {
    icicle_set_device(device);

    auto poseidon2 = Poseidon2::create<scalar_t>(t);
    auto err = poseidon2.hash(input.get(), t, config, output.get());
    if (large_field) {
      EXPECT_EQ(err, eIcicleError::INVALID_ARGUMENT);
    } else {
      EXPECT_EQ(err, eIcicleError::SUCCESS);
    }
  }
} // poseidon2_invalid_t

TEST_F(HashApiTest, poseidon2_3_single_hash_with_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::rand_host();
  config.batch = 1;

  auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);
  scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t - 1, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
}

TEST_F(HashApiTest, poseidon2_2_batch_16_with_dt)
{
  const unsigned t = 2;
  auto config = default_hash_config();
  const scalar_t domain_tag = scalar_t::from(0);
  config.batch = 1 << 4;

  auto input = std::make_unique<scalar_t[]>((t - 1) * config.batch);
  scalar_t::rand_host_many(input.get(), (t - 1) * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t, &domain_tag);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t - 1, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_2_batch_16_with_dt

TEST_F(HashApiTest, poseidon2_3_large_batch_without_dt_debug)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  config.batch = 1 << 14;

  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_large_batch_without_dt_debug

TEST_F(HashApiTest, poseidon2_3_1K_batch_without_dt)
{
  const unsigned t = 3;
  auto config = default_hash_config();
  config.batch = 1 << 10;

  auto input = std::make_unique<scalar_t[]>(t * config.batch);
  scalar_t::rand_host_many(input.get(), t * config.batch);

  auto run = [&](const std::string& dev_type, scalar_t* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    auto poseidon2 = Poseidon2::create<scalar_t>(t);

    START_TIMER(POSEIDON2_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(poseidon2.hash(input.get(), t, config, out));
    }
    END_TIMER(POSEIDON2_sync, oss.str().c_str(), measure);
  };

  auto output_cpu = std::make_unique<scalar_t[]>(config.batch);
  auto output_mainDev = std::make_unique<scalar_t[]>(config.batch);

  run(IcicleTestBase::reference_device(), output_cpu.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);
  run(IcicleTestBase::main_device(), output_mainDev.get(), VERBOSE /*=measure*/, "poseidon2", ITERS);

  ASSERT_EQ(0, memcmp(output_cpu.get(), output_mainDev.get(), config.batch * sizeof(scalar_t)));
} // poseidon2_3_1K_batch_without_dt

TEST_F(HashApiTest, poseidon2_merkle_tree)
{
  for (const auto& device : s_registered_devices) {
    ICICLE_LOG_INFO << "poseidon2_merkle_tree test on device=" << device;
    ICICLE_CHECK(icicle_set_device(device));

    constexpr int leaf_size = sizeof(scalar_t);
    constexpr int nof_leaves = 8;
    constexpr int input_size = nof_leaves * leaf_size;
    auto leaves = std::make_unique<scalar_t[]>(input_size);
    scalar_t::rand_host_many(leaves.get(), input_size);

    // define the merkle tree
    auto layer0_hash = Poseidon2::create<scalar_t>(2, nullptr, 1);
    auto layer1_hash = Poseidon2::create<scalar_t>(2);
    auto layer2_hash = Poseidon2::create<scalar_t>(2);
    auto layer3_hash = Poseidon2::create<scalar_t>(2);

    std::vector<Hash> hashes = {layer0_hash, layer1_hash, layer2_hash, layer3_hash};
    int output_store_min_layer = 0;

    auto config = default_merkle_tree_config();
    test_merkle_tree(
      hashes, config, output_store_min_layer, nof_leaves, reinterpret_cast<std::byte*>(leaves.get()), leaf_size);
  }
}

#endif // POSEIDON2

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

#ifdef SUMCHECK
/*======================= (TODO??) Move to Sumcheck test-suite =======================*/
  #include "icicle/sumcheck/sumcheck_transcript_config.h"

class SumcheckTest : public IcicleTestBase
{
public:
  // Helper function to convert a byte vector to a string for comparison
  static std::string bytes_to_string(const std::vector<std::byte>& bytes)
  {
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
};

TEST_F(SumcheckTest, InitializeWithConstChar)
{
  auto hasher = Keccak256::create();
  const char* domain_separator = "DomainLabel";
  const char* round_poly = "PolyLabel";
  const char* round_challenge = "ChallengeLabel";
  auto seed = scalar_t::rand_host();

  SumcheckTranscriptConfig config(hasher, domain_separator, round_poly, round_challenge, seed);

  EXPECT_EQ(bytes_to_string(config.get_domain_separator_label()), domain_separator);
  EXPECT_EQ(bytes_to_string(config.get_round_poly_label()), round_poly);
  EXPECT_EQ(bytes_to_string(config.get_round_challenge_label()), round_challenge);
  EXPECT_TRUE(config.is_little_endian());
  EXPECT_EQ(config.get_seed_rng(), seed);
}

TEST_F(SumcheckTest, InitializeWithByteVector)
{
  auto hasher = Keccak256::create();
  std::vector<std::byte> domain_label = {std::byte('d'), std::byte('s')};
  std::vector<std::byte> poly_label = {std::byte('p'), std::byte('l')};
  std::vector<std::byte> challenge_label = {std::byte('c'), std::byte('h')};
  auto seed = scalar_t::rand_host();

  SumcheckTranscriptConfig config(
    hasher, std::move(domain_label), std::move(poly_label), std::move(challenge_label), seed);

  EXPECT_EQ(bytes_to_string(config.get_domain_separator_label()), "ds");
  EXPECT_EQ(bytes_to_string(config.get_round_poly_label()), "pl");
  EXPECT_EQ(bytes_to_string(config.get_round_challenge_label()), "ch");
  EXPECT_TRUE(config.is_little_endian());
  EXPECT_EQ(config.get_seed_rng(), seed);
}
#endif // SUMCHECK
