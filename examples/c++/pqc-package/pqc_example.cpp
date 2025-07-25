#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>

// ICICLE PQC headers
#include "icicle/pqc/ml_kem.h"
#include "icicle/runtime.h"
#include "icicle/device.h"
#include "icicle/errors.h"

// Generate random bytes
void generate_random_bytes(std::byte* buffer, size_t size)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<std::byte>(dis(gen));
  }
}

// Helper function to format timing results with appropriate units
struct TimingResult {
  double avg_time;
  std::string avg_unit;
  double total_time;
  std::string total_unit;
  double ops_per_sec;
};

TimingResult format_timing(std::chrono::nanoseconds duration, int batch_size)
{
  TimingResult result;
  double total_ns = static_cast<double>(duration.count());
  double avg_ns = total_ns / batch_size;

  result.ops_per_sec = 1e9 / avg_ns; // Operations per second

  // Choose appropriate unit for average time
  if (avg_ns < 1000.0) {
    result.avg_time = avg_ns;
    result.avg_unit = "ns";
  } else if (avg_ns < 1000000.0) {
    result.avg_time = avg_ns / 1000.0;
    result.avg_unit = "μs";
  } else {
    result.avg_time = avg_ns / 1000000.0;
    result.avg_unit = "ms";
  }

  // Choose appropriate unit for total time
  if (total_ns < 1000.0) {
    result.total_time = total_ns;
    result.total_unit = "ns";
  } else if (total_ns < 1000000.0) {
    result.total_time = total_ns / 1000.0;
    result.total_unit = "μs";
  } else if (total_ns < 1000000000.0) {
    result.total_time = total_ns / 1000000.0;
    result.total_unit = "ms";
  } else {
    result.total_time = total_ns / 1000000000.0;
    result.total_unit = "s";
  }

  return result;
}

// Benchmark ML-KEM operations for a specific parameter set
template <typename Params>
void benchmark_ml_kem(const std::string& param_name, int batch_size)
{
  std::cout << "\n=== " << param_name << " Benchmark (batch_size=" << batch_size << ") ===" << std::endl;

  // Allocate buffers for batch operations
  std::vector<std::byte> entropy(icicle::pqc::ml_kem::ENTROPY_BYTES * batch_size);
  std::vector<std::byte> message(icicle::pqc::ml_kem::MESSAGE_BYTES * batch_size);
  std::vector<std::byte> public_key(Params::PUBLIC_KEY_BYTES * batch_size);
  std::vector<std::byte> secret_key(Params::SECRET_KEY_BYTES * batch_size);
  std::vector<std::byte> ciphertext(Params::CIPHERTEXT_BYTES * batch_size);
  std::vector<std::byte> shared_secret_enc(Params::SHARED_SECRET_BYTES * batch_size);
  std::vector<std::byte> shared_secret_dec(Params::SHARED_SECRET_BYTES * batch_size);

  // warmup
  {
    icicle::pqc::ml_kem::MlKemConfig warmup_config = {};
    warmup_config.batch_size = batch_size;
    // Use the same buffers as below, but don't measure time
    icicle::pqc::ml_kem::keygen<Params>(entropy.data(), warmup_config, public_key.data(), secret_key.data());
    icicle::pqc::ml_kem::encapsulate<Params>(
      message.data(), public_key.data(), warmup_config, ciphertext.data(), shared_secret_enc.data());
    icicle::pqc::ml_kem::decapsulate<Params>(
      secret_key.data(), ciphertext.data(), warmup_config, shared_secret_dec.data());
  }

  // Generate random data for all batch items
  generate_random_bytes(entropy.data(), entropy.size());
  generate_random_bytes(message.data(), message.size());

  icicle::pqc::ml_kem::MlKemConfig config = {};
  config.batch_size = batch_size;

  // Benchmark key generation
  auto start = std::chrono::high_resolution_clock::now();
  auto keygen_result =
    icicle::pqc::ml_kem::keygen<Params>(entropy.data(), config, public_key.data(), secret_key.data());
  auto end = std::chrono::high_resolution_clock::now();
  auto keygen_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  if (keygen_result != eIcicleError::SUCCESS) {
    std::cout << "Key generation failed!" << std::endl;
    return;
  }

  // Benchmark encapsulation
  start = std::chrono::high_resolution_clock::now();
  auto encaps_result = icicle::pqc::ml_kem::encapsulate<Params>(
    message.data(), public_key.data(), config, ciphertext.data(), shared_secret_enc.data());
  end = std::chrono::high_resolution_clock::now();
  auto encaps_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  if (encaps_result != eIcicleError::SUCCESS) {
    std::cout << "Encapsulation failed!" << std::endl;
    return;
  }

  // Benchmark decapsulation
  start = std::chrono::high_resolution_clock::now();
  auto decaps_result =
    icicle::pqc::ml_kem::decapsulate<Params>(secret_key.data(), ciphertext.data(), config, shared_secret_dec.data());
  end = std::chrono::high_resolution_clock::now();
  auto decaps_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  if (decaps_result != eIcicleError::SUCCESS) {
    std::cout << "Decapsulation failed!" << std::endl;
    return;
  }

  // Verify shared secrets match for all batch items
  bool all_secrets_match = true;
  for (int i = 0; i < batch_size; ++i) {
    bool secrets_match =
      (0 == memcmp(
              shared_secret_enc.data() + i * Params::SHARED_SECRET_BYTES,
              shared_secret_dec.data() + i * Params::SHARED_SECRET_BYTES, Params::SHARED_SECRET_BYTES));
    if (!secrets_match) {
      all_secrets_match = false;
      break;
    }
  }

  // Format timing results with appropriate units
  auto keygen_timing = format_timing(keygen_duration, batch_size);
  auto encaps_timing = format_timing(encaps_duration, batch_size);
  auto decaps_timing = format_timing(decaps_duration, batch_size);

  // Print results with adaptive precision
  std::cout << std::fixed;

  // Set precision based on unit (more precision for smaller units)
  auto print_timing = [](const std::string& name, const TimingResult& timing) {
    // Set precision for average time
    if (timing.avg_unit == "ns") {
      std::cout << std::setprecision(1);
    } else if (timing.avg_unit == "μs") {
      std::cout << std::setprecision(2);
    } else {
      std::cout << std::setprecision(3);
    }

    std::cout << name << " (avg): " << std::setw(8) << timing.avg_time << " " << timing.avg_unit << "  (total: ";

    // Set precision for total time
    if (timing.total_unit == "ns") {
      std::cout << std::setprecision(1);
    } else if (timing.total_unit == "μs") {
      std::cout << std::setprecision(2);
    } else {
      std::cout << std::setprecision(3);
    }

    std::cout << timing.total_time << " " << timing.total_unit << ", " << std::setprecision(0) << timing.ops_per_sec
              << " ops/sec)" << std::endl;
  };

  print_timing("Keygen", keygen_timing);
  print_timing("Encaps", encaps_timing);
  print_timing("Decaps", decaps_timing);
  std::cout << "Verification: " << (all_secrets_match ? "✅ PASS" : "❌ FAIL") << std::endl;
}

int main(int argc, char* argv[])
{
  // Parse command line arguments
  int batch_size = 1; // default batch size

  if (argc > 1) {
    try {
      batch_size = std::stoi(argv[1]);
      if (batch_size <= 0) {
        std::cout << "Error: Batch size must be positive" << std::endl;
        return 1;
      }
    } catch (const std::exception& e) {
      std::cout << "Error: Invalid batch size argument" << std::endl;
      std::cout << "Usage: " << argv[0] << " [batch_size]" << std::endl;
      return 1;
    }
  }

  std::cout << "ICICLE ML-KEM Benchmark\n" << std::endl;
  icicle::Device dev = {"CUDA-PQC", 0};
  icicle_set_device(dev);
  dev = icicle::DeviceAPI::get_thread_local_device();
  std::cout << "Using device: " << dev << std::endl;

  // Benchmark all parameter sets
  benchmark_ml_kem<icicle::pqc::ml_kem::Kyber512Params>("ML-KEM-512", batch_size);
  benchmark_ml_kem<icicle::pqc::ml_kem::Kyber768Params>("ML-KEM-768", batch_size);
  benchmark_ml_kem<icicle::pqc::ml_kem::Kyber1024Params>("ML-KEM-1024", batch_size);

  return 0;
}