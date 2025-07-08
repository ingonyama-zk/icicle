#include "labrador.h"           // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h" // For Hash
#include "examples_utils.h"
#include "icicle/runtime.h"

#include "types.h"
#include "utils.h"
#include "prover.h"
#include "verifier.h"
#include "shared.h"
#include "test_helpers.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>
#include <iomanip>
#include <fstream>

using namespace icicle::labrador;

// Benchmark configuration
struct BenchmarkConfig {
  size_t n;
  size_t r;
  size_t num_eq_const;
  size_t num_cz_const;
  size_t num_repetitions;
};

// Benchmark result
struct BenchmarkResult {
  size_t n;
  size_t r;
  size_t num_eq_const;
  size_t num_cz_const;
  size_t num_repetitions;
  double avg_time_ms;
  double min_time_ms;
  double max_time_ms;
  double std_dev_ms;
  bool all_verified;
};

// Function to run a single benchmark configuration
BenchmarkResult run_benchmark(const BenchmarkConfig& config)
{
  std::cout << "Running benchmark: n=" << config.n << ", r=" << config.r << ", eq_const=" << config.num_eq_const
            << ", cz_const=" << config.num_cz_const << ", repetitions=" << config.num_repetitions << std::endl;

  constexpr size_t d = Rq::d;
  const size_t max_value = 2;
  std::vector<double> times;
  bool all_verified = true;
  bool SKIP_VERIF = false;

  // Use current time as base for unique seeds
  auto base_time = std::chrono::system_clock::now();
  auto base_millis = std::chrono::duration_cast<std::chrono::milliseconds>(base_time.time_since_epoch()).count();

  for (size_t rep = 0; rep < config.num_repetitions; rep++) {
    // Generate witness for this configuration
    const std::vector<Rq> S = rand_poly_vec(config.r * config.n, max_value);

    // Create unique Ajtai seed for this repetition
    std::string ajtai_seed_str = std::to_string(base_millis + rep);

    // Calculate parameters
    double beta = sqrt(max_value * config.n * config.r * d);
    uint32_t base0 = calc_base0(config.r, OP_NORM_BOUND, beta);

    // Create LabradorParam
    LabradorParam param{
      config.r,
      config.n,
      {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
       reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
      secure_msis_rank(), // kappa
      secure_msis_rank(), // kappa1
      secure_msis_rank(), // kappa2
      base0,              // base1
      base0,              // base2
      base0,              // base3
      beta,               // beta
    };

    // Create LabradorInstance
    LabradorInstance lab_inst{param};

    // Add equality constraints
    for (size_t i = 0; i < config.num_eq_const; i++) {
      EqualityInstance eq_inst = create_rand_eq_inst(config.n, config.r, S);
      lab_inst.add_equality_constraint(eq_inst);
    }

    // Add constant zero constraints
    for (size_t i = 0; i < config.num_cz_const; i++) {
      ConstZeroInstance const_zero_inst = create_rand_const_zero_inst(config.n, config.r, S);
      lab_inst.add_const_zero_constraint(const_zero_inst);
    }

    // Create oracle seed
    std::string oracle_seed = "ORACLE_SEED_" + std::to_string(rep);

    // Create prover
    size_t NUM_REC = 1;
    LabradorProver prover{
      lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

    // Time the proving
    auto start = std::chrono::high_resolution_clock::now();
    auto [trs, final_proof] = prover.prove();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = end - start;
    double millis = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
    times.push_back(millis);

    if (!SKIP_VERIF) {
      // Verify the proof
      std::vector<BaseProverMessages> prover_msgs;
      for (const auto& transcript : trs) {
        prover_msgs.push_back(transcript.prover_msg);
      }

      LabradorVerifier verifier{lab_inst,           prover_msgs,
                                final_proof,        reinterpret_cast<const std::byte*>(oracle_seed.data()),
                                oracle_seed.size(), NUM_REC};

      if (!verifier.verify()) {
        all_verified = false;
        std::cerr << "Verification failed for repetition " << rep << std::endl;
      }
    }
    std::cout << "  Rep " << (rep + 1) << "/" << config.num_repetitions << ": " << std::fixed << std::setprecision(2)
              << millis << " ms" << std::endl;
  }

  // Calculate statistics
  double sum = 0.0;
  double min_time = times[0];
  double max_time = times[0];

  for (double time : times) {
    sum += time;
    min_time = std::min(min_time, time);
    max_time = std::max(max_time, time);
  }

  double avg_time = sum / times.size();

  // Calculate standard deviation
  double variance = 0.0;
  for (double time : times) {
    variance += (time - avg_time) * (time - avg_time);
  }
  variance /= times.size();
  double std_dev = sqrt(variance);

  return BenchmarkResult{
    config.n, config.r, config.num_eq_const, config.num_cz_const, config.num_repetitions, avg_time, min_time,
    max_time, std_dev,  all_verified};
}

// Function to print results table
void print_results(const std::vector<BenchmarkResult>& results)
{
  std::cout << "\n" << std::string(100, '=') << std::endl;
  std::cout << "BENCHMARK RESULTS" << std::endl;
  std::cout << std::string(100, '=') << std::endl;

  std::cout << std::left << std::setw(6) << "n" << std::setw(6) << "r" << std::setw(8) << "EQ" << std::setw(8) << "CZ"
            << std::setw(12) << "Avg (ms)" << std::setw(12) << "Min (ms)" << std::setw(12) << "Max (ms)"
            << std::setw(12) << "StdDev" << std::setw(10) << "Verified" << std::endl;

  std::cout << std::string(100, '-') << std::endl;

  for (const auto& result : results) {
    std::cout << std::left << std::setw(6) << result.n << std::setw(6) << result.r << std::setw(8)
              << result.num_eq_const << std::setw(8) << result.num_cz_const << std::setw(12) << std::fixed
              << std::setprecision(2) << result.avg_time_ms << std::setw(12) << std::fixed << std::setprecision(2)
              << result.min_time_ms << std::setw(12) << std::fixed << std::setprecision(2) << result.max_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << result.std_dev_ms << std::setw(10)
              << (result.all_verified ? "✓" : "✗") << std::endl;
  }

  std::cout << std::string(100, '=') << std::endl;
}

// Function to save results to CSV
void save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write header
  file << "n,r,eq_constraints,cz_constraints,num_repetitions,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,all_"
          "verified\n";
  file << std::fixed << std::setprecision(3);
  // Write data
  for (const auto& result : results) {
    file << result.n << "," << result.r << "," << result.num_eq_const << "," << result.num_cz_const << ","
         << result.num_repetitions << "," << result.avg_time_ms << "," << result.min_time_ms << ","
         << result.max_time_ms << "," << result.std_dev_ms << "," << (result.all_verified ? "true" : "false") << "\n";
  }

  file.close();
  std::cout << "Results saved to: " << filename << std::endl;
}

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador Benchmark";
  try_load_and_set_backend_device(argc, argv);

  // Benchmark parameters from the original code
  std::vector<size_t> arr_n{64};
  std::vector<size_t> arr_r{8};
  std::vector<std::tuple<size_t, size_t>> num_constraint{{1, 1}};
  size_t NUM_REP = 10;

  std::vector<BenchmarkResult> results;

  // Run benchmarks for all parameter combinations
  for (size_t n : arr_n) {
    for (size_t r : arr_r) {
      for (const auto& [num_eq, num_cz] : num_constraint) {
        BenchmarkConfig config{n, r, num_eq, num_cz, NUM_REP};

        try {
          BenchmarkResult result = run_benchmark(config);
          results.push_back(result);
        } catch (const std::exception& e) {
          std::cerr << "Error running benchmark for n=" << n << ", r=" << r << ", eq=" << num_eq << ", cz=" << num_cz
                    << ": " << e.what() << std::endl;
        }
      }
    }
  }

  // Print results
  print_results(results);

  // Save to CSV
  save_to_csv(results, "labrador_benchmark_results.csv");

  return 0;
}