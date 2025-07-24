#include "labrador.h"           // For Zq, Rq, Tq, and the APIs
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

/// Benchmark configuration

struct BenchmarkConfig {
  /// n: witness vector length
  size_t n;
  /// r: number of witnesses
  size_t r;
  /// num_eq_const: number of equality constraints
  size_t num_eq_const;
  /// num_cz_const: number of constantZero constraints
  size_t num_cz_const;
  /// num_repetitions: number of repetitions for the benchmarks
  size_t num_repetitions;
};

// Benchmark result
struct BenchmarkResult {
  /// n: witness vector length
  size_t n;
  /// r: number of witnesses
  size_t r;
  /// num_eq_const: number of equality constraints
  size_t num_eq_const;
  /// num_cz_const: number of constantZero constraints
  size_t num_cz_const;
  /// num_repetitions: number of repetitions for the benchmarks
  size_t num_repetitions;
  /// avg_time_ms: average execution time in milliseconds
  double avg_time_ms;
  /// min_time_ms: minimum execution time in milliseconds
  double min_time_ms;
  /// max_time_ms: maximum execution time in milliseconds
  double max_time_ms;
  /// std_dev_ms: standard deviation of execution times in milliseconds
  double std_dev_ms;
  /// all_verified: whether all benchmark runs passed verification
  bool all_verified;
};

/// Function to run a single benchmark configuration for the Labrador Prover with no recursion
/// @param config: benchmark config to run
/// @param SKIP_VERIF: if true skips verification
BenchmarkResult run_benchmark(const BenchmarkConfig& config, bool SKIP_VERIF);

// Function to print results table
void print_results(const std::vector<BenchmarkResult>& results);

// Function to save results to CSV
void save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename);

/// Function to run multiple benchmarks
/// @param arr_nr: vector of (n,r) tuples
/// @param num_constraint: vector of (number of equality constraints, number of constant zero constraints)
/// @param NUM_REP: number of times the benchmarks are repeated
/// @param SKIP_VERIF: if true skips verification
void benchmark_program(
  std::vector<std::tuple<size_t, size_t>> arr_nr,
  std::vector<std::tuple<size_t, size_t>> num_constraint,
  size_t NUM_REP,
  bool SKIP_VERIF);
