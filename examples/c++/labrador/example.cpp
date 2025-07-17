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
BenchmarkResult run_benchmark(const BenchmarkConfig& config, bool SKIP_VERIF)
{
  std::cout << "Running benchmark: n=" << config.n << ", r=" << config.r << ", eq_const=" << config.num_eq_const
            << ", cz_const=" << config.num_cz_const << ", repetitions=" << config.num_repetitions << std::endl;

  constexpr size_t d = Rq::d;
  const size_t max_value = 2;
  std::vector<double> times;
  bool all_verified = true;

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
    std::vector<EqualityInstance> eq_inst = create_rand_eq_inst(config.n, config.r, S, config.num_eq_const);
    lab_inst.add_equality_constraint(eq_inst);

    // Add constant zero constraints
    std::vector<ConstZeroInstance> const_zero_inst = create_rand_const_zero_inst(config.n, config.r, S, config.num_cz_const);
    lab_inst.add_const_zero_constraint(const_zero_inst);

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

void benchmark_program()
{
  ICICLE_LOG_INFO << "Labrador Benchmark";

  // Benchmark parameters from the original code
  std::vector<std::tuple<size_t, size_t>> arr_nr{{1 << 9, 1 << 5}};
  std::vector<std::tuple<size_t, size_t>> num_constraint{{10, 10}};
  size_t NUM_REP = 1;
  bool SKIP_VERIF = true;

  std::vector<BenchmarkResult> results;

  // Run benchmarks for all parameter combinations
  for (const auto& [n, r] : arr_nr) {
    for (const auto& [num_eq, num_cz] : num_constraint) {
      BenchmarkConfig config{n, r, num_eq, num_cz, NUM_REP};

      try {
        BenchmarkResult result = run_benchmark(config, SKIP_VERIF);
        results.push_back(result);
      } catch (const std::exception& e) {
        std::cerr << "Error running benchmark for n=" << n << ", r=" << r << ", eq=" << num_eq << ", cz=" << num_cz
                  << ": " << e.what() << std::endl;
      }
      if (!results.empty()) { save_to_csv(results, "labrador_benchmark_results.csv"); }
    }
  }

  // Print results
  print_results(results);

  // Save to CSV
  save_to_csv(results, "labrador_benchmark_results.csv");
}

void prover_verifier_trace()
{
  const int64_t q = get_q<Zq>();

  // TODO: icicle_malloc()/ DeviceVector<T>

  // randomize the witness Si with low norm
  const size_t n = 1 << 9;
  const size_t r = 1 << 5;
  constexpr size_t d = Rq::d;
  const size_t max_value = 2;
  size_t num_eq_const =10;
  size_t num_cz_const =10;

  const std::vector<Rq> S = rand_poly_vec(r * n, max_value);
  auto eq_inst = create_rand_eq_inst(n, r, S, num_eq_const);
  std::cout<<"Created Eq constraints\n";
  auto const_zero_inst = create_rand_const_zero_inst(n, r, S, num_cz_const);
  std::cout<<"Created Cz constraints\n";

  // for(auto &inst: eq_inst){
  //   assert(witness_legit_eq(inst, S));
  // }
  // std::cout<<"Checked Eq constraints\n";
  // for(auto &inst: const_zero_inst){
  //   assert(witness_legit_const_zero(inst, S));
  // }
  // std::cout<<"Checked Eq constraints\n";

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;

  double beta = sqrt(max_value * n * r * d);
  uint32_t base0 = calc_base0(r, OP_NORM_BOUND, beta);
  LabradorParam param{
    r,
    n,
    {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
     reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
    secure_msis_rank(), // kappa
    secure_msis_rank(), // kappa1
    secure_msis_rank(), // kappa2,
    base0,              // base1
    base0,              // base2
    base0,              // base3
    beta,               // beta
  };
  LabradorInstance lab_inst{param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);

  std::string oracle_seed = "ORACLE_SEED";

  size_t NUM_REC = 1;
  LabradorProver prover{
    lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

  std::cout << "Problem param: n,r = " << n << ", " << r << "\n";
  std::cout << "TESTING = " << (TESTING ? "true" : "false") << "\n";
  if (TESTING) { std::cout << "TESTING TRUE IMPLIES TIMING ESTIMATES ARE INCORRECT\n"; }
  auto [trs, final_proof] = prover.prove();

  // extract all prover_msg from trs vector into a vector prover_msgs
  std::vector<BaseProverMessages> prover_msgs;
  for (const auto& transcript : trs) {
    prover_msgs.push_back(transcript.prover_msg);
  }
  if (TESTING) {
    LabradorVerifier verifier{lab_inst,           prover_msgs,
                              final_proof,        reinterpret_cast<const std::byte*>(oracle_seed.data()),
                              oracle_seed.size(), NUM_REC};

    std::cout << "Verification result: \n";
    if (verifier.verify()) {
      std::cout << "Verification passed. \n";
    } else {
      std::cout << "Verification failed. \n";
    }
  }
  // LabradorBaseProver base_prover{
  //   lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size()};

  // auto [base_proof, trs] = base_prover.base_case_prover();

  // LabradorInstance verif_lab_inst{param};
  // verif_lab_inst.add_equality_constraint(eq_inst);
  // verif_lab_inst.add_const_zero_constraint(const_zero_inst);
  // verif_lab_inst.add_const_zero_constraint(const_zero_inst2);
  // // to break verification:
  // // verif_lab_inst.const_zero_constraints[0].b = verif_lab_inst.const_zero_constraints[0].b + Zq::one();
  // LabradorBaseVerifier base_verifier{
  //   verif_lab_inst, trs.prover_msg, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size()};

  // // // Assert that Verifier trs and Prover trs are equal
  // // auto bytes_eq = [](const std::vector<std::byte>& a, const std::vector<std::byte>& b) { return a == b; };
  // // auto zq_vec_eq = [](const std::vector<Zq>& a, const std::vector<Zq>& b) {
  // //   if (a.size() != b.size()) return false;
  // //   for (size_t i = 0; i < a.size(); ++i)
  // //     if (a[i] != b[i]) return false;
  // //   return true;
  // // };

  // // const auto& trs_P = trs;               // Prover's transcript
  // // const auto& trs_V = base_verifier.trs; // Verifier's transcript

  // // auto report = [&](bool ok, const std::string& name) {
  // //   if (!ok) std::cout << "  • " << name << " mismatch\n";
  // //   return ok;
  // // };

  // // bool ok = true;
  // // ok &= report(bytes_eq(trs_P.seed1, trs_V.seed1), "seed1");
  // // ok &= report(bytes_eq(trs_P.seed2, trs_V.seed2), "seed2");
  // // ok &= report(bytes_eq(trs_P.seed3, trs_V.seed3), "seed3");
  // // ok &= report(bytes_eq(trs_P.seed4, trs_V.seed4), "seed4");

  // // ok &= report(zq_vec_eq(trs_P.psi, trs_V.psi), "psi");
  // // ok &= report(zq_vec_eq(trs_P.omega, trs_V.omega), "omega");

  // // ok &= report(poly_vec_eq(trs_P.alpha_hat.data(), trs_V.alpha_hat.data(), trs_P.alpha_hat.size()), "alpha_hat");

  // // ok &= report(
  // //   poly_vec_eq(trs_P.challenges_hat.data(), trs_V.challenges_hat.data(), trs_P.challenges_hat.size()),
  // //   "challenges_hat");

  // // if (!ok) {
  // //   std::cerr << "\nTranscript mismatch detected above.\n";
  // //   return 1;
  // // }
  // // std::cout << "Transcript check passed ✅\n";

  // bool verification_result = base_verifier.fully_verify(base_proof);

  // if (verification_result) {
  //   std::cout << "Base proof verification passed\n";
  // } else {
  //   std::cout << "Base proof verification failed\n";
  // }

  // std::cout << "Beginning recursion... \n";
  // uint32_t base0 = 1 << 3;
  // size_t mu = 1 << 3, nu = 1 << 3;
  // LabradorInstance rec_inst = prepare_recursion_instance(
  //   param,                                        // prev_param
  //   base_prover.lab_inst.equality_constraints[0], // final_const,
  //   trs, base0, mu, nu);
  // LabradorProver dummy_prover{
  //   lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), 1};
  // std::vector<Rq> rec_S = dummy_prover.prepare_recursion_witness(base_proof, base0, mu, nu);
  // std::cout << "rec_inst.r = " << rec_inst.param.r << std::endl;
  // std::cout << "rec_inst.n = " << rec_inst.param.n << std::endl;
  // std::cout << "Num rec_inst.equality_constraints = " << rec_inst.equality_constraints.size() << std::endl;
  // std::cout << "Num rec_inst.const_zero_constraints = " << rec_inst.const_zero_constraints.size() << std::endl;

  // std::cout << "\tTesting rec-witness validity...";
  // assert(lab_witness_legit(rec_inst, rec_S));
  // std::cout << "VALID\n";

  // Single round TRS test
  // size_t NUM_REC = 1;
  // LabradorProver prover{
  //   lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

  // auto [trs, final_proof] = prover.prove();

  // // extract all prover_msg from trs vector into a vector prover_msgs
  // std::vector<BaseProverMessages> prover_msgs;
  // for (const auto& transcript : trs) {
  //   prover_msgs.push_back(transcript.prover_msg);
  // }

  // LabradorBaseVerifier base_verifier{
  //   lab_inst, prover_msgs[0], reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size()};

  // // // Assert that Verifier trs and Prover trs are equal
  // auto bytes_eq = [](const std::vector<std::byte>& a, const std::vector<std::byte>& b) { return a == b; };
  // auto zq_vec_eq = [](const std::vector<Zq>& a, const std::vector<Zq>& b) {
  //   if (a.size() != b.size()) return false;
  //   for (size_t i = 0; i < a.size(); ++i)
  //     if (a[i] != b[i]) return false;
  //   return true;
  // };

  // const auto& trs_P = trs[0];            // Prover's transcript
  // const auto& trs_V = base_verifier.trs; // Verifier's transcript

  // auto report = [&](bool ok, const std::string& name) {
  //   if (!ok) std::cout << "  • " << name << " mismatch\n";
  //   return ok;
  // };

  // bool ok = true;
  // ok &= report(bytes_eq(trs_P.seed1, trs_V.seed1), "seed1");
  // ok &= report(bytes_eq(trs_P.seed2, trs_V.seed2), "seed2");
  // ok &= report(bytes_eq(trs_P.seed3, trs_V.seed3), "seed3");
  // ok &= report(bytes_eq(trs_P.seed4, trs_V.seed4), "seed4");

  // ok &= report(zq_vec_eq(trs_P.psi, trs_V.psi), "psi");
  // ok &= report(zq_vec_eq(trs_P.omega, trs_V.omega), "omega");

  // ok &= report(poly_vec_eq(trs_P.alpha_hat.data(), trs_V.alpha_hat.data(), trs_P.alpha_hat.size()), "alpha_hat");

  // ok &= report(
  //   poly_vec_eq(trs_P.challenges_hat.data(), trs_V.challenges_hat.data(), trs_P.challenges_hat.size()),
  //   "challenges_hat");

  // if (!ok) {
  //   std::cerr << "\nTranscript mismatch detected above.\n";
  //   return 1;
  // }
  // std::cout << "Transcript check passed ✅\n";

  // bool verification_result = base_verifier.fully_verify(final_proof);

  // if (verification_result) {
  //   std::cout << "Base proof verification passed\n";
  // } else {
  //   std::cout << "Base proof verification failed\n";
  // }
  // return 0;
}

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  try_load_and_set_backend_device(argc, argv);

  // I. Use the following code for examining program trace:
  // prover_verifier_trace();

  // II. To run benchmark uncomment:
  benchmark_program();

  return 0;
}

// TODO: change poly_vec_eq