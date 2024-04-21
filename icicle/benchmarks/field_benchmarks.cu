#include <benchmark/benchmark.h>
#include "utils/test_functions.cuh"
#include "fields/field_config.cuh"

using namespace field_config;
using namespace benchmark;

template <class T>
static void BM_FieldAdd(State& state)
{
  constexpr int N = 256;
  int n = state.range(0) / N;
  T* scalars1;
  T* scalars2;
  assert(!cudaMalloc(&scalars1, n * sizeof(T)));
  assert(!cudaMalloc(&scalars2, n * sizeof(T)));

  assert(device_populate_random<T>(scalars1, n) == cudaSuccess);
  assert(device_populate_random<T>(scalars2, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_add<T, T, N>(scalars1, scalars2, scalars1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  state.counters["Throughput"] = Counter(state.range(0), Counter::kIsRate | Counter::kIsIterationInvariant);
  cudaFree(scalars1);
  cudaFree(scalars2);
}

template <class T>
static void BM_FieldMul(State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  T* scalars1;
  T* scalars2;
  assert(!cudaMalloc(&scalars1, n * sizeof(T)));
  assert(!cudaMalloc(&scalars2, n * sizeof(T)));

  assert(device_populate_random<T>(scalars1, n) == cudaSuccess);
  assert(device_populate_random<T>(scalars2, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_mul<T, T, N>(scalars1, scalars2, scalars1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  state.counters["Throughput"] = Counter(state.range(0), Counter::kIsRate | Counter::kIsIterationInvariant);
  cudaFree(scalars1);
  cudaFree(scalars2);
}

template <class T>
static void BM_FieldSqr(State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  T* scalars;
  assert(!cudaMalloc(&scalars, n * sizeof(T)));

  assert(device_populate_random<T>(scalars, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((field_vec_sqr<T, N>(scalars, scalars, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  state.counters["Throughput"] = Counter(state.range(0), Counter::kIsRate | Counter::kIsIterationInvariant);
  cudaFree(scalars);
}

BENCHMARK(BM_FieldAdd<scalar_t>)->Range(1 << 28, 1 << 28)->Unit(kMicrosecond);
BENCHMARK(BM_FieldMul<scalar_t>)->Range(1 << 27, 1 << 27)->Unit(kMicrosecond);
BENCHMARK(BM_FieldSqr<scalar_t>)->Range(1 << 27, 1 << 27)->Unit(kMicrosecond);

#ifdef EXT_FIELD
BENCHMARK(BM_FieldAdd<extension_t>)->Range(1 << 28, 1 << 28)->Unit(kMicrosecond);
BENCHMARK(BM_FieldMul<extension_t>)->Range(1 << 27, 1 << 27)->Unit(kMicrosecond);
BENCHMARK(BM_FieldSqr<extension_t>)->Range(1 << 27, 1 << 27)->Unit(kMicrosecond);
#endif
