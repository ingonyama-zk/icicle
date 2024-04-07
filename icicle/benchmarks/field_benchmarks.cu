#include <benchmark/benchmark.h>
#include "utils/test_functions.cuh"
#include "fields/field_config.cuh"

using namespace field_config;

static void BM_FieldAdd(benchmark::State& state)
{
  constexpr int N = 256;
  int n = state.range(0) / N;
  scalar_t* scalars1;
  scalar_t* scalars2;
  assert(!cudaMalloc(&scalars1, n * sizeof(scalar_t)));
  assert(!cudaMalloc(&scalars2, n * sizeof(scalar_t)));

  assert(device_populate_random<scalar_t>(scalars1, n) == cudaSuccess);
  assert(device_populate_random<scalar_t>(scalars2, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_add<scalar_t, scalar_t, N>(scalars1, scalars2, scalars1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  cudaFree(scalars1);
  cudaFree(scalars2);
}

static void BM_FieldMul(benchmark::State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  scalar_t* scalars1;
  scalar_t* scalars2;
  assert(!cudaMalloc(&scalars1, n * sizeof(scalar_t)));
  assert(!cudaMalloc(&scalars2, n * sizeof(scalar_t)));

  assert(device_populate_random<scalar_t>(scalars1, n) == cudaSuccess);
  assert(device_populate_random<scalar_t>(scalars2, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_mul<scalar_t, scalar_t, N>(scalars1, scalars2, scalars1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  cudaFree(scalars1);
  cudaFree(scalars2);
}

static void BM_FieldSqr(benchmark::State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  scalar_t* scalars;
  assert(!cudaMalloc(&scalars, n * sizeof(scalar_t)));

  assert(device_populate_random<scalar_t>(scalars, n) == cudaSuccess);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((field_vec_sqr<scalar_t, N>(scalars, scalars, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  cudaFree(scalars);
}

BENCHMARK(BM_FieldAdd)->Range(1 << 28, 1 << 28)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FieldMul)->Range(1 << 27, 1 << 27)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FieldSqr)->Range(1 << 27, 1 << 27)->Unit(benchmark::kMicrosecond);
