#include <benchmark/benchmark.h>
#include "utils/test_functions.cuh"
#include "curves/curve_config.cuh"

using namespace curve_config;
using namespace benchmark;

static void BM_MixedECAdd(State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  projective_t* points1;
  affine_t* points2;
  assert(!cudaMalloc(&points1, n * sizeof(projective_t)));
  assert(!cudaMalloc(&points2, n * sizeof(affine_t)));

  projective_t* h_points1 = (projective_t*)malloc(n * sizeof(projective_t));
  affine_t* h_points2 = (affine_t*)malloc(n * sizeof(affine_t));
  projective_t::RandHostMany(h_points1, n);
  projective_t::RandHostManyAffine(h_points2, n);
  cudaMemcpy(points1, h_points1, sizeof(projective_t) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(points2, h_points2, sizeof(affine_t) * n, cudaMemcpyHostToDevice);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_add<projective_t, affine_t, N>(points1, points2, points1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  state.counters["Throughput"] = Counter(state.range(0), Counter::kIsRate | Counter::kIsIterationInvariant);
  cudaFree(points1);
  cudaFree(points2);
}

static void BM_FullECAdd(benchmark::State& state)
{
  constexpr int N = 128;
  int n = state.range(0) / N;
  projective_t* points1;
  projective_t* points2;
  assert(!cudaMalloc(&points1, n * sizeof(projective_t)));
  assert(!cudaMalloc(&points2, n * sizeof(projective_t)));

  projective_t* h_points1 = (projective_t*)malloc(n * sizeof(projective_t));
  projective_t* h_points2 = (projective_t*)malloc(n * sizeof(projective_t));
  projective_t::RandHostMany(h_points1, n);
  projective_t::RandHostMany(h_points2, n);
  cudaMemcpy(points1, h_points1, sizeof(projective_t) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(points2, h_points2, sizeof(projective_t) * n, cudaMemcpyHostToDevice);

  for (auto _ : state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert((vec_add<projective_t, projective_t, N>(points1, points2, points1, n)) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    state.SetIterationTime((double)(milliseconds / 1000));
  }
  state.counters["Throughput"] = Counter(state.range(0), Counter::kIsRate | Counter::kIsIterationInvariant);
  cudaFree(points1);
  cudaFree(points2);
}

BENCHMARK(BM_FullECAdd)->Range(1 << 27, 1 << 27)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MixedECAdd)->Range(1 << 27, 1 << 27)->Unit(benchmark::kMillisecond);