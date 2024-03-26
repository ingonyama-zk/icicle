#include "field_benchmarks.cu"
#if CURVE_ID != BABY_BEAR
#include "curve_benchmarks.cu"
#endif

BENCHMARK_MAIN();