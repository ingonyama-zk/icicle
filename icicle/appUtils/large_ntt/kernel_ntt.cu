
#include "thread_ntt.cu"
#include "curves/curve_config.cuh"
#include "appUtils/large_ntt/large_ntt.cuh"

namespace ntt {

  __device__ uint32_t dig_rev(uint32_t num, uint32_t log_size, bool dit)
  {
    uint32_t rev_num = 0, temp, dig_len;
    if (dit) {
      for (int i = 4; i >= 0; i--) {
        dig_len = STAGE_SIZES_DEVICE[log_size][i];
        temp = num & ((1 << dig_len) - 1);
        num = num >> dig_len;
        rev_num = rev_num << dig_len;
        rev_num = rev_num | temp;
      }
    } else {
      for (int i = 0; i < 5; i++) {
        dig_len = STAGE_SIZES_DEVICE[log_size][i];
        temp = num & ((1 << dig_len) - 1);
        num = num >> dig_len;
        rev_num = rev_num << dig_len;
        rev_num = rev_num | temp;
      }
    }
    return rev_num;
  }

  __launch_bounds__(64) __global__
    void reorder_digits_kernel(uint4* arr, uint4* arr_reordered, uint32_t log_size, bool dit)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t rd = tid;
    uint32_t wr = dig_rev(tid, log_size, dit);
    arr_reordered[wr] = arr[rd];
    arr_reordered[wr + (1 << log_size)] = arr[rd + (1 << log_size)];
  }

  __launch_bounds__(64) __global__ void ntt64(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine engine;
    stage_metadata s_meta;
    extern __shared__ uint4 shmem[];

    s_meta.th_stride = 8;
    s_meta.ntt_block_size = 64;
    s_meta.ntt_block_id = (blockIdx.x << 3) + (strided ? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 3) : (threadIdx.x & 0x7);

    engine.loadBasicTwiddles(basic_twiddles);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride && dit) {
      engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles(internal_twiddles, strided);

#pragma unroll 1
    for (uint32_t phase = 0; phase < 2; phase++) {
      engine.ntt8win();
      if (phase == 0) {
        engine.SharedData64Columns8(shmem, true, false, strided); // store low
        __syncthreads();
        engine.SharedData64Rows8(shmem, false, false, strided); // load low
        engine.SharedData64Rows8(shmem, true, true, strided);   // store high
        __syncthreads();
        engine.SharedData64Columns8(shmem, false, true, strided); // load high
        engine.twiddlesInternal();
      }
    }

    if (twiddle_stride && !dit) {
      engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  __launch_bounds__(64) __global__ void ntt32(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine engine;
    stage_metadata s_meta;
    extern __shared__ uint4 shmem[];

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    engine.loadBasicTwiddles(basic_twiddles);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    engine.loadInternalTwiddles32(internal_twiddles, strided);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData32Columns8(shmem, true, false, strided); // store low
    __syncthreads();
    engine.SharedData32Rows4_2(shmem, false, false, strided); // load low
    engine.SharedData32Rows8(shmem, true, true, strided);     // store high
    __syncthreads();
    engine.SharedData32Columns4_2(shmem, false, true, strided); // load high
    engine.ntt4_2();
    if (twiddle_stride) {
      engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData32(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  __launch_bounds__(64) __global__ void ntt32dit(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine engine;
    stage_metadata s_meta;
    extern __shared__ uint4 shmem[];

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    engine.loadBasicTwiddles(basic_twiddles);
    engine.loadGlobalData32(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles32(internal_twiddles, strided);
    engine.ntt4_2();
    engine.SharedData32Columns4_2(shmem, true, false, strided); // store low
    __syncthreads();
    engine.SharedData32Rows8(shmem, false, false, strided); // load low
    engine.SharedData32Rows4_2(shmem, true, true, strided); // store high
    __syncthreads();
    engine.SharedData32Columns8(shmem, false, true, strided); // load high
    engine.twiddlesInternal();
    engine.ntt8win();
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  __launch_bounds__(64) __global__ void ntt16(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine engine;
    stage_metadata s_meta;
    extern __shared__ uint4 shmem[];

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    engine.loadBasicTwiddles(basic_twiddles);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    engine.loadInternalTwiddles16(internal_twiddles, strided);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData16Columns8(shmem, true, false, strided); // store low
    __syncthreads();
    engine.SharedData16Rows2_4(shmem, false, false, strided); // load low
    engine.SharedData16Rows8(shmem, true, true, strided);     // store high
    __syncthreads();
    engine.SharedData16Columns2_4(shmem, false, true, strided); // load high
    engine.ntt2_4();
    if (twiddle_stride) {
      engine.loadExternalTwiddles16(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData16(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  __launch_bounds__(64) __global__ void ntt16dit(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine engine;
    stage_metadata s_meta;
    extern __shared__ uint4 shmem[];

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    engine.loadBasicTwiddles(basic_twiddles);
    engine.loadGlobalData16(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      engine.loadExternalTwiddles16(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles16(internal_twiddles, strided);
    engine.ntt2_4();
    engine.SharedData16Columns2_4(shmem, true, false, strided); // store low
    __syncthreads();
    engine.SharedData16Rows8(shmem, false, false, strided); // load low
    engine.SharedData16Rows2_4(shmem, true, true, strided); // store high
    __syncthreads();
    engine.SharedData16Columns8(shmem, false, true, strided); // load high
    engine.twiddlesInternal();
    engine.ntt8win();
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  __global__ void normalize_kernel(uint4* data, uint32_t size, curve_config::scalar_t norm_factor)
  {
    curve_config::scalar_t temp;
    temp.store_half(data[threadIdx.x], false);
    temp.store_half(data[threadIdx.x + size], true);
    temp = temp * norm_factor;
    data[threadIdx.x] = temp.load_half(false);
    data[threadIdx.x + size] = temp.load_half(true);
  }

  __global__ void generate_base_table(curve_config::scalar_t basic_root, uint4* base_table, uint32_t skip)
  {
    curve_config::scalar_t w = basic_root;
    curve_config::scalar_t t = curve_config::scalar_t::one();
    for (int i = 0; i < 64; i += skip) {
      base_table[i] = t.load_half(false);
      base_table[i + 64] = t.load_half(true);
      t = t * w;
    }
  }

  __global__ void generate_basic_twiddles(curve_config::scalar_t basic_root, uint4* basic_twiddles)
  {
    curve_config::scalar_t w0 = basic_root * basic_root;
    curve_config::scalar_t w1 = (basic_root + w0 * basic_root) * curve_config::scalar_t::inv_log_size(1);
    curve_config::scalar_t w2 = (basic_root - w0 * basic_root) * curve_config::scalar_t::inv_log_size(1);
    basic_twiddles[0] = w0.load_half(false);
    basic_twiddles[3] = w0.load_half(true);
    basic_twiddles[1] = w1.load_half(false);
    basic_twiddles[4] = w1.load_half(true);
    basic_twiddles[2] = w2.load_half(false);
    basic_twiddles[5] = w2.load_half(true);
  }

  __global__ void generate_twiddle_combinations(
    uint4* w6_table,
    uint4* w12_table,
    uint4* w18_table,
    uint4* w24_table,
    uint4* w30_table,
    uint4* twiddles,
    uint32_t log_size,
    uint32_t stage_num,
    curve_config::scalar_t norm_factor)
  {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t range1 = 0, range2 = 0, ind;
    for (ind = 0; ind < stage_num; ind++)
      range1 += STAGE_SIZES_DEVICE[log_size][ind];
    range2 = STAGE_SIZES_DEVICE[log_size][ind];
    uint32_t root_order = range1 + range2;
    uint32_t exp = ((tid & ((1 << range1) - 1)) * (tid >> range1)) << (30 - root_order);
    curve_config::scalar_t w6, w12, w18, w24, w30;
    w6.store_half(w6_table[exp >> 24], false);
    w6.store_half(w6_table[(exp >> 24) + 64], true);
    w12.store_half(w12_table[((exp >> 18) & 0x3f)], false);
    w12.store_half(w12_table[((exp >> 18) & 0x3f) + 64], true);
    w18.store_half(w18_table[((exp >> 12) & 0x3f)], false);
    w18.store_half(w18_table[((exp >> 12) & 0x3f) + 64], true);
    w24.store_half(w24_table[((exp >> 6) & 0x3f)], false);
    w24.store_half(w24_table[((exp >> 6) & 0x3f) + 64], true);
    w30.store_half(w30_table[(exp & 0x3f)], false);
    w30.store_half(w30_table[(exp & 0x3f) + 64], true);
    curve_config::scalar_t t = w6 * w12 * w18 * w24 * w30 * norm_factor;
    twiddles[tid + LOW_W_OFFSETS[log_size][stage_num]] = t.load_half(false);
    twiddles[tid + HIGH_W_OFFSETS[log_size][stage_num]] = t.load_half(true);
  }

  void large_ntt(
    uint4* in,
    uint4* out,
    uint4* twiddles,
    uint4* internal_twiddles,
    uint4* basic_twiddles,
    uint32_t log_size,
    bool inv,
    bool dit)
  {
    // special cases:
    if (log_size == 1 || log_size == 2 || log_size == 3 || log_size == 7) {
      throw std::invalid_argument("size not implemented");
    }
    if (log_size == 4) {
      if (dit) {
        ntt16dit<<<1, 4, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      } else {
        ntt16<<<1, 4, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      }
      if (inv) normalize_kernel<<<1, 16>>>(out, 16, curve_config::scalar_t::inv_log_size(4));
      return;
    }
    if (log_size == 5) {
      if (dit) {
        ntt32dit<<<1, 4, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      } else {
        ntt32<<<1, 4, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      }
      if (inv) normalize_kernel<<<1, 32>>>(out, 32, curve_config::scalar_t::inv_log_size(5));
      return;
    }
    if (log_size == 6) {
      ntt64<<<1, 8, 8 * 64 * sizeof(uint4)>>>(
        in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      if (inv) normalize_kernel<<<1, 64>>>(out, 64, curve_config::scalar_t::inv_log_size(6));
      return;
    }
    if (log_size == 8) {
      if (dit)
        ntt16dit<<<1, 32, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      if (dit)
        ntt16dit<<<1, 64, 8 * 64 * sizeof(uint4)>>>(
          out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 16, 4, 16, true, 1, inv,
          dit); // we need threads 32+ although 16-31 are idle
      if (!dit)
        ntt16<<<1, 64, 8 * 64 * sizeof(uint4)>>>(
          in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 16, 4, 16, true, 1, inv,
          dit); // we need threads 32+ although 16-31 are idle
      if (!dit)
        ntt16<<<1, 32, 8 * 64 * sizeof(uint4)>>>(
          out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1, 0, 0, false, 0, inv, dit);
      return;
    }

    // general case:
    if (dit) {
      for (int i = 0; i < 5; i++) {
        uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += STAGE_SIZES_HOST[log_size][j];
        if (stage_size == 6)
          ntt64<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            i ? out : in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log, stride_log,
            i ? (1 << stride_log) : 0, i, i, inv, dit);
        if (stage_size == 5)
          ntt32dit<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            i ? out : in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log, stride_log,
            i ? (1 << stride_log) : 0, i, i, inv, dit);
        if (stage_size == 4)
          ntt16dit<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            i ? out : in, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log, stride_log,
            i ? (1 << stride_log) : 0, i, i, inv, dit);
      }
    } else {
      bool first_run = false, prev_stage = false;
      for (int i = 4; i >= 0; i--) {
        uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += STAGE_SIZES_HOST[log_size][j];
        first_run = stage_size && !prev_stage;
        if (stage_size == 6)
          ntt64<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            first_run ? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        if (stage_size == 5)
          ntt32<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            first_run ? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        if (stage_size == 4)
          ntt16<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(uint4)>>>(
            first_run ? in : out, out, twiddles, internal_twiddles, basic_twiddles, log_size, 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        prev_stage = stage_size;
      }
    }
  }

  /*================================ MixedRadixNTT =========================================*/
  MixedRadixNTT::MixedRadixNTT(int ntt_size, bool is_inverse, Ordering ordering, cudaStream_t cuda_stream)
      : m_ntt_size(ntt_size), m_ntt_log_size(int(log2(ntt_size))), m_is_inverse(is_inverse), m_ordering(ordering),
        m_cuda_stream(cuda_stream)
  {
    cudaError_t err_result = init();
    if (err_result != cudaSuccess) throw(IcicleError(err_result, "CUDA error"));
  }

  cudaError_t MixedRadixNTT::init()
  {
    CHK_IF_RETURN(
      cudaMalloc(&m_gpuTwiddles, sizeof(uint4) * (m_ntt_size + 2 * (m_ntt_size >> 4)) * 2)); // TODO - sketchy
    CHK_IF_RETURN(cudaMalloc(&m_gpuBasicTwiddles, sizeof(uint4) * 3 * 2));

    const auto basic_root =
      m_is_inverse ? curve_config::scalar_t::omega_inv(m_ntt_log_size) : curve_config::scalar_t::omega(m_ntt_log_size);
    CHK_IF_RETURN(generate_external_twiddles(basic_root));

    // temp memory for algorithm
    CHK_IF_RETURN(cudaMalloc(&m_gpuMemA, sizeof(uint4) * m_ntt_size * 2));
    CHK_IF_RETURN(cudaMalloc(&m_gpuMemB, sizeof(uint4) * m_ntt_size * 2));

    return CHK_LAST();
  }

  cudaError_t MixedRadixNTT::generate_external_twiddles(curve_config::scalar_t basic_root)
  {
    CHK_IF_RETURN(cudaMalloc(&m_w6_table, sizeof(uint4) * 64 * 2));
    CHK_IF_RETURN(cudaMalloc(&m_w12_table, sizeof(uint4) * 64 * 2));
    CHK_IF_RETURN(cudaMalloc(&m_w18_table, sizeof(uint4) * 64 * 2));
    CHK_IF_RETURN(cudaMalloc(&m_w24_table, sizeof(uint4) * 64 * 2));
    CHK_IF_RETURN(cudaMalloc(&m_w30_table, sizeof(uint4) * 64 * 2));

    curve_config::scalar_t temp_root = basic_root;
    generate_base_table<<<1, 1>>>(basic_root, m_w30_table, 1 << (30 - m_ntt_log_size));
    if (m_ntt_log_size > 24)
      for (int i = 0; i < 6 - (30 - m_ntt_log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1>>>(temp_root, m_w24_table, 1 << (m_ntt_log_size > 24 ? 0 : 24 - m_ntt_log_size));
    if (m_ntt_log_size > 18)
      for (int i = 0; i < 6 - (m_ntt_log_size > 24 ? 0 : 24 - m_ntt_log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1>>>(temp_root, m_w18_table, 1 << (m_ntt_log_size > 18 ? 0 : 18 - m_ntt_log_size));
    if (m_ntt_log_size > 12)
      for (int i = 0; i < 6 - (m_ntt_log_size > 18 ? 0 : 18 - m_ntt_log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1>>>(temp_root, m_w12_table, 1 << (m_ntt_log_size > 12 ? 0 : 12 - m_ntt_log_size));
    if (m_ntt_log_size > 6)
      for (int i = 0; i < 6 - (m_ntt_log_size > 12 ? 0 : 12 - m_ntt_log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1>>>(temp_root, m_w6_table, 1 << (m_ntt_log_size > 6 ? 0 : 6 - m_ntt_log_size));
    for (int i = 0; i < 3 - (m_ntt_log_size > 6 ? 0 : 6 - m_ntt_log_size); i++)
      temp_root = temp_root * temp_root;
    generate_basic_twiddles<<<1, 1>>>(temp_root, m_gpuBasicTwiddles);

    uint32_t temp = STAGE_SIZES_HOST[m_ntt_log_size][0];
    for (int i = 1; i < 5; i++) {
      if (!STAGE_SIZES_HOST[m_ntt_log_size][i]) break;
      temp += STAGE_SIZES_HOST[m_ntt_log_size][i];
      generate_twiddle_combinations<<<1 << (temp - 8), 256>>>(
        m_w6_table, m_w12_table, m_w18_table, m_w24_table, m_w30_table, m_gpuTwiddles, m_ntt_log_size, i,
        (temp == m_ntt_log_size && m_is_inverse) ? curve_config::scalar_t::inv_log_size(m_ntt_log_size)
                                                 : curve_config::scalar_t::one());
    }
    m_gpuIntTwiddles = m_w6_table;

    return CHK_LAST();
  }

  MixedRadixNTT::~MixedRadixNTT()
  {
    cudaFreeAsync(m_gpuTwiddles, m_cuda_stream);
    cudaFreeAsync(m_gpuBasicTwiddles, m_cuda_stream);
    cudaFreeAsync(m_w6_table, m_cuda_stream);
    cudaFreeAsync(m_w12_table, m_cuda_stream);
    cudaFreeAsync(m_w18_table, m_cuda_stream);
    cudaFreeAsync(m_w24_table, m_cuda_stream);
    cudaFreeAsync(m_w30_table, m_cuda_stream);
    cudaFreeAsync(m_gpuMemA, m_cuda_stream);
    cudaFreeAsync(m_gpuMemB, m_cuda_stream);
  }

  template <typename E>
  static __global__ void copy_input_large_ntt(E* input, uint4* gpuMemA, int ntt_size)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= ntt_size) return;
    gpuMemA[tid] = input[tid].load_half(false);
    gpuMemA[ntt_size + tid] = input[tid].load_half(true);
  }

  template <typename E>
  static __global__ void copy_output_large_ntt(uint4* gpuMemA, E* output, int ntt_size)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= ntt_size) return;
    output[tid].store_half(gpuMemA[tid], false);
    output[tid].store_half(gpuMemA[ntt_size + tid], true);
  }

  template <typename E>
  cudaError_t MixedRadixNTT::operator()(E* d_input, E* d_output) // TODO fix names
  {
    CHK_INIT_IF_RETURN();

    const int NOF_BLOCKS = (1 << (max(m_ntt_log_size, 6) - 6));
    const int NOF_THREADS = min(64, 1 << m_ntt_log_size);

    copy_input_large_ntt<<<NOF_BLOCKS, NOF_THREADS>>>(d_input, m_gpuMemA, m_ntt_size);

    const bool reverse_input = m_ordering == Ordering::kNN;
    const bool reverse_output = m_ordering == Ordering::kRR;
    const bool is_dit = m_ordering == Ordering::kNN || m_ordering == Ordering::kRN;

    if (reverse_input) {
      reorder_digits_kernel<<<NOF_BLOCKS, NOF_THREADS>>>(m_gpuMemA, m_gpuMemB, m_ntt_log_size, is_dit);
    }

    uint4* ntt_input = reverse_input ? m_gpuMemB : m_gpuMemA;
    uint4* ntt_output = reverse_input ? m_gpuMemA : m_gpuMemB;
    large_ntt(
      ntt_input, ntt_output, m_gpuTwiddles, m_gpuIntTwiddles, m_gpuBasicTwiddles, m_ntt_log_size, m_is_inverse, is_dit);

    if (reverse_output) {
      reorder_digits_kernel<<<NOF_BLOCKS, NOF_THREADS>>>(ntt_output, ntt_input, m_ntt_log_size, is_dit);
      ntt_output = ntt_input;
    }

    copy_output_large_ntt<<<NOF_BLOCKS, NOF_THREADS>>>(ntt_output, d_output, m_ntt_size);

    return CHK_LAST();
  }

  // Explicit instantiation for scalar type
  template cudaError_t
  MixedRadixNTT::operator()<curve_config::scalar_t>(curve_config::scalar_t*, curve_config::scalar_t*);

} // namespace ntt
