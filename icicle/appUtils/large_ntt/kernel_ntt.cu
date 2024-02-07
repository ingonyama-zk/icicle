
#include "appUtils/large_ntt/thread_ntt.cu"
#include "curves/curve_config.cuh"
#include "utils/sharedmem.cuh"
#include "appUtils/ntt/ntt.cuh" // for Ordering

namespace ntt {

  static __device__ uint32_t dig_rev(uint32_t num, uint32_t log_size, bool dit)
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

  // Note: the following reorder kernels are fused with normalization for INTT
  template <typename E, typename S, uint32_t MAX_GROUP_SIZE = 80>
  static __global__ void
  reorder_digits_inplace_kernel(E* arr, uint32_t log_size, bool dit, bool is_normalize, S inverse_N)
  {
    // launch N threads
    // each thread starts from one index and calculates the corresponding group
    // if its index is the smallest number in the group -> do the memory transformation
    //  else --> do nothing

    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t next_element = idx;
    uint32_t group[MAX_GROUP_SIZE];
    group[0] = idx;

    uint32_t i = 1;
    for (; i < MAX_GROUP_SIZE;) {
      next_element = dig_rev(next_element, log_size, dit);
      if (next_element < idx) return; // not handling this group
      if (next_element == idx) break; // calculated whole group
      group[i++] = next_element;
    }

    if (i == 1) { // single element in group --> nothing to do (except maybe normalize for INTT)
      if (is_normalize) { arr[idx] = arr[idx] * inverse_N; }
      return;
    }
    --i;
    // reaching here means I am handling this group
    const E last_element_in_group = arr[group[i]];
    for (; i > 0; --i) {
      arr[group[i]] = is_normalize ? (arr[group[i - 1]] * inverse_N) : arr[group[i - 1]];
    }
    arr[idx] = is_normalize ? (last_element_in_group * inverse_N) : last_element_in_group;
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__
    void reorder_digits_kernel(E* arr, E* arr_reordered, uint32_t log_size, bool dit, bool is_normalize, S inverse_N)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t rd = tid;
    uint32_t wr = dig_rev(tid, log_size, dit);
    arr_reordered[wr] = is_normalize ? arr[rd] * inverse_N : arr[rd];
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt64(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;
    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 8;
    s_meta.ntt_block_size = 64;
    s_meta.ntt_block_id = (blockIdx.x << 3) + (strided ? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 3) : (threadIdx.x & 0x7);

    engine.loadBasicTwiddles(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride && dit) {
      engine.loadExternalTwiddlesGeneric64(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles64(internal_twiddles, strided, inv);

#pragma unroll 1
    for (uint32_t phase = 0; phase < 2; phase++) {
      engine.ntt8win();
      if (phase == 0) {
        engine.SharedData64Columns8(shmem, true, false, strided); // store
        __syncthreads();
        engine.SharedData64Rows8(shmem, false, false, strided); // load
        engine.twiddlesInternal();
      }
    }

    if (twiddle_stride && !dit) {
      engine.loadExternalTwiddlesGeneric64(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt32(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    engine.loadBasicTwiddles(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    engine.loadInternalTwiddles32(internal_twiddles, strided, inv);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData32Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData32Rows4_2(shmem, false, false, strided); // load
    engine.ntt4_2();
    if (twiddle_stride) {
      engine.loadExternalTwiddlesGeneric32(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData32(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt32dit(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    int32_t tw_log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    engine.loadBasicTwiddles(basic_twiddles, inv);
    engine.loadGlobalData32(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      engine.loadExternalTwiddlesGeneric32(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles32(internal_twiddles, strided, inv);
    engine.ntt4_2();
    engine.SharedData32Columns4_2(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData32Rows8(shmem, false, false, strided); // load
    engine.twiddlesInternal();
    engine.ntt8win();
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt16(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    engine.loadBasicTwiddles(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    engine.loadInternalTwiddles16(internal_twiddles, strided, inv);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData16Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData16Rows2_4(shmem, false, false, strided); // load
    engine.ntt2_4();
    if (twiddle_stride) {
      engine.loadExternalTwiddlesGeneric16(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.storeGlobalData16(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt16dit(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    engine.loadBasicTwiddles(basic_twiddles, inv);
    engine.loadGlobalData16(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      engine.loadExternalTwiddlesGeneric16(
        external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.loadInternalTwiddles16(internal_twiddles, strided, inv);
    engine.ntt2_4();
    engine.SharedData16Columns2_4(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData16Rows8(shmem, false, false, strided); // load
    engine.twiddlesInternal();
    engine.ntt8win();
    engine.storeGlobalData(out, data_stride, log_data_stride, log_size, strided, s_meta);
  }

  template <typename E, typename S>
  __global__ void normalize_kernel(E* data, S norm_factor)
  {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = data[tid] * norm_factor;
  }

  template <typename S>
  __global__ void generate_base_table(S basic_root, S* base_table, uint32_t skip)
  {
    S w = basic_root;
    S t = S::one();
    for (int i = 0; i < 64; i += skip) {
      base_table[i] = t;
      t = t * w;
    }
  }

  template <typename S>
  __global__ void generate_basic_twiddles(S basic_root, S* w6_table, S* basic_twiddles)
  {
    S w0 = basic_root * basic_root;
    S w1 = (basic_root + w0 * basic_root) * S::inv_log_size(1);
    S w2 = (basic_root - w0 * basic_root) * S::inv_log_size(1);
    basic_twiddles[0] = w0;
    basic_twiddles[1] = w1;
    basic_twiddles[2] = w2;
    S basic_inv = w6_table[64 - 8];
    w0 = basic_inv * basic_inv;
    w1 = (basic_inv + w0 * basic_inv) * S::inv_log_size(1);
    w2 = (basic_inv - w0 * basic_inv) * S::inv_log_size(1);
    basic_twiddles[3] = w0;
    basic_twiddles[4] = w1;
    basic_twiddles[5] = w2;
  }

  template <typename S>
  __global__ void generate_twiddle_combinations_generic(
    S* w6_table, S* w12_table, S* w18_table, S* w24_table, S* w30_table, S* external_twiddles, uint32_t log_size)
  {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t exp = tid << (30 - log_size);
    S w6, w12, w18, w24, w30;
    w6 = w6_table[exp >> 24];
    w12 = w12_table[((exp >> 18) & 0x3f)];
    w18 = w18_table[((exp >> 12) & 0x3f)];
    w24 = w24_table[((exp >> 6) & 0x3f)];
    w30 = w30_table[(exp & 0x3f)];
    S t = w6 * w12 * w18 * w24 * w30;
    external_twiddles[tid] = t;
  }

  template <typename S>
  __global__ void set_value(S* arr, int idx, S val)
  {
    arr[idx] = val;
  }

  template <typename S>
  cudaError_t generate_external_twiddles_generic(
    const S& basic_root,
    S* external_twiddles,
    S*& internal_twiddles,
    S*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream)
  {
    CHK_INIT_IF_RETURN();

    const int n = pow(2, log_size);
    CHK_IF_RETURN(cudaMallocAsync(&basic_twiddles, 6 * sizeof(S), stream));

    S* w6_table;
    S* w12_table;
    S* w18_table;
    S* w24_table;
    S* w30_table;
    CHK_IF_RETURN(cudaMallocAsync(&w6_table, sizeof(S) * 64, stream));
    CHK_IF_RETURN(cudaMallocAsync(&w12_table, sizeof(S) * 64, stream));
    CHK_IF_RETURN(cudaMallocAsync(&w18_table, sizeof(S) * 64, stream));
    CHK_IF_RETURN(cudaMallocAsync(&w24_table, sizeof(S) * 64, stream));
    CHK_IF_RETURN(cudaMallocAsync(&w30_table, sizeof(S) * 64, stream));

    // Note: for compatibility with radix-2 INTT, need ONE in last element (in addition to first element)
    set_value<<<1, 1, 0, stream>>>(external_twiddles, n /*last element idx*/, S::one());

    cudaStreamSynchronize(stream);

    S temp_root = basic_root;
    generate_base_table<<<1, 1, 0, stream>>>(basic_root, w30_table, 1 << (30 - log_size));

    if (log_size > 24)
      for (int i = 0; i < 6 - (30 - log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1, 0, stream>>>(temp_root, w24_table, 1 << (log_size > 24 ? 0 : 24 - log_size));

    if (log_size > 18)
      for (int i = 0; i < 6 - (log_size > 24 ? 0 : 24 - log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1, 0, stream>>>(temp_root, w18_table, 1 << (log_size > 18 ? 0 : 18 - log_size));

    if (log_size > 12)
      for (int i = 0; i < 6 - (log_size > 18 ? 0 : 18 - log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1, 0, stream>>>(temp_root, w12_table, 1 << (log_size > 12 ? 0 : 12 - log_size));

    if (log_size > 6)
      for (int i = 0; i < 6 - (log_size > 12 ? 0 : 12 - log_size); i++)
        temp_root = temp_root * temp_root;
    generate_base_table<<<1, 1, 0, stream>>>(temp_root, w6_table, 1 << (log_size > 6 ? 0 : 6 - log_size));

    if (log_size > 2)
      for (int i = 0; i < 3 - (log_size > 6 ? 0 : 6 - log_size); i++)
        temp_root = temp_root * temp_root;
    generate_basic_twiddles<<<1, 1, 0, stream>>>(temp_root, w6_table, basic_twiddles);

    const int NOF_BLOCKS = (log_size >= 8) ? (1 << (log_size - 8)) : 1;
    const int NOF_THREADS = (log_size >= 8) ? 256 : (1 << log_size);
    generate_twiddle_combinations_generic<<<NOF_BLOCKS, NOF_THREADS, 0, stream>>>(
      w6_table, w12_table, w18_table, w24_table, w30_table, external_twiddles, log_size);

    internal_twiddles = w6_table;

    CHK_IF_RETURN(cudaFreeAsync(w12_table, stream));
    CHK_IF_RETURN(cudaFreeAsync(w18_table, stream));
    CHK_IF_RETURN(cudaFreeAsync(w24_table, stream));
    CHK_IF_RETURN(cudaFreeAsync(w30_table, stream));

    return CHK_LAST();
  }

  template <typename E, typename S>
  cudaError_t large_ntt(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    bool inv,
    bool normalize,
    bool dit,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    if (log_size == 1 || log_size == 2 || log_size == 3 || log_size == 7) {
      throw IcicleError(IcicleError_t::InvalidArgument, "size not implemented for mixed-radix-NTT");
    }

    if (log_size == 4) {
      if (dit) {
        ntt16dit<<<1, 2, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
      } else { // dif
        ntt16<<<1, 2, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
      }
      if (normalize) normalize_kernel<<<1, 16, 0, cuda_stream>>>(out, S::inv_log_size(4));
      return CHK_LAST();
    }

    if (log_size == 5) {
      if (dit) {
        ntt32dit<<<1, 4, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
      } else { // dif
        ntt32<<<1, 4, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
      }
      if (normalize) normalize_kernel<<<1, 32, 0, cuda_stream>>>(out, S::inv_log_size(5));
      return CHK_LAST();
    }

    if (log_size == 6) {
      ntt64<<<1, 8, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
        dit);
      if (normalize) normalize_kernel<<<1, 64, 0, cuda_stream>>>(out, S::inv_log_size(6));
      return CHK_LAST();
    }

    if (log_size == 8) {
      if (dit) {
        ntt16dit<<<1, 32, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
        ntt16dit<<<1, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 16, 4, 16, true, 1,
          inv,
          dit); // we need threads 32+ although 16-31 are idle
      } else {  // dif
        ntt16<<<1, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 16, 4, 16, true, 1, inv,
          dit); // we need threads 32+ although 16-31 are idle
        ntt16<<<1, 32, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, 1, 0, 0, false, 0, inv,
          dit);
      }
      if (normalize) normalize_kernel<<<1, 256, 0, cuda_stream>>>(out, S::inv_log_size(8));
      return CHK_LAST();
    }

    // general case:
    if (dit) {
      for (int i = 0; i < 5; i++) {
        uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += STAGE_SIZES_HOST[log_size][j];
        if (stage_size == 6)
          ntt64<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        else if (stage_size == 5)
          ntt32dit<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        else if (stage_size == 4)
          ntt16dit<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
      }
    } else { // dif
      bool first_run = false, prev_stage = false;
      for (int i = 4; i >= 0; i--) {
        uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += STAGE_SIZES_HOST[log_size][j];
        first_run = stage_size && !prev_stage;
        if (stage_size == 6)
          ntt64<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        else if (stage_size == 5)
          ntt32<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        else if (stage_size == 4)
          ntt16<<<1 << (log_size - 9), 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit);
        prev_stage = stage_size;
      }
    }
    if (normalize) normalize_kernel<<<1 << (log_size - 8), 256, 0, cuda_stream>>>(out, S::inv_log_size(log_size));

    return CHK_LAST();
  }

  template <typename E, typename S>
  cudaError_t mixed_radix_ntt(
    E* d_input,
    E* d_output,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    int ntt_size,
    int max_logn,
    bool is_inverse,
    Ordering ordering,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    // TODO: can we support all orderings? Note that reversal is generally digit reverse (generalization of bit reverse)
    if (ordering != Ordering::kNN) {
      throw IcicleError(IcicleError_t::InvalidArgument, "Mixed-Radix NTT supports NN ordering only");
    }

    const int logn = int(log2(ntt_size));

    const int NOF_BLOCKS = (1 << (max(logn, 6) - 6));
    const int NOF_THREADS = min(64, 1 << logn);

    const bool reverse_input = ordering == Ordering::kNN;
    const bool is_dit = ordering == Ordering::kNN || ordering == Ordering::kRN;
    bool is_normalize = is_inverse;

    if (reverse_input) {
      // Note: fusing reorder with normalize for INTT
      const bool is_reverse_in_place = (d_input == d_output);
      if (is_reverse_in_place) {
        reorder_digits_inplace_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_output, logn, is_dit, is_normalize, S::inv_log_size(logn));
      } else {
        reorder_digits_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_input, d_output, logn, is_dit, is_normalize, S::inv_log_size(logn));
      }
      is_normalize = false;
    }

    // inplace ntt
    CHK_IF_RETURN(large_ntt(
      d_output, d_output, external_twiddles, internal_twiddles, basic_twiddles, logn, max_logn, is_inverse,
      is_normalize, is_dit, cuda_stream));

    return CHK_LAST();
  }

  // Explicit instantiation for scalar type
  template cudaError_t generate_external_twiddles_generic(
    const curve_config::scalar_t& basic_root,
    curve_config::scalar_t* external_twiddles,
    curve_config::scalar_t*& internal_twiddles,
    curve_config::scalar_t*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template cudaError_t mixed_radix_ntt<curve_config::scalar_t, curve_config::scalar_t>(
    curve_config::scalar_t* d_input,
    curve_config::scalar_t* d_output,
    curve_config::scalar_t* external_twiddles,
    curve_config::scalar_t* internal_twiddles,
    curve_config::scalar_t* basic_twiddles,
    int ntt_size,
    int max_logn,
    bool is_inverse,
    Ordering ordering,
    cudaStream_t cuda_stream);

} // namespace ntt
