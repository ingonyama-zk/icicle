
#include "appUtils/ntt/thread_ntt.cu"
#include "curves/curve_config.cuh"
#include "utils/sharedmem.cuh"
#include "appUtils/ntt/ntt.cuh" // for Ordering

namespace ntt {

  static inline __device__ uint32_t dig_rev(uint32_t num, uint32_t log_size, bool dit, bool fast_tw)
  {
    uint32_t rev_num = 0, temp, dig_len;
    if (dit) {
      for (int i = 4; i >= 0; i--) {
        dig_len = fast_tw ? STAGE_SIZES_DEVICE_FT[log_size][i] : STAGE_SIZES_DEVICE[log_size][i];
        temp = num & ((1 << dig_len) - 1);
        num = num >> dig_len;
        rev_num = rev_num << dig_len;
        rev_num = rev_num | temp;
      }
    } else {
      for (int i = 0; i < 5; i++) {
        dig_len = fast_tw ? STAGE_SIZES_DEVICE_FT[log_size][i] : STAGE_SIZES_DEVICE[log_size][i];
        temp = num & ((1 << dig_len) - 1);
        num = num >> dig_len;
        rev_num = rev_num << dig_len;
        rev_num = rev_num | temp;
      }
    }
    return rev_num;
  }

  static inline __device__ uint32_t bit_rev(uint32_t num, uint32_t log_size) { return __brev(num) >> (32 - log_size); }

  enum eRevType { None, RevToMixedRev, MixedRevToRev, NaturalToMixedRev, NaturalToRev, MixedRevToNatural };

  static __device__ uint32_t generalized_rev(uint32_t num, uint32_t log_size, bool dit, bool fast_tw, eRevType rev_type)
  {
    switch (rev_type) {
    case eRevType::RevToMixedRev:
      // R -> N -> MR
      return dig_rev(bit_rev(num, log_size), log_size, dit, fast_tw);
    case eRevType::MixedRevToRev:
      // MR -> N -> R
      return bit_rev(dig_rev(num, log_size, dit, fast_tw), log_size);
    case eRevType::NaturalToMixedRev:
    case eRevType::MixedRevToNatural:
      return dig_rev(num, log_size, dit, fast_tw);
    case eRevType::NaturalToRev:
      return bit_rev(num, log_size);
    default:
      return num;
    }
    return num;
  }

  // Note: the following reorder kernels are fused with normalization for INTT
  template <typename E, typename S, uint32_t MAX_GROUP_SIZE = 80>
  static __global__ void reorder_digits_inplace_and_normalize_kernel(
    E* arr, uint32_t log_size, bool dit, bool fast_tw, eRevType rev_type, bool is_normalize, S inverse_N)
  {
    // launch N threads (per batch element)
    // each thread starts from one index and calculates the corresponding group
    // if its index is the smallest number in the group -> do the memory transformation
    //  else --> do nothing

    const uint32_t size = 1 << log_size;
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t idx = tid % size;
    const uint32_t batch_idx = tid / size;
    if (tid==0) printf("reorder 1\n");

    uint32_t next_element = idx;
    uint32_t group[MAX_GROUP_SIZE];
    group[0] = next_element + size * batch_idx;

    uint32_t i = 1;
    for (; i < MAX_GROUP_SIZE;) {
      next_element = generalized_rev(next_element, log_size, dit, fast_tw, rev_type);
      if (next_element < idx) return; // not handling this group
      if (next_element == idx) break; // calculated whole group
      group[i++] = next_element + size * batch_idx;
    }

    --i;
    // reaching here means I am handling this group
    const E last_element_in_group = arr[group[i]];
    for (; i > 0; --i) {
      arr[group[i]] = is_normalize ? (arr[group[i - 1]] * inverse_N) : arr[group[i - 1]];
    }
    arr[group[0]] = is_normalize ? (last_element_in_group * inverse_N) : last_element_in_group;
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void reorder_digits_and_normalize_kernel(
    E* arr,
    E* arr_reordered,
    uint32_t log_size,
    bool dit,
    bool fast_tw,
    eRevType rev_type,
    bool is_normalize,
    S inverse_N)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid==0) printf("reorder 2\n");
    uint32_t rd = tid;
    uint32_t wr =
      ((tid >> log_size) << log_size) + generalized_rev(tid & ((1 << log_size) - 1), log_size, dit, fast_tw, rev_type);
    arr_reordered[wr] = is_normalize ? arr[rd] * inverse_N : arr[rd];
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void reorder_digits_and_normalize_columns_batch_kernel(
    E* arr,
    E* arr_reordered,
    uint32_t log_size,
    uint32_t batch_size,
    bool dit,
    bool fast_tw,
    eRevType rev_type,
    bool is_normalize,
    S inverse_N)
  {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid==0) printf("reorder 2.5\n");
    uint32_t rd = tid;
    uint32_t wr = generalized_rev((tid / batch_size) & ((1 << log_size) - 1), log_size, dit, fast_tw, rev_type);
    arr_reordered[wr * batch_size + (tid % batch_size)] = is_normalize ? arr[rd] * inverse_N : arr[rd];
  }

  template <typename E, typename S>
  static __global__ void batch_elementwise_mul_with_reorder(
    E* in_vec,
    int n_elements,
    int batch_size,
    S* scalar_vec,
    int step,
    int n_scalars,
    int logn,
    eRevType rev_type,
    bool dit,
    E* out_vec)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid==0) printf("reorder 3\n");
    if (tid >= n_elements * batch_size) return;
    int64_t scalar_id = tid % n_elements;
    if (rev_type != eRevType::None) scalar_id = generalized_rev(tid, logn, dit, false, rev_type);
    out_vec[tid] = *(scalar_vec + ((scalar_id * step) % n_scalars)) * in_vec[tid];
  }


template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt64batch(
    E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t batch_size,
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;
    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 8;
    s_meta.ntt_block_size = 64;
    s_meta.batch_id = (threadIdx.x & 0x7) + ((blockIdx.x % ((batch_size+7)/8)) << 3);
    
    if (s_meta.batch_id >= batch_size) return;

    s_meta.ntt_block_id = blockIdx.x / ((batch_size+7)/8);
    s_meta.ntt_inp_id = threadIdx.x >> 3;

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    // if (fast_tw)
    //   engine.loadBasicTwiddles(basic_twiddles);
    // else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalDataBatched(in, data_stride, log_data_stride, log_size, strided, s_meta, batch_size);

    // if (threadIdx.x == 0){
    //   printf("after load\n");
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }

    if (twiddle_stride && dit) {
      // if (fast_tw)
      //   engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      // else
        engine.loadExternalTwiddlesGeneric64(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    // if (fast_tw)
    //   engine.loadInternalTwiddles64(internal_twiddles, strided);
    // else
      engine.loadInternalTwiddlesGeneric64(internal_twiddles, true, inv);

#pragma unroll 1
    for (uint32_t phase = 0; phase < 2; phase++) {
      engine.ntt8win();
      // if (threadIdx.x == 0){
      // printf("after compute\n");
      // for (int i = 0; i < 8; i++)
      // {
      //   printf("%d, ",engine.X[i]);
      // }
      // printf("\n");
    // }
      if (phase == 0) {
        engine.SharedData64Columns8(shmem, true, false, true); // store
        __syncthreads();
        engine.SharedData64Rows8(shmem, false, false, true); // load
        engine.twiddlesInternal();

    //     if (threadIdx.x == 0){
    //   printf("after transpose\n");
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
      }
    }

    if (twiddle_stride && !dit) {
      // if (fast_tw)
      //   engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      // else
        engine.loadExternalTwiddlesGeneric64(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    engine.storeGlobalDataBatched(out, data_stride, log_data_stride, log_size, strided, s_meta, batch_size);
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
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;
    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 8;
    s_meta.ntt_block_size = 64;
    s_meta.ntt_block_id = (blockIdx.x << 3) + (strided ? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 3) : (threadIdx.x & 0x7);

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride && dit) {
      if (fast_tw)
        engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
        engine.loadExternalTwiddlesGeneric64(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (fast_tw)
      engine.loadInternalTwiddles64(internal_twiddles, strided);
    else
      engine.loadInternalTwiddlesGeneric64(internal_twiddles, strided, inv);

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
      if (fast_tw)
        engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
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
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (fast_tw)
      engine.loadInternalTwiddles32(internal_twiddles, strided);
    else
      engine.loadInternalTwiddlesGeneric32(internal_twiddles, strided, inv);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData32Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData32Rows4_2(shmem, false, false, strided); // load
    engine.ntt4_2();
    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles32(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
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
    uint32_t tw_log_size,
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalData32(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles32(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
        engine.loadExternalTwiddlesGeneric32(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (fast_tw)
      engine.loadInternalTwiddles32(internal_twiddles, strided);
    else
      engine.loadInternalTwiddlesGeneric32(internal_twiddles, strided, inv);
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
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalData(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (fast_tw)
      engine.loadInternalTwiddles16(internal_twiddles, strided);
    else
      engine.loadInternalTwiddlesGeneric16(internal_twiddles, strided, inv);
    engine.ntt8win();
    engine.twiddlesInternal();
    engine.SharedData16Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData16Rows2_4(shmem, false, false, strided); // load
    engine.ntt2_4();
    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles16(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
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
    uint32_t nof_ntt_blocks,
    uint32_t data_stride,
    uint32_t log_data_stride,
    uint32_t twiddle_stride,
    bool strided,
    uint32_t stage_num,
    bool inv,
    bool dit,
    bool fast_tw)
  {
    NTTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    if (s_meta.ntt_block_id >= nof_ntt_blocks) return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    engine.loadGlobalData16(in, data_stride, log_data_stride, log_size, strided, s_meta);
    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles16(external_twiddles, twiddle_stride, log_data_stride, strided, s_meta);
      else
        engine.loadExternalTwiddlesGeneric16(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (fast_tw)
      engine.loadInternalTwiddles16(internal_twiddles, strided);
    else
      engine.loadInternalTwiddlesGeneric16(internal_twiddles, strided, inv);
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

  // Generic twiddles: 1N twiddles for forward and inverse NTT
  template <typename S>
  __global__ void generate_basic_twiddles_generic(S basic_root, S* w6_table, S* basic_twiddles)
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
    generate_basic_twiddles_generic<<<1, 1, 0, stream>>>(temp_root, w6_table, basic_twiddles);

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

  // Fast-twiddles: 2N twiddles for forward, 2N for inverse
  template <typename S>
  __global__ void generate_basic_twiddles_fast_twiddles_mode(S basic_root, S* basic_twiddles)
  {
    S w0 = basic_root * basic_root;
    S w1 = (basic_root + w0 * basic_root) * S::inv_log_size(1);
    S w2 = (basic_root - w0 * basic_root) * S::inv_log_size(1);
    basic_twiddles[0] = w0;
    basic_twiddles[1] = w1;
    basic_twiddles[2] = w2;
  }

  template <typename S>
  __global__ void generate_twiddle_combinations_fast_twiddles_mode(
    S* w6_table,
    S* w12_table,
    S* w18_table,
    S* w24_table,
    S* w30_table,
    S* external_twiddles,
    uint32_t log_size,
    uint32_t prev_log_size)
  {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t exp = ((tid & ((1 << prev_log_size) - 1)) * (tid >> prev_log_size)) << (30 - log_size);
    S w6, w12, w18, w24, w30;
    w6 = w6_table[exp >> 24];
    w12 = w12_table[((exp >> 18) & 0x3f)];
    w18 = w18_table[((exp >> 12) & 0x3f)];
    w24 = w24_table[((exp >> 6) & 0x3f)];
    w30 = w30_table[(exp & 0x3f)];
    S t = w6 * w12 * w18 * w24 * w30;
    external_twiddles[tid + (1 << log_size) - 1] = t;
  }

  template <typename S>
  cudaError_t generate_external_twiddles_fast_twiddles_mode(
    const S& basic_root,
    S* external_twiddles,
    S*& internal_twiddles,
    S*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream)
  {
    CHK_INIT_IF_RETURN();

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
    CHK_IF_RETURN(cudaMallocAsync(&basic_twiddles, 3 * sizeof(S), stream));

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
    for (int i = 0; i < 3 - (log_size > 6 ? 0 : 6 - log_size); i++)
      temp_root = temp_root * temp_root;
    generate_basic_twiddles_fast_twiddles_mode<<<1, 1, 0, stream>>>(temp_root, basic_twiddles);

    for (int i = 8; i < log_size + 1; i++) {
      generate_twiddle_combinations_fast_twiddles_mode<<<1 << (i - 8), 256, 0, stream>>>(
        w6_table, w12_table, w18_table, w24_table, w30_table, external_twiddles, i, STAGE_PREV_SIZES[i]);
    }
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
    uint32_t batch_size,
    bool columns_batch,
    bool inv,
    bool normalize,
    bool dit,
    bool fast_tw,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    if (log_size == 1 || log_size == 2 || log_size == 3 || log_size == 7) {
      throw IcicleError(IcicleError_t::InvalidArgument, "size not implemented for mixed-radix-NTT");
    }

    if (log_size == 4) {
      const int NOF_THREADS = min(64, 2 * batch_size);
      const int NOF_BLOCKS = (2 * batch_size + NOF_THREADS - 1) / NOF_THREADS;

      if (dit) {
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size, 1, 0, 0,
          false, 0, inv, dit, fast_tw);
      } else { // dif
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size, 1, 0, 0,
          false, 0, inv, dit, fast_tw);
      }
      if (normalize) normalize_kernel<<<batch_size, 16, 0, cuda_stream>>>(out, S::inv_log_size(4));
      return CHK_LAST();
    }

    if (log_size == 5) {
      const int NOF_THREADS = min(64, 4 * batch_size);
      const int NOF_BLOCKS = (4 * batch_size + NOF_THREADS - 1) / NOF_THREADS;
      if (dit) {
        ntt32dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size, 1, 0, 0,
          false, 0, inv, dit, fast_tw);
      } else { // dif
        ntt32<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size, 1, 0, 0,
          false, 0, inv, dit, fast_tw);
      }
      if (normalize) normalize_kernel<<<batch_size, 32, 0, cuda_stream>>>(out, S::inv_log_size(5));
      return CHK_LAST();
    }

    if (log_size == 6) {
      if (columns_batch){
        const int NOF_THREADS = 64;
        const int NOF_BLOCKS = (batch_size+7)/8;
        ntt64batch<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size, 1,
        1 ,0, 0, false, 0, inv, dit, fast_tw);
        cudaDeviceSynchronize();
        printf("batched err %d\n", cudaGetLastError());
        return CHK_LAST();
      }

      const int NOF_THREADS = min(64, 8 * batch_size);
      const int NOF_BLOCKS = (8 * batch_size + NOF_THREADS - 1) / NOF_THREADS;
      ntt64<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size, batch_size,
        1 ,0, 0, false, 0, inv, dit, fast_tw);
      if (normalize) normalize_kernel<<<batch_size, 64, 0, cuda_stream>>>(out, S::inv_log_size(6));
      return CHK_LAST();
    }

    if (log_size == 8) {
      const int NOF_THREADS = 64;
      const int NOF_BLOCKS = (32 * batch_size + NOF_THREADS - 1) / NOF_THREADS;
      if (dit) {
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          (1 << log_size - 4) * batch_size, 1, 0, 0, false, 0, inv, dit, fast_tw);
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          (1 << log_size - 4) * batch_size, 16, 4, 16, true, 1, inv, dit, fast_tw);
      } else { // dif
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          (1 << log_size - 4) * batch_size, 16, 4, 16, true, 1, inv, dit, fast_tw);
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          (1 << log_size - 4) * batch_size, 1, 0, 0, false, 0, inv, dit, fast_tw);
      }
      if (normalize) normalize_kernel<<<batch_size, 256, 0, cuda_stream>>>(out, S::inv_log_size(8));
      return CHK_LAST();
    }

    // general case:
    uint32_t nof_blocks = (1 << (log_size - 9)) * batch_size;
    if (dit) {
      for (int i = 0; i < 5; i++) {
        uint32_t stage_size = fast_tw ? STAGE_SIZES_HOST_FT[log_size][i] : STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += fast_tw ? STAGE_SIZES_HOST_FT[log_size][j] : STAGE_SIZES_HOST[log_size][j];
        if (stage_size == 6)
          ntt64<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 6) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
        else if (stage_size == 5)
          ntt32dit<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 5) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
        else if (stage_size == 4)
          ntt16dit<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 4) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
      }
    } else { // dif
      bool first_run = false, prev_stage = false;
      for (int i = 4; i >= 0; i--) {
        uint32_t stage_size = fast_tw ? STAGE_SIZES_HOST_FT[log_size][i] : STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += fast_tw ? STAGE_SIZES_HOST_FT[log_size][j] : STAGE_SIZES_HOST[log_size][j];
        first_run = stage_size && !prev_stage;
        if (stage_size == 6)
          ntt64<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 6) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
        else if (stage_size == 5)
          ntt32<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 5) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
        else if (stage_size == 4)
          ntt16<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            (1 << log_size - 4) * batch_size, 1 << stride_log, stride_log, i ? (1 << stride_log) : 0, i, i, inv, dit,
            fast_tw);
        prev_stage = stage_size;
      }
    }
    if (normalize)
      normalize_kernel<<<(1 << (log_size - 8)) * batch_size, 256, 0, cuda_stream>>>(out, S::inv_log_size(log_size));

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
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    Ordering ordering,
    S* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    const int logn = int(log2(ntt_size));
    const int NOF_BLOCKS = ((1 << logn) * batch_size + 64 - 1) / 64;
    const int NOF_THREADS = min(64, (1 << logn) * batch_size);

    bool is_normalize = is_inverse;
    const bool is_on_coset = (coset_gen_index != 0) || arbitrary_coset;
    const int n_twiddles = 1 << max_logn;
    // Note: for evaluation on coset, need to reorder the coset too to match the data for element-wise multiplication
    eRevType reverse_input = None, reverse_output = None, reverse_coset = None;
    bool dit = false;
    switch (ordering) {
    case Ordering::kNN:
      reverse_input = eRevType::NaturalToMixedRev;
      dit = true;
      break;
    case Ordering::kRN:
      reverse_input = eRevType::RevToMixedRev;
      dit = true;
      reverse_coset = is_inverse ? eRevType::None : eRevType::NaturalToRev;
      break;
    case Ordering::kNR:
      reverse_output = eRevType::MixedRevToRev;
      reverse_coset = is_inverse ? eRevType::NaturalToRev : eRevType::None;
      break;
    case Ordering::kRR:
      reverse_input = eRevType::RevToMixedRev;
      dit = true;
      reverse_output = eRevType::NaturalToRev;
      reverse_coset = eRevType::NaturalToRev;
      break;
    case Ordering::kMN:
      dit = true;
      reverse_coset = is_inverse ? None : eRevType::NaturalToMixedRev;
      break;
    case Ordering::kNM:
      reverse_coset = is_inverse ? eRevType::NaturalToMixedRev : eRevType::None;
      break;
    }

    if (is_on_coset && !is_inverse) {
      batch_elementwise_mul_with_reorder<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_input, ntt_size, batch_size, arbitrary_coset ? arbitrary_coset : external_twiddles,
        arbitrary_coset ? 1 : coset_gen_index, n_twiddles, logn, reverse_coset, dit, d_output);

      d_input = d_output;
    }

    if (reverse_input != eRevType::None) {
      const bool is_reverse_in_place = (d_input == d_output);
      if (is_reverse_in_place) {
        reorder_digits_inplace_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_output, logn, dit, fast_tw, reverse_input, is_normalize, S::inv_log_size(logn));
      } else {
        if (columns_batch){
          reorder_digits_and_normalize_columns_batch_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_input, d_output, logn, batch_size, dit, fast_tw, reverse_input, is_normalize, S::inv_log_size(logn));
        }
        else {
          reorder_digits_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_input, d_output, logn, dit, fast_tw, reverse_input, is_normalize, S::inv_log_size(logn));
        }
      }
      is_normalize = false;
      d_input = d_output;
    }

    // inplace ntt
    CHK_IF_RETURN(large_ntt(
      d_input, d_output, external_twiddles, internal_twiddles, basic_twiddles, logn, max_logn, batch_size, columns_batch, is_inverse,
      (is_normalize && reverse_output == eRevType::None), dit, fast_tw, cuda_stream));

    if (reverse_output != eRevType::None) {
      reorder_digits_inplace_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_output, logn, dit, fast_tw, reverse_output, is_normalize, S::inv_log_size(logn));
    }

    if (is_on_coset && is_inverse) {
      batch_elementwise_mul_with_reorder<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_output, ntt_size, batch_size, arbitrary_coset ? arbitrary_coset : external_twiddles + n_twiddles,
        arbitrary_coset ? 1 : -coset_gen_index, n_twiddles, logn, reverse_coset, dit, d_output);
    }

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

  template cudaError_t generate_external_twiddles_fast_twiddles_mode(
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
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    Ordering ordering,
    curve_config::scalar_t* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream);

} // namespace ntt
