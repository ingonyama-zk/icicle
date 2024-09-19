#include "fields/field_config.cuh"

using namespace field_config;

#include "thread_ntt.cu"
#include "gpu-utils/sharedmem.cuh"
#include "ntt/ntt.cuh" // for ntt::Ordering

namespace mxntt {

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
    E* arr,
    uint32_t log_size,
    bool columns_batch,
    uint32_t batch_size,
    bool dit,
    bool fast_tw,
    eRevType rev_type,
    bool is_normalize,
    S inverse_N)
  {
    // launch N threads (per batch element)
    // each thread starts from one index and calculates the corresponding group
    // if its index is the smallest number in the group -> do the memory transformation
    //  else --> do nothing

    const uint64_t size = 1UL << log_size;
    const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const uint64_t idx = columns_batch ? tid / batch_size : tid % size;
    const uint64_t batch_idx = columns_batch ? tid % batch_size : tid / size;
    if (tid >= uint64_t(size) * batch_size) return;

    uint64_t next_element = idx;
    uint64_t group[MAX_GROUP_SIZE];
    group[0] = columns_batch ? next_element * batch_size + batch_idx : next_element + size * batch_idx;

    uint32_t i = 1;
    for (; i < MAX_GROUP_SIZE;) {
      next_element = generalized_rev(next_element, log_size, dit, fast_tw, rev_type);
      if (next_element < idx) return; // not handling this group
      if (next_element == idx) break; // calculated whole group
      group[i++] = columns_batch ? next_element * batch_size + batch_idx : next_element + size * batch_idx;
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
    const E* arr,
    E* arr_reordered,
    uint32_t log_size,
    bool columns_batch,
    uint32_t batch_size,
    uint32_t columns_batch_size,
    bool dit,
    bool fast_tw,
    eRevType rev_type,
    bool is_normalize,
    S inverse_N)
  {
    const uint64_t size = 1UL << log_size;
    const uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= uint64_t(size) * batch_size) return;

    uint64_t rd = tid;
    uint64_t wr = (columns_batch ? 0 : ((tid >> log_size) << log_size)) +
                  generalized_rev((tid / columns_batch_size) & (size - 1), log_size, dit, fast_tw, rev_type);
    arr_reordered[wr * columns_batch_size + (tid % columns_batch_size)] = is_normalize ? arr[rd] * inverse_N : arr[rd];
  }

  template <typename E, typename S>
  static __global__ void batch_elementwise_mul_with_reorder_kernel(
    const E* in_vec,
    uint32_t size,
    bool columns_batch,
    uint32_t batch_size,
    uint32_t columns_batch_size,
    S* scalar_vec,
    int step,
    uint32_t n_scalars,
    uint32_t log_size,
    eRevType rev_type,
    bool fast_tw,
    E* out_vec)
  {
    uint64_t tid = uint64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= uint64_t(size) * batch_size) return;
    int64_t scalar_id = (tid / columns_batch_size) % size;
    if (rev_type != eRevType::None) {
      // Note: when we multiply an in_vec that is mixed (by DIF (I)NTT), we want to shuffle the
      // scalars the same way (then multiply element-wise). This would be a DIT-digit-reverse shuffle. (this is
      // confusing but) BUT to avoid shuffling the scalars, we instead want to ask which element in the non-shuffled
      // vec is now placed at index tid, which is the opposite of a DIT-digit-reverse --> this is the DIF-digit-reverse.
      // Therefore we use the DIF-digit-reverse to know which element moved to index tid and use it to access the
      // corresponding element in scalars vec.
      const bool dif = rev_type == eRevType::NaturalToMixedRev;
      scalar_id = generalized_rev((tid / columns_batch_size) & (size - 1), log_size, !dif, fast_tw, rev_type);
    }
    out_vec[tid] = *(scalar_vec + ((scalar_id * step) % n_scalars)) * in_vec[tid];
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt64(
    const E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    s_meta.ntt_block_id = columns_batch_size ? blockIdx.x / ((columns_batch_size + 7) / 8)
                                             : (blockIdx.x << 3) + (strided ? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 3) : (threadIdx.x & 0x7);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0x7) + ((blockIdx.x % ((columns_batch_size + 7) / 8)) << 3) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);
    if (columns_batch_size)
      engine.loadGlobalDataColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

    if (twiddle_stride && dit) {
      if (fast_tw)
        engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, s_meta);
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
        engine.loadExternalTwiddles64(external_twiddles, twiddle_stride, log_data_stride, s_meta);
      else
        engine.loadExternalTwiddlesGeneric64(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (columns_batch_size)
      engine.storeGlobalDataColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt32(
    const E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    s_meta.ntt_block_id = columns_batch_size ? blockIdx.x / ((columns_batch_size + 15) / 16)
                                             : (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0xf) + ((blockIdx.x % ((columns_batch_size + 15) / 16)) << 4) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);

    if (columns_batch_size)
      engine.loadGlobalDataColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

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
        engine.loadExternalTwiddles32(external_twiddles, twiddle_stride, log_data_stride, s_meta);
      else
        engine.loadExternalTwiddlesGeneric32(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (columns_batch_size)
      engine.storeGlobalData32ColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData32(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt32dit(
    const E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    s_meta.ntt_block_id = columns_batch_size ? blockIdx.x / ((columns_batch_size + 15) / 16)
                                             : (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0xf) + ((blockIdx.x % ((columns_batch_size + 15) / 16)) << 4) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);

    if (columns_batch_size)
      engine.loadGlobalData32ColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData32(in, data_stride, log_data_stride, strided, s_meta);
    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles32(external_twiddles, twiddle_stride, log_data_stride, s_meta);
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
    if (columns_batch_size)
      engine.storeGlobalDataColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData(out, data_stride, log_data_stride, strided, s_meta);
  }

#define DCCT
#ifdef DCCT
  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt64_dcct(
    const E* in,
    E* out,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    DCCTEngine<E, S> engine;
    stage_metadata s_meta;
    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 8;
    s_meta.ntt_block_size = 64;
    s_meta.ntt_block_id = columns_batch_size ? blockIdx.x / ((columns_batch_size + 7) / 8)
                                             : (blockIdx.x << 3) + (strided ? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 3) : (threadIdx.x & 0x7);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0x7) + ((blockIdx.x % ((columns_batch_size + 7) / 8)) << 3) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (columns_batch_size)
      engine.loadGlobalDataColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

    // printf(
    //   "T Before: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
    //   threadIdx.x,
    //   engine.X[0].limbs_storage.limbs[0],
    //   engine.X[1].limbs_storage.limbs[0],
    //   engine.X[2].limbs_storage.limbs[0],
    //   engine.X[3].limbs_storage.limbs[0],
    //   engine.X[4].limbs_storage.limbs[0],
    //   engine.X[5].limbs_storage.limbs[0],
    //   engine.X[6].limbs_storage.limbs[0],
    //   engine.X[7].limbs_storage.limbs[0]
    // );
    engine.loadBasicTwiddlesGeneric64(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, false);
#pragma unroll 1
    for (uint32_t phase = 0; phase < 2; phase++) {
      engine.ntt8();

      if (phase == 0) {
        engine.loadBasicTwiddlesGeneric64(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, true);
        engine.SharedData64Columns8(shmem, true, false, strided); // store
        __syncthreads();
        engine.SharedData64Rows8(shmem, false, false, strided); // load
        // printf(
        //   "T AFTER: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
        //   threadIdx.x,
        //   engine.X[0].limbs_storage.limbs[0],
        //   engine.X[1].limbs_storage.limbs[0],
        //   engine.X[2].limbs_storage.limbs[0],
        //   engine.X[3].limbs_storage.limbs[0],
        //   engine.X[4].limbs_storage.limbs[0],
        //   engine.X[5].limbs_storage.limbs[0],
        //   engine.X[6].limbs_storage.limbs[0],
        //   engine.X[7].limbs_storage.limbs[0]
        // );
      }
    }
    // printf(
    //   "T AFTER Second NTT: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
    //   threadIdx.x,
    //   engine.X[0].limbs_storage.limbs[0],
    //   engine.X[1].limbs_storage.limbs[0],
    //   engine.X[2].limbs_storage.limbs[0],
    //   engine.X[3].limbs_storage.limbs[0],
    //   engine.X[4].limbs_storage.limbs[0],
    //   engine.X[5].limbs_storage.limbs[0],
    //   engine.X[6].limbs_storage.limbs[0],
    //   engine.X[7].limbs_storage.limbs[0]
    // );

    if (columns_batch_size)
      engine.storeGlobalDataColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt32_dcct(
    const E* in,
    E* out,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    DCCTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 4;
    s_meta.ntt_block_size = 32;
    s_meta.ntt_block_id = columns_batch_size ? blockIdx.x / ((columns_batch_size + 15) / 16)
                                             : (blockIdx.x << 4) + (strided ? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 4) : (threadIdx.x & 0x3);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0xf) + ((blockIdx.x % ((columns_batch_size + 15) / 16)) << 4) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    engine.loadBasicTwiddlesGeneric32(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, false);

    if (columns_batch_size)
      engine.loadGlobalDataColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

    engine.ntt8();
    engine.SharedData32Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData32Rows4_2(shmem, false, false, strided); // load
    engine.loadBasicTwiddlesGeneric32(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, true);
    engine.ntt4_2();

    if (columns_batch_size)
      engine.storeGlobalData32ColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData32(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt16_dcct(
    const E* in,
    E* out,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    DCCTEngine<E, S> engine;
    stage_metadata s_meta;

    SharedMemory<E> smem;
    E* shmem = smem.getPointer();

    s_meta.th_stride = 2;
    s_meta.ntt_block_size = 16;
    s_meta.ntt_block_id = columns_batch_size
                            ? blockIdx.x / ((columns_batch_size + 31) / 32)
                            : (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0x1f) + ((blockIdx.x % ((columns_batch_size + 31) / 32)) << 5) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    engine.loadBasicTwiddlesGeneric16(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, false);
    engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

    // if (s_meta.ntt_block_id < 4) {
      // printf(
      //   "Before T: %d, I: %d, B: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
      //   threadIdx.x,
      //   s_meta.ntt_inp_id,
      //   s_meta.ntt_block_id,
      //   engine.X[0].limbs_storage.limbs[0],
      //   engine.X[1].limbs_storage.limbs[0],
      //   engine.X[2].limbs_storage.limbs[0],
      //   engine.X[3].limbs_storage.limbs[0],
      //   engine.X[4].limbs_storage.limbs[0],
      //   engine.X[5].limbs_storage.limbs[0],
      //   engine.X[6].limbs_storage.limbs[0],
      //   engine.X[7].limbs_storage.limbs[0]
      // );
    // }

    engine.ntt8();
    // if (s_meta.ntt_block_id < 2) {
    //   printf(
    //     "T BEFORE Transpose: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
    //     threadIdx.x,
    //     engine.X[0].limbs_storage.limbs[0],
    //     engine.X[1].limbs_storage.limbs[0],
    //     engine.X[2].limbs_storage.limbs[0],
    //     engine.X[3].limbs_storage.limbs[0],
    //     engine.X[4].limbs_storage.limbs[0],
    //     engine.X[5].limbs_storage.limbs[0],
    //     engine.X[6].limbs_storage.limbs[0],
    //     engine.X[7].limbs_storage.limbs[0]
    //   );
    // }

    engine.SharedData16Columns8(shmem, true, false, strided); // store
    __syncthreads();
    engine.SharedData16Rows2_4(shmem, false, false, strided); // load

    // if (s_meta.ntt_block_id < 2) {
    //   printf(
    //     "T AFTER Transpose: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
    //     threadIdx.x,
    //     engine.X[0].limbs_storage.limbs[0],
    //     engine.X[1].limbs_storage.limbs[0],
    //     engine.X[2].limbs_storage.limbs[0],
    //     engine.X[3].limbs_storage.limbs[0],
    //     engine.X[4].limbs_storage.limbs[0],
    //     engine.X[5].limbs_storage.limbs[0],
    //     engine.X[6].limbs_storage.limbs[0],
    //     engine.X[7].limbs_storage.limbs[0]
    //   );
    // }

    engine.loadBasicTwiddlesGeneric16(basic_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv, true);
    engine.ntt2_4();

    // if (s_meta.ntt_block_id < 2) {
      // printf(
      //   "T FINAL: %d\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n0x%x\n",
      //   threadIdx.x,
      //   engine.X[0].limbs_storage.limbs[0],
      //   engine.X[1].limbs_storage.limbs[0],
      //   engine.X[2].limbs_storage.limbs[0],
      //   engine.X[3].limbs_storage.limbs[0],
      //   engine.X[4].limbs_storage.limbs[0],
      //   engine.X[5].limbs_storage.limbs[0],
      //   engine.X[6].limbs_storage.limbs[0],
      //   engine.X[7].limbs_storage.limbs[0]
      // );
    // }

    engine.storeGlobalData16(out, data_stride, log_data_stride, strided, s_meta);
  }
#endif

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt16(
    const E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    s_meta.ntt_block_id = columns_batch_size
                            ? blockIdx.x / ((columns_batch_size + 31) / 32)
                            : (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0x1f) + ((blockIdx.x % ((columns_batch_size + 31) / 32)) << 5) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);

    if (columns_batch_size)
      engine.loadGlobalDataColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData(in, data_stride, log_data_stride, strided, s_meta);

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
        engine.loadExternalTwiddles16(external_twiddles, twiddle_stride, log_data_stride, s_meta);
      else
        engine.loadExternalTwiddlesGeneric16(
          external_twiddles, twiddle_stride, log_data_stride, s_meta, tw_log_size, inv);
      engine.twiddlesExternal();
    }
    if (columns_batch_size)
      engine.storeGlobalData16ColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData16(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __launch_bounds__(64) __global__ void ntt16dit(
    const E* in,
    E* out,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    uint32_t log_size,
    uint32_t tw_log_size,
    uint32_t columns_batch_size,
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
    s_meta.ntt_block_id = columns_batch_size
                            ? blockIdx.x / ((columns_batch_size + 31) / 32)
                            : (blockIdx.x << 5) + (strided ? (threadIdx.x & 0x1f) : (threadIdx.x >> 1));
    s_meta.ntt_inp_id = strided ? (threadIdx.x >> 5) : (threadIdx.x & 0x1);

    s_meta.batch_id =
      columns_batch_size ? (threadIdx.x & 0x1f) + ((blockIdx.x % ((columns_batch_size + 31) / 32)) << 5) : 0;
    if (s_meta.ntt_block_id >= nof_ntt_blocks || (columns_batch_size > 0 && s_meta.batch_id >= columns_batch_size))
      return;

    if (fast_tw)
      engine.loadBasicTwiddles(basic_twiddles);
    else
      engine.loadBasicTwiddlesGeneric(basic_twiddles, inv);

    if (columns_batch_size)
      engine.loadGlobalData16ColumnBatch(in, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.loadGlobalData16(in, data_stride, log_data_stride, strided, s_meta);

    if (twiddle_stride) {
      if (fast_tw)
        engine.loadExternalTwiddles16(external_twiddles, twiddle_stride, log_data_stride, s_meta);
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
    if (columns_batch_size)
      engine.storeGlobalDataColumnBatch(out, data_stride, log_data_stride, s_meta, columns_batch_size);
    else
      engine.storeGlobalData(out, data_stride, log_data_stride, strided, s_meta);
  }

  template <typename E, typename S>
  __global__ void normalize_kernel(E* data, S norm_factor, uint64_t size)
  {
    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    data[tid] = data[tid] * norm_factor;
  }

#ifdef DCCT
  template <typename S, typename R>
  __global__ void
  generate_dcct_twiddles_layer(S* external_twiddles, R basic_root, R step, uint32_t stage, uint32_t stage_rev, bool is_first)
  {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    R twiddle = basic_root * (R::pow(step, bit_rev(tid / (1 << stage_rev), stage)));
    external_twiddles[tid] = is_first ? twiddle.imaginary : twiddle.real;
  }

  template <typename S, typename R>
  cudaError_t generate_twiddles_dcct(
    const R& basic_root,
    S* basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream)
  {
    R step = R::pow(basic_root, 4);
    R temp_root = basic_root;

    int stage = log_size - 1;
    uint32_t stage_rev = 0;
    S* stage_ptr = basic_twiddles + (stage * (1 << stage));
    const int NOF_BLOCKS = (stage >= 8) ? (1 << (stage - 8)) : 1;
    const int NOF_THREADS = (stage >= 8) ? 256 : (1 << stage);
    // std::cout << "Stage: " << stage << "; nof_blocks: " << NOF_BLOCKS << "; nof_threads: " << NOF_THREADS << "; step:
    // " << step << "; temp_root: " << temp_root <<"; stage_ptr: " << stage_ptr << std::endl;
    generate_dcct_twiddles_layer<<<NOF_BLOCKS, NOF_THREADS, 0, stream>>>(stage_ptr, temp_root, step, stage, stage_rev, true);
    CHK_IF_RETURN(cudaPeekAtLastError());

    for (--stage; stage >= 0; stage--) {
      stage_ptr -= 1 << (log_size - 1);
      stage_rev++;
      // std::cout << "Stage: " << stage << "; nof_blocks: " << NOF_BLOCKS << "; nof_threads: " << NOF_THREADS << ";
      // step: " << step << "; temp_root: " << temp_root <<"; stage_ptr: " << stage_ptr<< std::endl;
      generate_dcct_twiddles_layer<<<NOF_BLOCKS, NOF_THREADS, 0, stream>>>(stage_ptr, temp_root, step, stage, stage_rev, false);
      CHK_IF_RETURN(cudaPeekAtLastError());

      temp_root = R::sqr(temp_root);
      step = R::sqr(step);
    }

    return CHK_LAST();
  }
// #else
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

#endif
  template <typename E, typename S>
  cudaError_t large_ntt(
    const E* in,
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
      const int NOF_THREADS = columns_batch ? 64 : min(64, 2 * batch_size);
      const int NOF_BLOCKS =
        columns_batch ? ((batch_size + 31) / 32) : (2 * batch_size + NOF_THREADS - 1) / NOF_THREADS;
      if (dit) {
#ifdef DCCT
        ntt16_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#else
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#endif
      } else { // dif
#ifdef DCCT
        ntt16_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#else
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#endif
      }
      if (normalize)
        normalize_kernel<<<batch_size, 16, 0, cuda_stream>>>(out, S::inv_log_size(4), (1UL << log_size) * batch_size);
      return CHK_LAST();
    }

    if (log_size == 5) {
      const int NOF_THREADS = columns_batch ? 64 : min(64, 4 * batch_size);
      const int NOF_BLOCKS =
        columns_batch ? ((batch_size + 15) / 16) : (4 * batch_size + NOF_THREADS - 1) / NOF_THREADS;
#ifdef DCCT
      ntt32_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, basic_twiddles, log_size, tw_log_size,
        columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#else
      if (dit) {
        ntt32dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
      } else { // dif
        ntt32<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
      }
#endif
      if (normalize)
        normalize_kernel<<<batch_size, 32, 0, cuda_stream>>>(out, S::inv_log_size(5), (1UL << log_size) * batch_size);
      return CHK_LAST();
    }

    if (log_size == 6) {
      const int NOF_THREADS = columns_batch ? 64 : min(64, 8 * batch_size);
      const int NOF_BLOCKS =
        columns_batch ? ((batch_size + 7) / 8) : ((8 * batch_size + NOF_THREADS - 1) / NOF_THREADS);
#ifdef DCCT
      std::cout << NOF_BLOCKS << " " << NOF_THREADS << std::endl;
      ntt64_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, basic_twiddles, log_size, tw_log_size,
        columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
      CHK_IF_RETURN(cudaPeekAtLastError());
      CHK_IF_RETURN(cudaDeviceSynchronize());
#else
      ntt64<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
        in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
        columns_batch ? batch_size : 0, columns_batch ? 1 : batch_size, 1, 0, 0, columns_batch, 0, inv, dit, fast_tw);
#endif
      if (normalize)
        normalize_kernel<<<batch_size, 64, 0, cuda_stream>>>(out, S::inv_log_size(6), (1UL << log_size) * batch_size);
      return CHK_LAST();
    }

    if (log_size == 8) {
      const int NOF_THREADS = 64;
      const int NOF_BLOCKS =
        columns_batch ? ((batch_size + 31) / 32 * 16) : ((32 * batch_size + NOF_THREADS - 1) / NOF_THREADS);
      if (dit) {
#ifdef DCCT
        ntt16_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 16, 4, 16, true, 1,
          inv, dit, fast_tw);
        ntt16_dcct<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1, 0, 0, columns_batch, 0,
          inv, dit, fast_tw);
 #else
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1, 0, 0,
          columns_batch, 0, inv, dit, fast_tw);
        ntt16dit<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 16, 4, 16, true, 1,
          inv, dit, fast_tw);
#endif
      } else { // dif
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 16, 4, 16, true, 1,
          inv, dit, fast_tw);
        ntt16<<<NOF_BLOCKS, NOF_THREADS, 8 * 64 * sizeof(E), cuda_stream>>>(
          out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
          columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1, 0, 0,
          columns_batch, 0, inv, dit, fast_tw);
      }
      if (normalize)
        normalize_kernel<<<batch_size, 256, 0, cuda_stream>>>(out, S::inv_log_size(8), (1UL << log_size) * batch_size);
      return CHK_LAST();
    }

    // general case:
    uint32_t nof_blocks = (1UL << (log_size - 9)) * (columns_batch ? ((batch_size + 31) / 32) * 32 : batch_size);
    if (dit) {
      for (int i = 0; i < 5; i++) {
        uint32_t stage_size = fast_tw ? STAGE_SIZES_HOST_FT[log_size][i] : STAGE_SIZES_HOST[log_size][i];
        uint32_t stride_log = 0;
        for (int j = 0; j < i; j++)
          stride_log += fast_tw ? STAGE_SIZES_HOST_FT[log_size][j] : STAGE_SIZES_HOST[log_size][j];
#ifdef DCCT
        if (stage_size == 6)
          ntt64_dcct<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 6) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 5)
          ntt32_dcct<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 5) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 4)
          ntt16_dcct<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        
        CHK_IF_RETURN(cudaDeviceSynchronize());
#else
        if (stage_size == 6)
          ntt64<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 6) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 5)
          ntt32dit<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 5) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 4)
          ntt16dit<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            i ? out : in, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);

#endif
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
            columns_batch ? batch_size : 0, (1 << log_size - 6) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 5)
          ntt32<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 5) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        else if (stage_size == 4)
          ntt16<<<nof_blocks, 64, 8 * 64 * sizeof(E), cuda_stream>>>(
            first_run ? in : out, out, external_twiddles, internal_twiddles, basic_twiddles, log_size, tw_log_size,
            columns_batch ? batch_size : 0, (1 << log_size - 4) * (columns_batch ? 1 : batch_size), 1 << stride_log,
            stride_log, i ? (1 << stride_log) : 0, i || columns_batch, i, inv, dit, fast_tw);
        prev_stage = stage_size;
      }
    }
    if (normalize)
      normalize_kernel<<<(1 << (log_size - 8)) * batch_size, 256, 0, cuda_stream>>>(
        out, S::inv_log_size(log_size), (1UL << log_size) * batch_size);

    return CHK_LAST();
  }

  template <typename E, typename S>
  cudaError_t mixed_radix_ntt(
    const E* d_input,
    E* d_output,
    S* external_twiddles,
    S* internal_twiddles,
    S* basic_twiddles,
    S* linear_twiddle, // twiddles organized as [1,w,w^2,...] for coset-eval in fast-tw mode
    int ntt_size,
    int max_logn,
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    ntt::Ordering ordering,
    S* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    const uint64_t total_nof_elements = uint64_t(ntt_size) * batch_size;
    const uint64_t logn = uint64_t(log2(ntt_size));
    const uint64_t NOF_BLOCKS_64b = (total_nof_elements + 64 - 1) / 64;
    const uint32_t NOF_THREADS = total_nof_elements < 64 ? total_nof_elements : 64;
    // CUDA grid is 32b fields. Assert that I don't need a larger grid.
    const uint32_t NOF_BLOCKS = NOF_BLOCKS_64b;
    if (NOF_BLOCKS != NOF_BLOCKS_64b) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "NTT dimensions (ntt_size, batch) are too large. Unsupported!");
    }

    bool is_normalize = is_inverse;
    const bool is_on_coset = (coset_gen_index != 0) || arbitrary_coset;
    const uint32_t n_twiddles = 1U << max_logn;
    // Note: for evaluation on coset, need to reorder the coset too to match the data for element-wise multiplication
    eRevType reverse_input = None, reverse_output = None, reverse_coset = None;
    bool dit = false;
    switch (ordering) {
    case ntt::Ordering::kNN:
      reverse_input = eRevType::NaturalToMixedRev;
      dit = true;
      break;
    case ntt::Ordering::kRN:
      reverse_input = eRevType::RevToMixedRev;
      dit = true;
      reverse_coset = is_inverse ? eRevType::None : eRevType::NaturalToRev;
      break;
    case ntt::Ordering::kNR:
      reverse_output = eRevType::MixedRevToRev;
      reverse_coset = is_inverse ? eRevType::NaturalToRev : eRevType::None;
      break;
    case ntt::Ordering::kRR:
      reverse_input = eRevType::RevToMixedRev;
      dit = true;
      reverse_output = eRevType::NaturalToRev;
      reverse_coset = eRevType::NaturalToRev;
      break;
    case ntt::Ordering::kMN:
      dit = true;
      reverse_coset = is_inverse ? None : eRevType::NaturalToMixedRev;
      break;
    case ntt::Ordering::kNM:
      reverse_coset = is_inverse ? eRevType::NaturalToMixedRev : eRevType::None;
      break;
    }

    if (is_on_coset && !is_inverse) {
      batch_elementwise_mul_with_reorder_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_input, ntt_size, columns_batch, batch_size, columns_batch ? batch_size : 1,
        arbitrary_coset ? arbitrary_coset : linear_twiddle, arbitrary_coset ? 1 : coset_gen_index, n_twiddles, logn,
        reverse_coset, fast_tw, d_output);

      d_input = d_output;
    }

    if (reverse_input != eRevType::None) {
      const bool is_reverse_in_place = (d_input == d_output);
      if (is_reverse_in_place) {
        reorder_digits_inplace_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_output, logn, columns_batch, batch_size, dit, fast_tw, reverse_input, is_normalize, S::inv_log_size(logn));
      } else {
        reorder_digits_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
          d_input, d_output, logn, columns_batch, batch_size, columns_batch ? batch_size : 1, dit, fast_tw,
          reverse_input, is_normalize, S::inv_log_size(logn));
      }
      is_normalize = false;
      d_input = d_output;
    }

    std::cout << "Entering large ntt" << std::endl;
    // inplace ntt
    CHK_IF_RETURN(large_ntt(
      d_input, d_output, external_twiddles, internal_twiddles, basic_twiddles, logn, max_logn, batch_size,
      columns_batch, is_inverse, (is_normalize && reverse_output == eRevType::None), dit, fast_tw, cuda_stream));

    if (reverse_output != eRevType::None) {
      reorder_digits_inplace_and_normalize_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_output, logn, columns_batch, batch_size, dit, fast_tw, reverse_output, is_normalize, S::inv_log_size(logn));
    }

    if (is_on_coset && is_inverse) {
      batch_elementwise_mul_with_reorder_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, cuda_stream>>>(
        d_output, ntt_size, columns_batch, batch_size, columns_batch ? batch_size : 1,
        arbitrary_coset ? arbitrary_coset : linear_twiddle + n_twiddles, arbitrary_coset ? 1 : -coset_gen_index,
        n_twiddles, logn, reverse_coset, fast_tw, d_output);
    }

    return CHK_LAST();
  }

#ifdef DCCT
  template cudaError_t generate_twiddles_dcct(
    const quad_extension_t& basic_root,
    scalar_t* basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);
#else
  // Explicit instantiation for scalar type
  template cudaError_t generate_external_twiddles_generic(
    const scalar_t& basic_root,
    scalar_t* external_twiddles,
    scalar_t*& internal_twiddles,
    scalar_t*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template cudaError_t generate_external_twiddles_fast_twiddles_mode(
    const scalar_t& basic_root,
    scalar_t* external_twiddles,
    scalar_t*& internal_twiddles,
    scalar_t*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);
#endif

  template cudaError_t mixed_radix_ntt<scalar_t, scalar_t>(
    const scalar_t* d_input,
    scalar_t* d_output,
    scalar_t* external_twiddles,
    scalar_t* internal_twiddles,
    scalar_t* basic_twiddles,
    scalar_t* linear_twiddles,

    int ntt_size,
    int max_logn,
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    ntt::Ordering ordering,
    scalar_t* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream);

#if defined(EXT_FIELD)
  template cudaError_t mixed_radix_ntt<extension_t, scalar_t>(
    const extension_t* d_input,
    extension_t* d_output,
    scalar_t* external_twiddles,
    scalar_t* internal_twiddles,
    scalar_t* basic_twiddles,
    scalar_t* linear_twiddles,

    int ntt_size,
    int max_logn,
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    ntt::Ordering ordering,
    scalar_t* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream);
#endif

  // TODO: we may reintroduce mixed-radix ECNTT based on upcoming benching PR
  // #if defined(ECNTT)
  //   template cudaError_t mixed_radix_ntt<projective_t, scalar_t>(
  //     projective_t* d_input,
  //     projective_t* d_output,
  //     scalar_t* external_twiddles,
  //     scalar_t* internal_twiddles,
  //     scalar_t* basic_twiddles,
  //     int ntt_size,
  //     int max_logn,
  //     int batch_size,
  //     bool columns_batch,
  //     bool is_inverse,
  //     bool fast_tw,
  //     ntt::Ordering ordering,
  //     scalar_t* arbitrary_coset,
  //     int coset_gen_index,
  //     cudaStream_t cuda_stream);
  // #endif // ECNTT
} // namespace mxntt
