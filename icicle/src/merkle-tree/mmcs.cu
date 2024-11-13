#include "hash/hash.cuh"
#include "merkle-tree/merkle.cuh"
#include "matrix/matrix.cuh"
#include "vec_ops/vec_ops.cuh"

#include <algorithm>

using matrix::Matrix;

namespace merkle_tree {

  template <typename L, typename D>
  cudaError_t hash_leaves(
    const Matrix<L>* leaves,
    unsigned int number_of_inputs,
    uint64_t number_of_rows,
    D* digests,
    unsigned int digest_elements,
    const Hasher<L, D>& hasher,
    const device_context::DeviceContext& ctx)
  {
    HashConfig sponge_config = default_hash_config(ctx);
    sponge_config.are_inputs_on_device = true;
    sponge_config.are_outputs_on_device = true;
    sponge_config.is_async = true;

    uint64_t number_of_rows_padded = next_pow_of_two(number_of_rows);

    CHK_IF_RETURN(hasher.hash_2d(leaves, digests, number_of_inputs, digest_elements, number_of_rows, ctx));

    if (number_of_rows_padded - number_of_rows) {
      // Pad with default digests
      cudaMemsetAsync(
        (void*)(digests + number_of_rows), 0, (number_of_rows_padded - number_of_rows) * digest_elements * sizeof(D),
        ctx.stream);
    }

    return CHK_LAST();
  }

  template <typename L, typename D>
  struct SubtreeParams {
    unsigned int number_of_inputs; // Number of input matrices
    unsigned int arity;            // Arity of the tree
    unsigned int digest_elements;  // Number of output elements per hash
    size_t number_of_rows;         // Current number of input rows to operate on
    size_t number_of_rows_padded;  // next power of arity for number_of_rows
    size_t subtree_idx;            // The subtree id
    size_t number_of_subtrees;     // Total number of subtrees
    uint64_t subtree_height;       // Height of one subtree

    /// One segment corresponds to one layer of output digests
    size_t segment_size;                     // The size of current segment.
    size_t segment_offset;                   // An offset for the current segment
    unsigned int leaves_offset;              // An offset in the sorted list of input matrices
    unsigned int number_of_leaves_to_inject; // Number of leaves to inject in current level
    unsigned int keep_rows;                  // Number of rows to keep
    bool are_inputs_on_device;
    bool caps_mode;
    const Hasher<L, D>* hasher = nullptr;
    const Hasher<L, D>* compression = nullptr;
    const device_context::DeviceContext* ctx = nullptr;
  };

  template <typename L, typename D>
  cudaError_t slice_and_copy_leaves(
    const std::vector<Matrix<L>>& leaves, L* d_leaves, Matrix<L>* d_leaves_info, SubtreeParams<L, D>& params)
  {
    uint64_t target_height = params.number_of_rows_padded * params.number_of_subtrees;
    params.number_of_leaves_to_inject = 0;
    while (params.leaves_offset < params.number_of_inputs &&
           next_pow_of_two(leaves[params.leaves_offset].height) >= target_height) {
      if (next_pow_of_two(leaves[params.leaves_offset].height) == target_height) params.number_of_leaves_to_inject++;
      params.leaves_offset++;
    }

    if (params.number_of_leaves_to_inject) {
      size_t rows_offset = params.subtree_idx * params.number_of_rows_padded;
      size_t actual_layer_rows = leaves[params.leaves_offset - params.number_of_leaves_to_inject].height;
      params.number_of_rows = std::min(actual_layer_rows - rows_offset, params.number_of_rows_padded);

      Matrix<L>* leaves_info = static_cast<Matrix<L>*>(malloc(params.number_of_leaves_to_inject * sizeof(Matrix<L>)));
      L* d_leaves_ptr = d_leaves;
      for (auto i = 0; i < params.number_of_leaves_to_inject; i++) {
        Matrix<L> leaf = leaves[params.leaves_offset - params.number_of_leaves_to_inject + i];
        if (!params.are_inputs_on_device) {
          CHK_IF_RETURN(cudaMemcpyAsync(
            d_leaves_ptr, leaf.values + rows_offset * leaf.width, params.number_of_rows * leaf.width * sizeof(L),
            cudaMemcpyHostToDevice, params.ctx->stream));
        } else {
          d_leaves_ptr = leaf.values + rows_offset * leaf.width;
        }

        leaves_info[i] = {d_leaves_ptr, leaf.width, params.number_of_rows};
        d_leaves_ptr += params.number_of_rows * leaf.width;
      }
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_leaves_info, leaves_info, params.number_of_leaves_to_inject * sizeof(Matrix<L>), cudaMemcpyHostToDevice,
        params.ctx->stream));
      free(leaves_info);
    }

    return CHK_LAST();
  }

  /// Checks if the current row needs to be copied out to the resulting digests array
  /// Computes the needed offsets using segments model
  template <typename L, typename D>
  cudaError_t maybe_copy_digests(D* digests, L* big_tree_digests, SubtreeParams<L, D>& params)
  {
    if (!params.keep_rows || params.subtree_height < params.keep_rows + (int)params.caps_mode) {
      D* digests_with_offset = big_tree_digests + params.segment_offset +
                               params.subtree_idx * params.number_of_rows_padded * params.digest_elements;
      CHK_IF_RETURN(cudaMemcpyAsync(
        digests_with_offset, digests, params.number_of_rows_padded * params.digest_elements * sizeof(D),
        cudaMemcpyDeviceToHost, params.ctx->stream));
      params.segment_offset += params.segment_size;
    }
    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t fold_layer(
    const std::vector<Matrix<L>>& leaves,
    D* prev_layer,
    D* next_layer,
    L* aux_leaves_mem,
    Matrix<L>* d_leaves_info,
    SubtreeParams<L, D>& params)
  {
    CHK_IF_RETURN(slice_and_copy_leaves<L>(leaves, aux_leaves_mem, d_leaves_info, params));

    if (params.number_of_leaves_to_inject) {
      CHK_IF_RETURN(params.compression->compress_and_inject(
        d_leaves_info, params.number_of_leaves_to_inject, params.number_of_rows, prev_layer, next_layer,
        params.digest_elements, *params.ctx));
    } else {
      CHK_IF_RETURN(params.compression->run_hash_many_kernel(
        prev_layer, next_layer, params.number_of_rows_padded, params.compression->width, params.digest_elements,
        *params.ctx));
    }

    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t build_mmcs_subtree(
    const std::vector<Matrix<L>>& leaves,
    L* d_leaves,
    D* states,
    L* aux_leaves_mem,
    L* big_tree_digests,
    SubtreeParams<L, D>& params)
  {
    // Leaves info
    Matrix<L>* d_leaves_info;
    CHK_IF_RETURN(cudaMallocAsync(&d_leaves_info, params.number_of_inputs * sizeof(Matrix<L>), params.ctx->stream));

    CHK_IF_RETURN(slice_and_copy_leaves(leaves, d_leaves, d_leaves_info, params));

    // Reuse leaves memory
    D* digests = (D*)d_leaves;

    size_t size = sizeof(L) * leaves[0].height * leaves[0].width;
    L* buffer = (L*)malloc(size);
    cudaMemcpy(buffer, leaves[0].values, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      if (i % 4 == 0) { std::cout << std::endl << i / 32 << ": "; }
      printf("%.2X", buffer[i]);
    }

    CHK_IF_RETURN(hash_leaves(
      d_leaves_info, params.number_of_leaves_to_inject, params.number_of_rows, states, params.digest_elements,
      *params.hasher, *params.ctx));

    CHK_IF_RETURN(maybe_copy_digests(states, big_tree_digests, params));

    params.number_of_rows_padded /= params.arity;
    params.segment_size /= params.arity;
    params.subtree_height--;

    D* prev_layer = states;
    D* next_layer = digests;
    while (params.number_of_rows_padded > 0) {
      CHK_IF_RETURN(fold_layer(leaves, prev_layer, next_layer, aux_leaves_mem, d_leaves_info, params));
      CHK_IF_RETURN(maybe_copy_digests(next_layer, big_tree_digests, params));
      swap<D>(&prev_layer, &next_layer);
      params.segment_size /= params.arity;
      params.subtree_height--;
      params.number_of_rows_padded /= params.arity;
    }
    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t mmcs_commit(
    const Matrix<L>* inputs,
    const unsigned int number_of_inputs,
    D* digests,
    const Hasher<L, D>& hasher,
    const Hasher<L, D>& compression,
    const TreeBuilderConfig& tree_config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = tree_config.ctx.stream;

    if (number_of_inputs == 0) THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "No matrices provided");

    if (compression.preimage_max_length < tree_config.arity * tree_config.digest_elements)
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument,
        "Hash max preimage length does not match merkle tree arity multiplied by digest elements");

    std::vector<Matrix<L>> sorted_inputs(number_of_inputs);
    std::partial_sort_copy(
      inputs, inputs + number_of_inputs, sorted_inputs.begin(), sorted_inputs.end(),
      [](const Matrix<L>& left, const Matrix<L>& right) { return left.height > right.height; });

    // Check that the height of any two given matrices either rounds up
    // to the same next power of two or otherwise equal
    for (unsigned int i = 0; i < number_of_inputs - 1; i++) {
      unsigned int left = sorted_inputs[i].height;
      unsigned int right = sorted_inputs[i + 1].height;

      if (next_pow_of_two(left) == next_pow_of_two(right) && left != right)
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "Matrix heights that round up to the same power of two must be equal");
    }

    uint64_t max_height = sorted_inputs[0].height;

    // Calculate maximum additional memory needed for injected matrices
    uint64_t max_aux_total_elements = 0;
    uint64_t current_aux_total_elements = 0;
    uint64_t current_height = 0;
    uint64_t bottom_layer_leaves_elements = 0;
    if (!tree_config.are_inputs_on_device) {
      for (auto it = sorted_inputs.begin(); it < sorted_inputs.end(); it++) {
        if (it->height == max_height) {
          bottom_layer_leaves_elements += it->height * it->width;
          continue;
        }

        if (it->height != current_height) {
          current_height = it->height;
          current_aux_total_elements = 0;
        }

        current_aux_total_elements += it->width * it->height;
        if (current_aux_total_elements > max_aux_total_elements) {
          max_aux_total_elements = current_aux_total_elements;
        }
      }
    }

    uint64_t number_of_bottom_layer_rows = next_pow_of_two(max_height);
    size_t leaves_info_memory = number_of_inputs * sizeof(Matrix<L>);

    unsigned int tree_height = get_height(number_of_bottom_layer_rows);

    // This will determine how much splitting do we need to do
    // `number_of_streams` subtrees should fit in the device
    // This means each subtree should fit in `STREAM_CHUNK_SIZE` memory
    uint64_t number_of_subtrees = 1;
    uint64_t subtree_height = tree_height;
    uint64_t subtree_bottom_layer_rows = number_of_bottom_layer_rows;
    uint64_t subtree_states_size = subtree_bottom_layer_rows * hasher.width;
    uint64_t subtree_digests_size = subtree_bottom_layer_rows * tree_config.digest_elements;
    uint64_t subtree_leaves_elements = 0;
    for (int i = 0; i < number_of_inputs && sorted_inputs[i].height == max_height; i++) {
      subtree_leaves_elements += sorted_inputs[i].width * sorted_inputs[i].height;
    }
    uint64_t subtree_aux_elements = max_aux_total_elements;

    size_t subtree_leaves_memory = std::max(subtree_leaves_elements * sizeof(L), subtree_digests_size * sizeof(D));
    size_t subtree_memory_required =
      sizeof(D) * subtree_states_size + subtree_leaves_memory + subtree_aux_elements * sizeof(L) + leaves_info_memory;
    while (subtree_memory_required > STREAM_CHUNK_SIZE) {
      number_of_subtrees *= tree_config.arity;
      subtree_height--;
      subtree_bottom_layer_rows /= tree_config.arity;
      subtree_states_size /= tree_config.arity;
      subtree_digests_size /= tree_config.arity;
      subtree_leaves_elements /= tree_config.arity;
      subtree_aux_elements /= tree_config.arity;
      subtree_leaves_memory = std::max(subtree_leaves_elements * sizeof(L), subtree_digests_size * sizeof(D));
      subtree_memory_required =
        sizeof(D) * subtree_states_size + subtree_leaves_memory + subtree_aux_elements * sizeof(L) + leaves_info_memory;
    }
    unsigned int cap_height = tree_height - subtree_height;
    size_t caps_len = pow(tree_config.arity, cap_height) * tree_config.digest_elements;

    size_t available_memory, _total_memory;
    CHK_IF_RETURN(cudaMemGetInfo(&available_memory, &_total_memory));
    if (available_memory < (GIGA / 8 + STREAM_CHUNK_SIZE)) {
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument,
        "Not enough GPU memory to build a tree. At least 1.125 GB of GPU memory required");
    }
    available_memory -= GIGA / 8; // Leave 128 MB just in case

    // We can effectively parallelize memory copy with streams
    // as long as they don't operate on more than `STREAM_CHUNK_SIZE` bytes
    const size_t number_of_streams = std::min((uint64_t)(available_memory / STREAM_CHUNK_SIZE), number_of_subtrees);
    std::vector<cudaStream_t> streams(number_of_streams);
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamCreate(&streams[i]));
    }

    // If keep_rows is smaller then the remaining top-tree height
    // we need to allocate additional memory to store the roots
    // of subtrees, in order to proceed from there
    bool caps_mode = tree_config.keep_rows && tree_config.keep_rows <= cap_height;
    D* caps;
    if (caps_mode) { caps = static_cast<D*>(malloc(caps_len * sizeof(D))); }

#ifdef MERKLE_DEBUG
    std::cout << "MMCS DEBUG" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Available memory = " << available_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Number of streams = " << number_of_streams << std::endl;
    std::cout << "Number of subtrees = " << number_of_subtrees << std::endl;
    std::cout << "Height of a subtree = " << subtree_height << std::endl;
    std::cout << "Cutoff height = " << tree_height - subtree_height << std::endl;
    std::cout << "Number of leaves in a subtree = " << subtree_bottom_layer_rows << std::endl;
    std::cout << "State of a subtree = " << subtree_states_size << std::endl;
    std::cout << "Digest elements for a subtree = " << subtree_digests_size << std::endl;
    std::cout << "Size of 1 subtree states = " << subtree_states_size * sizeof(D) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Size of 1 subtree digests = " << subtree_digests_size * sizeof(D) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Cap height = " << cap_height << std::endl;
    std::cout << "Enabling caps mode? " << caps_mode << std::endl;

    std::cout << "Allocating " << subtree_states_size * number_of_streams << " elements for states" << std::endl;
    std::cout << "Allocating " << subtree_leaves_memory * number_of_streams << " bytes for leaves" << std::endl;
    std::cout << "Allocating " << subtree_aux_elements * number_of_streams << " elements for aux leaves" << std::endl;
    std::cout << std::endl;
#endif

    // Allocate memory for the states, injected leaves (aux) and digests
    // These are shared by streams in a pool
    D* states_ptr;
    L *aux_ptr, *leaves_ptr;
    CHK_IF_RETURN(cudaMallocAsync(&states_ptr, subtree_states_size * number_of_streams * sizeof(D), stream));
    CHK_IF_RETURN(cudaMemsetAsync(states_ptr, 0, subtree_states_size * number_of_streams * sizeof(D), stream));
    CHK_IF_RETURN(cudaMallocAsync(&leaves_ptr, subtree_leaves_memory * number_of_streams, stream));
    CHK_IF_RETURN(cudaMallocAsync(&aux_ptr, subtree_aux_elements * number_of_streams * sizeof(L), stream));
    // Wait for these allocations to finish
    CHK_IF_RETURN(cudaStreamSynchronize(stream));

    // Build subtrees in parallel. This for loop invokes kernels that can run in a pool of size `number_of_streams`
    for (size_t subtree_idx = 0; subtree_idx < number_of_subtrees; subtree_idx++) {
      size_t stream_idx = subtree_idx % number_of_streams;
      cudaStream_t subtree_stream = streams[stream_idx];

      D* subtree_state = states_ptr + stream_idx * subtree_states_size;
      L* subtree_leaves = (L*)((unsigned char*)leaves_ptr + stream_idx * subtree_leaves_memory);
      L* subtree_aux = aux_ptr + stream_idx * subtree_aux_elements;

      unsigned int subtree_keep_rows = 0;
      if (tree_config.keep_rows) {
        int diff = tree_config.keep_rows - cap_height;
        subtree_keep_rows = std::max(1, diff);
      }
      device_context::DeviceContext subtree_context{subtree_stream, tree_config.ctx.device_id, tree_config.ctx.mempool};

      SubtreeParams<L, D> params = {};

      params.number_of_inputs = number_of_inputs;
      params.arity = tree_config.arity;
      params.digest_elements = tree_config.digest_elements;
      params.number_of_rows = subtree_bottom_layer_rows;
      params.number_of_rows_padded = subtree_bottom_layer_rows;

      params.subtree_idx = subtree_idx;
      params.subtree_height = subtree_height;
      params.number_of_subtrees = number_of_subtrees;

      params.segment_size = number_of_bottom_layer_rows * tree_config.digest_elements;
      params.keep_rows = subtree_keep_rows;
      params.are_inputs_on_device = tree_config.are_inputs_on_device;
      params.hasher = &hasher;
      params.compression = &compression;
      params.ctx = &subtree_context;

      cudaError_t subtree_result = build_mmcs_subtree<L, D>(
        sorted_inputs,
        subtree_leaves,             // d_leaves
        subtree_state,              // states
        subtree_aux,                // aux_leaves_mem
        caps_mode ? caps : digests, // big_tree_digests
        params                      // params
      );
      CHK_IF_RETURN(subtree_result);
    }

    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
    }

    // Finish the top-level tree if any
    if (cap_height > 0) {
      D* digests_ptr = (D*)leaves_ptr;
      size_t start_segment_size = caps_len / tree_config.arity;
      size_t start_segment_offset = 0;
      if (!caps_mode) { // Calculate offset
        size_t keep_rows = tree_config.keep_rows ? tree_config.keep_rows : tree_height + 1;
        size_t layer_size = pow(tree_config.arity, keep_rows - 1) * tree_config.digest_elements;
        for (int i = 0; i < keep_rows - cap_height; i++) {
          start_segment_offset += layer_size;
          layer_size /= tree_config.arity;
        }
      }

      CHK_IF_RETURN(cudaMemcpyAsync(
        states_ptr, caps_mode ? caps : (digests + start_segment_offset - caps_len), caps_len * sizeof(D),
        (caps_mode || !tree_config.are_outputs_on_device) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice, stream));

      uint64_t number_of_states = caps_len / tree_config.arity / tree_config.digest_elements;
      Matrix<L>* d_leaves_info;
      CHK_IF_RETURN(cudaMallocAsync(&d_leaves_info, number_of_inputs * sizeof(Matrix<L>), tree_config.ctx.stream));

      SubtreeParams<L, D> top_params = {};

      top_params.number_of_inputs = number_of_inputs;
      top_params.arity = tree_config.arity;
      top_params.digest_elements = tree_config.digest_elements;
      top_params.number_of_rows = number_of_states;
      top_params.number_of_rows_padded = number_of_states;

      top_params.subtree_height = cap_height;
      top_params.number_of_subtrees = 1;

      top_params.segment_offset = start_segment_offset;
      top_params.segment_size = start_segment_size;
      top_params.keep_rows = tree_config.keep_rows;
      top_params.are_inputs_on_device = tree_config.are_inputs_on_device;
      top_params.caps_mode = caps_mode;
      top_params.hasher = &hasher;
      top_params.compression = &compression;
      top_params.ctx = &tree_config.ctx;

      D* prev_layer = states_ptr;
      D* next_layer = digests_ptr;
      while (top_params.number_of_rows_padded > 0) {
        CHK_IF_RETURN(fold_layer(sorted_inputs, prev_layer, next_layer, aux_ptr, d_leaves_info, top_params));
        CHK_IF_RETURN(maybe_copy_digests(next_layer, digests, top_params));
        swap<D>(&prev_layer, &next_layer);
        top_params.segment_size /= top_params.arity;
        top_params.subtree_height--;
        top_params.number_of_rows_padded /= top_params.arity;
      }
    }

    if (caps_mode) { free(caps); }
    CHK_IF_RETURN(cudaFreeAsync(states_ptr, stream));
    CHK_IF_RETURN(cudaFreeAsync(leaves_ptr, stream));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamDestroy(streams[i]));
    }
    if (!tree_config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    return CHK_LAST();
  }

} // namespace merkle_tree
