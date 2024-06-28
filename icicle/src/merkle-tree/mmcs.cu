#include "hash/hash.cuh"
#include "merkle-tree/merkle.cuh"
#include "matrix/matrix.cuh"

#include <algorithm>

using matrix::Matrix;

namespace merkle_tree {

  template <typename L, typename D>
  cudaError_t hash_leaves(
    const Matrix<L>* leaves,
    unsigned int number_of_inputs,
    uint64_t number_of_rows,
    D* states,
    D* digests,
    unsigned int digest_elements,
    const SpongeHasher<L, D>& hasher,
    const device_context::DeviceContext& ctx)
  {
    SpongeConfig sponge_config = default_sponge_config(ctx);
    sponge_config.are_inputs_on_device = true;
    sponge_config.are_outputs_on_device = true;
    sponge_config.is_async = true;

    uint64_t number_of_rows_padded = next_pow_of_two(number_of_rows);

    std::cout << "Absorbing 2d" << std::endl;
    CHK_IF_RETURN(hasher.absorb_2d(leaves, states, number_of_inputs, number_of_rows, ctx));
    std::cout << "Squeezing" << std::endl;
    CHK_IF_RETURN(hasher.squeeze_many(states, digests, number_of_rows, digest_elements, sponge_config));

    if (number_of_rows_padded - number_of_rows) {
      // Pad with default digests
      cudaMemsetAsync(
        (void*)digests, 0, (number_of_rows_padded - number_of_rows) * digest_elements * sizeof(D), ctx.stream);
    }

    return CHK_LAST();
  }

  template <typename L, typename D>
  struct SubtreeParams {
    unsigned int number_of_inputs;
    unsigned int arity;
    unsigned int digest_elements;
    size_t number_of_rows;
    size_t number_of_rows_padded;
    size_t subtree_idx;
    size_t number_of_subtrees;
    uint64_t subtree_height;
    size_t segment_size;
    size_t segment_offset;
    unsigned int leaves_offset;
    unsigned int number_of_leaves_to_inject;
    unsigned int keep_rows;
    const SpongeHasher<L, D>* hasher = nullptr;
    const SpongeHasher<L, D>* compression = nullptr;
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
      std::cout << "Leaves size: " << leaves.size() << "; Leaves ptr: " << &leaves
                << "; Actual layer rows = " << actual_layer_rows << "; rows_offset = " << rows_offset
                << "; number_of_rows_padded = " << params.number_of_rows_padded << std::endl;
      params.number_of_rows = std::min(actual_layer_rows - rows_offset, params.number_of_rows_padded);

      Matrix<L>* leaves_info = static_cast<Matrix<L>*>(malloc(params.number_of_leaves_to_inject * sizeof(Matrix<L>)));
      L* d_leaves_ptr = d_leaves;
      for (auto i = 0; i < params.number_of_leaves_to_inject; i++) {
        Matrix<L> leaf = leaves[params.leaves_offset - params.number_of_leaves_to_inject + i];
        std::cout << "leaf pointer: " << leaf.values << "; leaf.width = " << leaf.width
                  << "; leaf.height = " << leaf.height << std::endl;
        std::cout << "Matrix " << params.leaves_offset - params.number_of_leaves_to_inject + i
                  << "; number of rows: " << params.number_of_rows << "; Copying " << params.number_of_rows * leaf.width
                  << " elements from " << leaf.values + rows_offset * leaf.width << " to " << d_leaves_ptr << std::endl;
        CHK_IF_RETURN(cudaMemcpyAsync(
          d_leaves_ptr, leaf.values + rows_offset * leaf.width, params.number_of_rows * leaf.width * sizeof(L),
          cudaMemcpyHostToDevice, params.ctx->stream));
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

  template <typename L, typename D>
  cudaError_t maybe_copy_digests(D* digests, L* big_tree_digests, SubtreeParams<L, D>& params)
  {
    if (!params.keep_rows || params.subtree_height < params.keep_rows) {
      D* digests_with_offset = big_tree_digests + params.segment_offset +
                               params.subtree_idx * params.number_of_rows_padded * params.digest_elements;
      std::cout << "Copying " << params.number_of_rows_padded * params.digest_elements << " digests to host "
                << digests_with_offset << "; offset = "
                << params.segment_offset + params.subtree_idx * params.number_of_rows_padded * params.digest_elements
                << std::endl;
      CHK_IF_RETURN(cudaMemcpyAsync(
        digests_with_offset, digests, params.number_of_rows_padded * params.digest_elements * sizeof(D),
        cudaMemcpyDeviceToHost, params.ctx->stream));
      params.segment_offset += params.segment_size;
    }
    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t fold_layers(
    const std::vector<Matrix<L>>& leaves,
    D* states,
    D* digests,
    L* aux_leaves_mem,
    Matrix<L>* d_leaves_info,
    L* big_tree_digests,
    SubtreeParams<L, D>& params)
  {
    while (params.number_of_rows_padded > 0) {
      CHK_IF_RETURN(slice_and_copy_leaves<L>(leaves, aux_leaves_mem, d_leaves_info, params));

      if (params.number_of_leaves_to_inject) {
        std::cout << "Injecting " << params.number_of_rows << std::endl;
        CHK_IF_RETURN(params.compression->compress_and_inject(
          d_leaves_info, params.number_of_leaves_to_inject, params.number_of_rows, states, digests,
          params.digest_elements, *params.ctx));
        std::cout << "Injected" << std::endl;
      } else {
        std::cout << "Compressing" << std::endl;
        CHK_IF_RETURN(params.compression->compress_many(
          states, digests, params.number_of_rows_padded, params.digest_elements, *params.ctx));
        std::cout << "Compressed" << std::endl;
      }

      CHK_IF_RETURN(maybe_copy_digests(digests, big_tree_digests, params));
      swap<D>(&digests, &states);
      params.segment_size /= params.arity;
      params.subtree_height--;
      params.number_of_rows_padded /= params.arity;
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

    std::cout << "Hashing bottom layer leaves " << params.number_of_rows << std::endl;
    CHK_IF_RETURN(hash_leaves(
      d_leaves_info, params.number_of_leaves_to_inject, params.number_of_rows, states, digests, params.digest_elements,
      *params.hasher, *params.ctx));
    std::cout << "Hashed bottom layer leaves" << std::endl;

    CHK_IF_RETURN(maybe_copy_digests(digests, big_tree_digests, params));

    params.number_of_rows_padded /= params.arity;
    params.segment_size /= params.arity;
    params.subtree_height--;

    swap<D>(&digests, &states);
    return fold_layers<L, D>(leaves, states, digests, aux_leaves_mem, d_leaves_info, big_tree_digests, params);
  }

  template <typename L, typename D>
  cudaError_t mmcs_commit(
    const Matrix<L>* inputs,
    const unsigned int number_of_inputs,
    D* digests,
    const SpongeHasher<L, D>& hasher,
    const SpongeHasher<L, D>& compression,
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

    // for (int i = 0; i < number_of_inputs; i++) {
    //   std::cout << "Orig Matrix " << i << ": " << inputs[i].width << ", " << inputs[i].height << std::endl;
    //   for (int j = 0; j < inputs[i].height; j++) {
    //     for (int k = 0; k < inputs[i].width; k++) {
    //       std::cout << inputs[i].values[j * inputs[i].width + k] << " ";
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << "Matrix " << i << ": " << sorted_inputs[i].width << ", " << sorted_inputs[i].height << std::endl;
    // }

    uint64_t max_height = sorted_inputs[0].height;

    // Calculate maximum additional memory needed for injected matrices
    uint64_t max_aux_total_elements = 0;
    uint64_t current_aux_total_elements = 0;
    uint64_t current_height = 0;
    uint64_t bottom_layer_leaves_elements = 0;
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
      if (current_aux_total_elements > max_aux_total_elements) { max_aux_total_elements = current_aux_total_elements; }
    }

    std::cout << "Max aux total elements " << max_aux_total_elements << std::endl;

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
    available_memory -= GIGA / 8; // Leave 128 MB just in case

    // We can effectively parallelize memory copy with streams
    // as long as they don't operate on more than `STREAM_CHUNK_SIZE` bytes
    const size_t number_of_streams = std::min((uint64_t)(available_memory / STREAM_CHUNK_SIZE), number_of_subtrees);
    cudaStream_t* streams = static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * number_of_streams));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamCreate(&streams[i]));
    }

    bool caps_mode = tree_config.keep_rows && tree_config.keep_rows <= cap_height;
    D* caps;
    if (caps_mode) { caps = static_cast<D*>(malloc(caps_len * sizeof(D))); }

    // #ifdef MERKLE_DEBUG
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
    std::cout << std::endl;
    // #endif

    // Allocate memory for the states, injected leaves and digests
    // These are shared by streams in a pool
    D* states_ptr;
    L *aux_ptr, *leaves_ptr;
    CHK_IF_RETURN(cudaMallocAsync(&states_ptr, subtree_states_size * number_of_streams * sizeof(D), stream));
    CHK_IF_RETURN(cudaMemsetAsync(states_ptr, 0, subtree_states_size * number_of_streams * sizeof(D), stream));
    CHK_IF_RETURN(cudaMallocAsync(&leaves_ptr, subtree_leaves_memory * number_of_streams, stream));
    CHK_IF_RETURN(cudaMallocAsync(&aux_ptr, subtree_aux_elements * number_of_streams * sizeof(L), stream));
    // Wait for these allocations to finish
    CHK_IF_RETURN(cudaStreamSynchronize(stream));

    std::cout << "Allocated " << subtree_states_size * number_of_streams << " elements for states" << std::endl;
    std::cout << "Allocated " << subtree_leaves_memory * number_of_streams << " bytes for leaves" << std::endl;
    std::cout << "Allocated " << subtree_aux_elements * number_of_streams << " elements for aux leaves" << std::endl;

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
      params.hasher = &hasher;
      params.compression = &compression;
      params.ctx = &subtree_context;

      std::cout << "SubTree " << params.subtree_idx << "; bottom_layer_rows: " << params.number_of_rows << std::endl;

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
      top_params.hasher = &hasher;
      top_params.compression = &compression;
      top_params.ctx = &tree_config.ctx;

      cudaError_t fold_error =
        fold_layers<L, D>(sorted_inputs, states_ptr, digests_ptr, aux_ptr, d_leaves_info, digests, top_params);
      CHK_IF_RETURN(fold_error);
    }

    if (caps_mode) { free(caps); }
    CHK_IF_RETURN(cudaFreeAsync(states_ptr, stream));
    CHK_IF_RETURN(cudaFreeAsync(leaves_ptr, stream));
    if (!tree_config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
      CHK_IF_RETURN(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    return CHK_LAST();
  }

} // namespace merkle_tree
