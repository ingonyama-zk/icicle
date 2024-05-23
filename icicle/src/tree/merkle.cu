#include "hash/hash.cuh"
#include "merkle-tree/merkle.cuh"

namespace merkle_tree {
  /// Flattens the tree digests and sum them up to get
  /// the memory needed to contain all the digests
  template <typename S>
  size_t get_digests_len(uint32_t height, uint32_t arity)
  {
    size_t digests_len = 0;
    size_t row_length = 1;
    for (int i = 0; i < height; i++) {
      digests_len += row_length;
      row_length *= arity;
    }

    return digests_len;
  }

  /// Constructs merkle subtree without parallelization
  /// The digests are aligned sequentially per row
  /// Example:
  ///
  /// Big tree:
  ///
  ///        1      <- Root
  ///       / \     <- Arity = 2
  ///      2   3    <- Digests
  ///     / \ / \   <- Height = 2 (as the number of edges)
  ///    4  5 6  7  <- height^arity leaves
  ///    |  | |  |  <- Sponge hash 1 to 1
  ///    a  b c  d  <- Input vector 1x4
  ///
  /// Subtree 1    Subtree 2
  ///    2            3
  ///   / \          / \
  ///  4   5        6   7
  ///
  /// Digests array for subtree 1:
  /// [4 5 . . 2 . .]
  /// |   |    |
  /// -----    V
  ///   |    Segment (offset = 4, subtree_idx = 0)
  ///   v
  /// Segment (offset = 0, subtree_idx = 0)
  ///
  /// Digests array for subtree 2:
  /// [. . 6 7 . 3 .]
  ///     |   |
  ///     -----
  ///       |
  ///       v
  ///    Segment (offset = 0, subtree_idx = 1)
  ///
  /// Total digests array:
  /// [4 5 6 7 2 3 .]
  template <typename L, typename D>
  cudaError_t build_merkle_subtree(
    L* leaves,
    D* states,
    D* digests,
    size_t subtree_idx,
    size_t subtree_height,
    L* big_tree_digests,
    size_t start_segment_size,
    size_t start_segment_offset,
    int keep_rows,
    const SpongeHasher<L, D>& sponge,
    const CompressionHasher<D>& compression,
    const SpongeConfig& sponge_config,
    cudaStream_t& stream)
  {
    unsigned int arity = compression.preimage_max_length;

    size_t leaves_size = pow(arity, subtree_height);

    sponge.absorb_many(leaves, states, leaves_size, sponge_config);
    sponge.squeeze_many(states, digests, leaves_size, sponge_config);

    uint32_t number_of_states = leaves_size;
    size_t segment_size = start_segment_size;
    size_t segment_offset = start_segment_offset;

    device_context::DeviceContext subtree_ctx{
      stream,
      sponge_config.ctx.device_id,
      sponge_config.ctx.mempool,
    };

    while (number_of_states > 0) {
      cudaError_t compression_error =
        compression.compress_many(digests, digests, number_of_states, sponge_config.offset, states, subtree_ctx, true);
      CHK_IF_RETURN(compression_error);

      if (!keep_rows || subtree_height <= keep_rows) {
        S* digests_with_offset = big_tree_digests + segment_offset + subtree_idx * number_of_states;
        CHK_IF_RETURN(
          cudaMemcpyAsync(digests_with_offset, digests, number_of_states * sizeof(D), cudaMemcpyDeviceToHost, stream));
        segment_offset += segment_size;
      }

      segment_size /= arity;
      subtree_height--;
      number_of_states /= arity;
      // config.aligned = true;
    }

    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t build_merkle_tree(
    const L* leaves,
    D* digests,
    uint32_t height,
    const SpongeHasher<L, D>& sponge,
    const CompressionHasher<D>& compression,
    const SpongeConfig& sponge_config,
    const TreeBuilderConfig& tree_config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;

    if (sponge_config.input_block_len >= sponge.preimage_max_length)
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument,
        "Sponge construction at the bottom of the tree doesn't support inputs bigger than the size of the state");
    if (sponge_config.output_len != 1)
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument,
        "Sponge construction at the bottom of the tree should have an output len of 1 element");
    if (compression.preimage_max_length != tree_config.arity)
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Hash max preimage length does not match merkle tree arity");

    uint32_t number_of_leaves = pow(tree_config.arity, height);
    uint32_t total_number_of_leaves = number_of_leaves * sponge_config.input_block_len;

    // This will determine how much splitting do we need to do
    // `number_of_streams` subtrees should fit in the device
    // This means each subtree should fit in `STREAM_CHUNK_SIZE` memory
    uint32_t number_of_subtrees = 1;
    uint32_t subtree_height = height;
    uint32_t subtree_leaves_size = number_of_leaves;
    uint32_t subtree_states_size = subtree_leaves_size * sponge.width;
    uint32_t subtree_digests_size = subtree_leaves_size;
    size_t subtree_memory_required = sizeof(D) * (subtree_states_size + subtree_digests_size);
    while (subtree_memory_required > STREAM_CHUNK_SIZE) {
      number_of_subtrees *= tree_config.arity;
      subtree_height--;
      subtree_leaves_size /= tree_config.arity;
      subtree_states_size /= tree_config.arity;
      subtree_digests_size /= tree_config.arity;
      subtree_memory_required = sizeof(S) * (subtree_states_size + subtree_digests_size);
    }
    int cap_height = height - subtree_height;
    size_t caps_len = pow(tree_config.arity, cap_height);

    size_t available_memory, _total_memory;
    CHK_IF_RETURN(cudaMemGetInfo(&available_memory, &_total_memory));
    available_memory -= GIGA / 8; // Leave 128 MB just in case

    // We can effectively parallelize memory copy with streams
    // as long as they don't operate on more than `STREAM_CHUNK_SIZE` bytes
    const size_t number_of_streams = std::min((uint32_t)(available_memory / STREAM_CHUNK_SIZE), number_of_subtrees);
    cudaStream_t* streams = static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * number_of_streams));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamCreate(&streams[i]));
    }

    // Allocate memory for the leaves and digests
    // These are shared by streams in a pool
    D *states_ptr, *digests_ptr;
    CHK_IF_RETURN(cudaMallocAsync(&states_ptr, subtree_states_size * number_of_streams * sizeof(D), stream))
    CHK_IF_RETURN(cudaMallocAsync(&digests_ptr, subtree_digests_size * number_of_streams * sizeof(D), stream))
    // Wait for these allocations to finish
    CHK_IF_RETURN(cudaStreamSynchronize(stream));

    bool caps_mode = tree_config.keep_rows && tree_config.keep_rows < cap_height;
    D* caps;
    if (caps_mode) { caps = static_cast<D*>(malloc(caps_len * sizeof(D))); }

#if !defined(__CUDA_ARCH__) && defined(MERKLE_DEBUG)
    std::cout << "Available memory = " << available_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Number of streams = " << number_of_streams << std::endl;
    std::cout << "Number of subtrees = " << number_of_subtrees << std::endl;
    std::cout << "Height of a subtree = " << subtree_height << std::endl;
    std::cout << "Cutoff height = " << height - subtree_height << std::endl;
    std::cout << "Number of leaves in a subtree = " << subtree_leaves_size << std::endl;
    std::cout << "State of a subtree = " << subtree_states_size << std::endl;
    std::cout << "Digest elements for a subtree = " << get_digests_len<S>(subtree_height, tree_config.arity)
              << std::endl;
    std::cout << "Size of 1 subtree states = " << subtree_states_size * sizeof(S) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Size of 1 subtree digests = " << subtree_digests_size * sizeof(S) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Cap height" << cap_height << std::endl;
    std::cout << "Enabling caps mode? " << caps_mode << std::endl;
#endif

    // Build subtrees in parallel. This for loop invokes kernels that can run in a pool of size `number_of_streams`
    for (size_t subtree_idx = 0; subtree_idx < number_of_subtrees; subtree_idx++) {
      size_t stream_idx = subtree_idx % number_of_streams;
      cudaStream_t subtree_stream = streams[stream_idx];

      const L* subtree_leaves = leaves + subtree_idx * subtree_leaves_size * sponge_config.input_block_len;
      D* subtree_state = states_ptr + stream_idx * subtree_states_size;
      D* subtree_digests = digests_ptr + stream_idx * subtree_digests_size;

      int subtree_keep_rows = 0;
      if (config.keep_rows) {
        int diff = config.keep_rows - cap_height;
        subtree_keep_rows = diff <= 0 ? 1 : diff;
      }
      cudaError_t subtree_result = build_merkle_subtree<L, D>(
        subtree_leaves,             // leaves
        subtree_state,              // state
        subtree_digests,            // digests
        subtree_idx,                // subtree_idx
        subtree_height,             // subtree_height
        caps_mode ? caps : digests, // big_tree_digests
        number_of_leaves,           // start_segment_size
        0,                          // start_segment_offset
        subtree_keep_rows,          // keep_rows
        sponge,                     // hash
        compression,                // hash
        sponge_config,              // hash
        subtree_stream              // stream
      );
      CHK_IF_RETURN(subtree_result);
    }

    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
    }

    // Finish the top-level tree if any
    // if (cap_height > 0) {
    //   size_t start_segment_size = caps_len / tree_config.arity;
    //   size_t start_segment_offset = 0;
    //   if (!caps_mode) {
    //     size_t layer_size = pow(tree_config.arity, config.keep_rows - 1);
    //     for (int i = 0; i < config.keep_rows - cap_height + 1; i++) {
    //       start_segment_offset += layer_size;
    //       layer_size /= tree_config.arity;
    //     }
    //   }
    //   CHK_IF_RETURN(cudaMemcpy2DAsync(
    //     states_ptr, T * sizeof(S), caps_mode ? caps : (digests + start_segment_offset - caps_len), tree_config.arity
    //     * sizeof(S), tree_config.arity * sizeof(S), caps_len / tree_config.arity,                 // Size of the
    //     source cudaMemcpyHostToDevice, stream)); // Direction and stream

    //   cudaError_t top_tree_result = build_merkle_subtree<S, T>(
    //     states_ptr,           // state
    //     digests_ptr,          // digests
    //     0,                    // subtree_idx
    //     cap_height,           // subtree_height
    //     digests,              // big_tree_digests
    //     start_segment_size,   // start_segment_size
    //     start_segment_offset, // start_segment_offset
    //     config.keep_rows,     // keep_rows
    //     poseidon,             // hash
    //     stream                // stream
    //   );
    //   CHK_IF_RETURN(top_tree_result);
    //   if (caps_mode) { free(caps); }
    // }

    CHK_IF_RETURN(cudaFreeAsync(states_ptr, stream));
    CHK_IF_RETURN(cudaFreeAsync(digests_ptr, stream));
    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
      CHK_IF_RETURN(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    return CHK_LAST();
  }

} // namespace merkle_tree