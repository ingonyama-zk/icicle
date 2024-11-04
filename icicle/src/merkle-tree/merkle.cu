#include "hash/hash.cuh"
#include "merkle-tree/merkle.cuh"

namespace merkle_tree {
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
  ///    |  | |  |  <- Bottom layer hash 1 to 1
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
  ///
  /// Example for custom config:
  ///
  /// arity = 2
  /// input_block_len = 2
  /// digest_elements = 2
  /// bottom_layer hash width = 4
  /// compression width = 4
  /// height = 2
  ///
  ///                    [a, b]    <- Root of the tree
  ///                     |  |
  ///                    [a, b, c, d]
  ///                     /  \  /  \ 
  ///                    [i, j, m, n]
  ///           ┌──┬──────┴──┴──┴──┴──────┬──┐
  ///           |  |                      |  |
  ///          [i, j, k, l]              [m, n, o, p]       <- compression states
  ///           /  \  /  \                /  \  /  \        <- Running permutation
  ///          [1, 2, 5, 6]              [9, 1, 4, 5]       <- compression states
  ///    ┌──┬───┴──┴──┼──┤         ┌──┬───┴──┴──┼──┤
  ///    |  |         |  |         |  |         |  |        <- digest_element * height^arity leaves
  ///   [1, 2, 3, 4] [5, 6, 7, 8] [9, 1, 2, 3] [4, 5, 6, 7] <- Permuted states
  ///    /  \  /  \   /  \  /  \   /  \  /  \   /  \  /  \  <- Running permutation
  ///   [a, b, 0, 0] [c, d, 0, 0] [e, f, 0, 0] [g, h, 0, 0] <- States of the bottom layer hash
  ///    |  |         |  |         |  |         |  |        <- Bottom layer hash 2 to 2
  ///    a  b         c  d         e  f         g  h        <- Input vector 2x4
  ///
  /// Input matrix:
  ///   ┌     ┐
  ///   | a b |
  ///   | c d |
  ///   | e f |
  ///   | g h |
  ///   └     ┘

  template <typename L, typename D>
  cudaError_t build_merkle_subtree(
    const L* leaves,
    L* d_leaves,
    D* digests,
    size_t subtree_idx,
    size_t subtree_height,
    D* big_tree_digests,
    size_t start_segment_size,
    size_t start_segment_offset,
    uint64_t keep_rows,
    uint64_t input_block_len,
    const Hasher<L, D>& bottom_layer,
    const Hasher<L, D>& compression,
    const TreeBuilderConfig& tree_config,
    device_context::DeviceContext& ctx)
  {
    uint64_t arity = tree_config.arity;

    HashConfig hash_config = default_hash_config(ctx);
    hash_config.are_inputs_on_device = true;
    hash_config.are_outputs_on_device = true;
    hash_config.is_async = true;

    size_t bottom_layer_states = pow(arity, subtree_height);

    if (!tree_config.are_inputs_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_leaves, leaves, bottom_layer_states * input_block_len * sizeof(L), cudaMemcpyHostToDevice, ctx.stream));
    }

    bottom_layer.hash_many(
      tree_config.are_inputs_on_device ? leaves : d_leaves, digests, bottom_layer_states, input_block_len,
      tree_config.digest_elements, hash_config);

    uint64_t number_of_states = bottom_layer_states / arity;
    size_t segment_size = start_segment_size;
    size_t segment_offset = start_segment_offset;

    if (!keep_rows || subtree_height < keep_rows) {
      D* digests_with_offset = big_tree_digests + segment_offset + subtree_idx * bottom_layer_states;
      CHK_IF_RETURN(cudaMemcpyAsync(
        digests_with_offset, digests, bottom_layer_states * tree_config.digest_elements * sizeof(D),
        cudaMemcpyDeviceToHost, ctx.stream));
      segment_offset += segment_size;
    }
    segment_size /= arity;
    subtree_height--;

    D* prev_layer = digests;
    D* next_layer = (D*)d_leaves;

    while (number_of_states > 0) {
      CHK_IF_RETURN(compression.run_hash_many_kernel(
        (L*)prev_layer, next_layer, number_of_states,
        tree_config.digest_elements * tree_config.arity * (sizeof(D) / sizeof(L)), tree_config.digest_elements,
        hash_config.ctx));

      if (!keep_rows || subtree_height < keep_rows) {
        D* digests_with_offset =
          big_tree_digests + segment_offset + subtree_idx * number_of_states * tree_config.digest_elements;
        CHK_IF_RETURN(cudaMemcpyAsync(
          digests_with_offset, next_layer, number_of_states * tree_config.digest_elements * sizeof(D),
          cudaMemcpyDeviceToHost, ctx.stream));
        segment_offset += segment_size;
      }
      swap<D>(&prev_layer, &next_layer);
      segment_size /= arity;
      subtree_height--;
      number_of_states /= arity;
    }

    return CHK_LAST();
  }

  template <typename L, typename D>
  cudaError_t build_merkle_tree(
    const L* leaves,
    D* digests,
    unsigned int height,
    unsigned int input_block_len,
    const Hasher<L, D>& compression,
    const Hasher<L, D>& bottom_layer,
    const TreeBuilderConfig& tree_config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = tree_config.ctx.stream;

    uint64_t number_of_bottom_layer_states = pow(tree_config.arity, height);

    // This will determine how much splitting do we need to do
    // `number_of_streams` subtrees should fit in the device
    // This means each subtree should fit in `STREAM_CHUNK_SIZE` memory
    uint64_t number_of_subtrees = 1;
    uint64_t subtree_height = height;
    uint64_t subtree_bottom_layer_states = number_of_bottom_layer_states;
    uint64_t subtree_leaves_size = subtree_bottom_layer_states * input_block_len;
    uint64_t subtree_digests_size = subtree_bottom_layer_states * tree_config.digest_elements;

    size_t subtree_d_leaves_memory = std::max(
      tree_config.are_inputs_on_device ? 0 : (sizeof(L) * subtree_leaves_size),
      subtree_digests_size * sizeof(D) / tree_config.arity);
    size_t subtree_memory_required = sizeof(D) * subtree_digests_size + subtree_d_leaves_memory;
    while (subtree_memory_required > STREAM_CHUNK_SIZE) {
      number_of_subtrees *= tree_config.arity;
      subtree_height--;
      subtree_bottom_layer_states /= tree_config.arity;
      subtree_digests_size /= tree_config.arity;
      subtree_leaves_size /= tree_config.arity;
      subtree_d_leaves_memory /= tree_config.arity;
      subtree_memory_required = sizeof(D) * subtree_digests_size + subtree_d_leaves_memory;
    }
    int cap_height = height - subtree_height;
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

#ifdef MERKLE_DEBUG
    std::cout << "Available memory = " << available_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Number of streams = " << number_of_streams << std::endl;
    std::cout << "Number of subtrees = " << number_of_subtrees << std::endl;
    std::cout << "Height of a subtree = " << subtree_height << std::endl;
    std::cout << "Cutoff height = " << height - subtree_height << std::endl;
    std::cout << "Number of leaves in a subtree = " << subtree_bottom_layer_states << std::endl;
    std::cout << "Digest elements for a subtree = " << subtree_digests_size << std::endl;
    std::cout << "Size of 1 subtree digests = " << subtree_digests_size * sizeof(D) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Cap height = " << cap_height << std::endl;
    std::cout << "Enabling caps mode? " << caps_mode << std::endl;
    std::cout << "Allocated " << subtree_d_leaves_memory << " bytes for d_leaves" << std::endl;
#endif

    // Allocate memory for the leaves and digests
    // These are shared by streams in a pool
    L* d_leaves_ptr;
    D* digests_ptr;
    CHK_IF_RETURN(cudaMallocAsync(&d_leaves_ptr, subtree_d_leaves_memory * number_of_streams, stream));
    CHK_IF_RETURN(cudaMallocAsync(&digests_ptr, subtree_digests_size * number_of_streams * sizeof(D), stream));
    // Wait for these allocations to finish
    CHK_IF_RETURN(cudaStreamSynchronize(stream));

    // Build subtrees in parallel. This for loop invokes kernels that can run in a pool of size `number_of_streams`
    for (size_t subtree_idx = 0; subtree_idx < number_of_subtrees; subtree_idx++) {
      size_t stream_idx = subtree_idx % number_of_streams;
      cudaStream_t subtree_stream = streams[stream_idx];

      const L* subtree_leaves = leaves + subtree_idx * subtree_bottom_layer_states * input_block_len;
      L* subtree_d_leaves = (L*)((unsigned char*)d_leaves_ptr + stream_idx * subtree_d_leaves_memory);
      D* subtree_digests = digests_ptr + stream_idx * subtree_digests_size;

      int subtree_keep_rows = 0;
      if (tree_config.keep_rows) {
        int diff = tree_config.keep_rows - cap_height;
        subtree_keep_rows = std::max(1, diff);
      }
      device_context::DeviceContext subtree_context{subtree_stream, tree_config.ctx.device_id, tree_config.ctx.mempool};

      uint64_t start_segment_size = number_of_bottom_layer_states * tree_config.digest_elements;
      cudaError_t subtree_result = build_merkle_subtree<L, D>(
        subtree_leaves,             // leaves
        subtree_d_leaves,           // d_leves
        subtree_digests,            // digests
        subtree_idx,                // subtree_idx
        subtree_height,             // subtree_height
        caps_mode ? caps : digests, // big_tree_digests
        start_segment_size,         // start_segment_size
        0,                          // start_segment_offset
        subtree_keep_rows,          // keep_rows
        input_block_len,            // input_block_len
        bottom_layer,               // bottom_layer
        compression,                // compression
        tree_config,                // tree_config
        subtree_context             // subtree_context
      );
      CHK_IF_RETURN(subtree_result);
    }

    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
    }

    // Finish the top-level tree if any
    if (cap_height > 0) {
      size_t start_segment_size = caps_len / tree_config.arity;
      size_t start_segment_offset = 0;
      if (!caps_mode) { // Calculate offset
        size_t keep_rows = tree_config.keep_rows ? tree_config.keep_rows : height + 1;
        size_t layer_size = pow(tree_config.arity, keep_rows - 1) * tree_config.digest_elements;
        for (int i = 0; i < keep_rows - cap_height; i++) {
          start_segment_offset += layer_size;
          layer_size /= tree_config.arity;
        }
      }
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_leaves_ptr, caps_mode ? caps : (digests + start_segment_offset - caps_len), caps_len * sizeof(D),
        (caps_mode || !tree_config.are_outputs_on_device) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice, stream));

      uint64_t number_of_states = caps_len / tree_config.arity / tree_config.digest_elements;

      D* prev_layer = (D*)d_leaves_ptr;
      D* next_layer = digests_ptr;

      size_t segment_size = start_segment_size;
      size_t segment_offset = start_segment_offset;
      while (number_of_states > 0) {
        CHK_IF_RETURN(compression.run_hash_many_kernel(
          (L*)prev_layer, next_layer, number_of_states,
          tree_config.digest_elements * tree_config.arity * (sizeof(D) / sizeof(L)), tree_config.digest_elements,
          tree_config.ctx));
        if (!tree_config.keep_rows || cap_height < tree_config.keep_rows + (int)caps_mode) {
          D* digests_with_offset = digests + segment_offset;
          CHK_IF_RETURN(cudaMemcpyAsync(
            digests_with_offset, next_layer, number_of_states * tree_config.digest_elements * sizeof(D),
            cudaMemcpyDeviceToHost, stream));
          segment_offset += segment_size;
        }

        swap<D>(&prev_layer, &next_layer);

        segment_size /= tree_config.arity;
        cap_height--;
        number_of_states /= tree_config.arity;
      }
      if (caps_mode) { free(caps); }
    }

    CHK_IF_RETURN(cudaFreeAsync(d_leaves_ptr, stream));
    CHK_IF_RETURN(cudaFreeAsync(digests_ptr, stream));
    if (!tree_config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamSynchronize(streams[i]));
      CHK_IF_RETURN(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    return CHK_LAST();
  }

} // namespace merkle_tree