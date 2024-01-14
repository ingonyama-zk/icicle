#include "merkle.cuh"

namespace merkle {
  static constexpr size_t GIGA = 1024 * 1024 * 1024;

  /// Bytes per stream
  static constexpr size_t STREAM_CHUNK_SIZE = 1024 * 1024 * 1024;

  /// Flattens the tree digests and sum them up to get
  /// the memory needed to contain all the digests
  size_t get_digests_len(uint32_t height, uint32_t arity)
  {
    size_t digests_len = 0;
    size_t row_length = 1;
    for (int i = 1; i < height; i++) {
      digests_len += row_length;
      row_length *= arity;
    }

    return digests_len;
  }

  /// Construct merkle subtree without parallelization
  /// We would need to align the digests of a big tree correctly
  /// The digests are aligned sequentially per row
  /// Example:
  ///
  /// Big tree:
  ///
  ///        1
  ///       / \
  ///      2   3
  ///     / \ / \
  ///    4  5 6  7
  ///
  /// Subtree 1    Subtree 2
  ///    2            3
  ///   / \          / \
  ///  4   5        6   7
  ///
  /// Digests array for subtree 1:
  /// [4 5 . . 2 .]
  /// |   |    |
  /// -----    V
  ///   |    Segment (offset = 4, subtree_idx = 0)
  ///   v
  /// Segment (offset = 0, subtree_idx = 0)
  ///
  /// Digests array for subtree 2:
  /// [. . 6 7 . 3]
  ///     |   |
  ///     -----
  ///       |
  ///       v
  ///    Segment (offset = 0, subtree_idx = 1)
  ///
  /// Total digests array:
  /// [4 5 6 7 2 3]
  template <typename S, int T>
  cudaError_t __build_merkle_subtree(
    S* state,
    S* digests,
    size_t subtree_idx,
    size_t leaves_size,
    S* big_tree_digests,
    size_t start_segment_size,
    PoseidonConstants<S>& poseidon,
    cudaStream_t &stream)
  {
    PoseidonConfig config = default_poseidon_config(T);
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;
    config.input_is_a_state = true;
    config.loop_results = true;
    config.ctx.stream = stream;
    int arity = T - 1;
    uint32_t number_of_blocks = leaves_size / arity;
    size_t segment_size = start_segment_size;
    size_t segment_offset = 0;

    while (number_of_blocks > 0) {
      cudaError_t poseidon_res = poseidon_hash<S, T>(state, digests, number_of_blocks, poseidon, config);
      CHK_IF_RETURN(poseidon_res);

      S* digests_with_offset = big_tree_digests + segment_offset + subtree_idx * number_of_blocks;
      CHK_IF_RETURN(cudaMemcpyAsync(digests_with_offset, digests, number_of_blocks * sizeof(S), cudaMemcpyDeviceToHost, stream));

      number_of_blocks /= arity;
      segment_offset += segment_size;
      segment_size /= arity;
      config.aligned = true;
    }

    return CHK_LAST();
  }

  template <typename S, int T>
  cudaError_t build_merkle_tree(const S* leaves, S* digests, uint32_t height, poseidon::PoseidonConstants<S>& poseidon, MerkleConfig& config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;

    int arity = T - 1;
    uint32_t number_of_leaves = pow(arity, (height - 1));

    // This will determine how much splitting do we need to do
    // `number_of_streams` subtrees should fit in the device
    // This means each subtree should fit in `STREAM_CHUNK_SIZE` memory
    uint32_t number_of_subtrees = 1;
    uint32_t subtree_height = height;
    uint32_t subtree_leaves_size = pow(arity, height - 1);
    uint32_t subtree_state_size = subtree_leaves_size / arity * T;
    uint32_t subtree_digests_size = subtree_state_size / arity;
    size_t subtree_memory_required = sizeof(S) * (subtree_state_size + subtree_digests_size);
    while (subtree_memory_required > STREAM_CHUNK_SIZE) {
      number_of_subtrees *= arity;
      subtree_height--;
      subtree_leaves_size /= arity;
      subtree_state_size = subtree_leaves_size / arity * T;
      subtree_digests_size = subtree_state_size / arity;
      subtree_memory_required = sizeof(S) * (subtree_state_size + subtree_digests_size);
    }

    size_t available_memory, _total_memory;
    CHK_IF_RETURN(cudaMemGetInfo(&available_memory, &_total_memory));
    available_memory -= GIGA / 8; // Leave 128 MB

    // We can effectively parallelize memory copy with streams
    // as long as they don't operate on more than `STREAM_CHUNK_SIZE` bytes
    const size_t number_of_streams = std::min((uint32_t)(available_memory / STREAM_CHUNK_SIZE), number_of_subtrees);
    cudaStream_t* streams = static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * number_of_streams));
    for (size_t i = 0; i < number_of_streams; i++) {
      CHK_IF_RETURN(cudaStreamCreate(&streams[i]));
    }

  #if !defined(__CUDA_ARCH__) && defined(MERKLE_DEBUG)
    std::cout << "Available memory = " << available_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Number of streams = " << number_of_streams << std::endl;
    std::cout << "Number of subtrees = " << number_of_subtrees << std::endl;
    std::cout << "Height of a subtree = " << subtree_height << std::endl;
    std::cout << "Cutoff height = " << height - subtree_height + 1 << std::endl;
    std::cout << "Number of leaves in a subtree = " << subtree_leaves_size << std::endl;
    std::cout << "State of a subtree = " << subtree_state_size << std::endl;
    std::cout << "Digest elements for a subtree = " << get_digests_len(subtree_height, arity) << std::endl;
    std::cout << "Size of 1 subtree states = " << subtree_state_size * sizeof(S) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Size of 1 subtree digests = " << subtree_digests_size * sizeof(S) / 1024 / 1024 << " MB" << std::endl;
  #endif

    // Allocate memory for the leaves and digests
    // These are shared by streams in a pool
    S *states_ptr, *digests_ptr;
    CHK_IF_RETURN(cudaMallocAsync(&states_ptr, subtree_state_size * number_of_streams * sizeof(S), stream))
    CHK_IF_RETURN(cudaMallocAsync(&digests_ptr, subtree_digests_size * number_of_streams * sizeof(S), stream))
    // We should wait for these allocations to finish in order to proceed
    CHK_IF_RETURN(cudaStreamSynchronize(stream));

    for (size_t subtree_idx = 0; subtree_idx < number_of_subtrees; subtree_idx++) {
      size_t stream_idx = subtree_idx % number_of_streams;
      cudaStream_t subtree_stream = streams[stream_idx];

      const S* subtree_leaves = leaves + subtree_idx * subtree_leaves_size;
      S* subtree_state = states_ptr + stream_idx * subtree_state_size;
      S* subtree_digests = digests_ptr + stream_idx * subtree_digests_size;

      // We need to copy the first level from RAM to device
      // The pitch property of cudaMemcpy2D will allow us to deal with shape differences
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        subtree_state, T * sizeof(S),      // Device pointer and device pitch
        subtree_leaves, arity * sizeof(S), // Host pointer and pitch
        arity * sizeof(S),                 // Size of the source matrix (Arity)
        subtree_leaves_size / arity,       // Size of the source matrix (Number of blocks)
        cudaMemcpyHostToDevice, subtree_stream));    // Direction and stream
      
      cudaError_t subtree_result = __build_merkle_subtree<S, T>(
              subtree_state, subtree_digests, subtree_idx, subtree_leaves_size, digests, number_of_leaves / arity,
              poseidon, subtree_stream);
      CHK_IF_RETURN(subtree_result);
    }

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
}