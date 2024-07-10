
template <class T>
__global__ void add_elements_kernel(const T* x, const T* y, T* result, const unsigned count)
{
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  result[tid] = x[tid] + y[tid];
}

template <class T>
__global__ void fake_ntt_kernel(const T* x, T* result, const unsigned thread_count)
{
  extern __shared__ T shmem[];
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= thread_count) return;
  shmem[4*threadIdx.x] = x[4*tid] + x[4*tid+1];
  shmem[4*threadIdx.x+1] = x[4*tid] + T::neg(x[4*tid+1]);
  shmem[4*threadIdx.x+2] = x[4*tid+2] + x[4*tid+3];
  shmem[4*threadIdx.x+3] = x[4*tid+2] + T::neg(x[4*tid+3]);
  __syncthreads();
  result[4*tid] = shmem[2*threadIdx.x] + shmem[2*threadIdx.x + 4*blockDim.x/2];
  result[4*tid+1] = shmem[2*threadIdx.x] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x/2]);
  result[4*tid+2] = shmem[2*threadIdx.x+1] + shmem[2*threadIdx.x + 4*blockDim.x/2+1];
  result[4*tid+3] = shmem[2*threadIdx.x+1] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x/2+1]);
}


template <class T>
__global__ void bugged_add_elements_kernel(const T* x, const T* y, T* result, const unsigned count)
{
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  // if (tid >= count) return;
  // printf("tid %d\n", tid);
  result[tid] = x[tid] + y[tid];
}

template <class T>
__global__ void bugged_fake_ntt_kernel(const T* x, T* result, const unsigned thread_count)
{
  extern __shared__ T shmem[];
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // if (tid >= thread_count) return;
  // if (tid == 0){
  //   for (int i = 0; i < 8; i++)
  //   {
  //     shmem[i]=T::zero();
  //   }
  // }

  shmem[4*threadIdx.x] = x[4*tid] + x[4*tid+1];
  shmem[4*threadIdx.x+1] = x[4*tid] + T::neg(x[4*tid+1]);
  shmem[4*threadIdx.x+2] = x[4*tid+2] + x[4*tid+1];
  shmem[4*threadIdx.x+4] = x[4*tid+2] + T::neg(x[4*tid+1]);

  __syncthreads();

  // if (tid == 0){
  //   for (int i = 0; i < 8; i++)
  //   {
  //     printf("%d ",shmem[i]);
  //   }
  //   printf("\n");
  // }

  // printf("tid: %d, addr1: %d, addr2: %d\n", tid, 2*threadIdx.x, 2*threadIdx.x + 4*blockDim.x);
  result[4*tid] = shmem[2*threadIdx.x] + shmem[2*threadIdx.x + 4*blockDim.x];  // Incorrect offset
  result[4*tid+1] = shmem[2*threadIdx.x] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x]);  // Incorrect offset
  result[4*tid+2] = shmem[2*threadIdx.x+1] + shmem[2*threadIdx.x + 4*blockDim.x+1];  // Incorrect offset
  result[4*tid+3] = shmem[2*threadIdx.x+1] + T::neg(shmem[2*threadIdx.x +4*blockDim.x+1]);  // Incorrect offset
}

template <class T>
__global__ void bucket_acc_naive(T* buckets, unsigned* indices, unsigned* sizes, unsigned nof_buckets){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nof_buckets) return;
  for (int i = 0; i < sizes[tid]; i++)
  {
    buckets[indices[tid]] = buckets[indices[tid]] + buckets[indices[tid]];
  }
}

template <class T>
__global__ void bucket_acc_memory_baseline(T* buckets1, T* buckets2, unsigned* indices, unsigned nof_buckets){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= nof_buckets) return;
  buckets2[indices[tid]] = buckets1[indices[tid]];
}

template <class T>
__global__ void bucket_acc_compute_baseline(T* buckets, unsigned* indices, unsigned* sizes, unsigned nof_buckets){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= nof_buckets) return;
  T bucket = buckets[indices[tid]];
  for (int j = 0; j < 100; j++)
  {
    for (int i = 0; i < sizes[tid]; i++)
    {
      bucket = bucket + bucket;
    }
  }
  buckets[indices[tid]] = bucket;
}

template <class T>
__global__ void bucket_acc_reg(T* buckets, unsigned* indices, unsigned* sizes, unsigned nof_buckets){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= nof_buckets) return;
  T bucket = buckets[indices[tid]];
  for (int i = 0; i < sizes[tid]; i++)
  {
    bucket = bucket + bucket;
  }
  buckets[indices[tid]] = bucket;
}


// #define NOF_TH 32*64


template <class T, int SIZE_T>
__global__ void device_memory_copy(void* arr1_raw, void* arr2_raw, unsigned size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= size/SIZE_T) return;
  T* arr1=(T*)arr1_raw;
  T* arr2=(T*)arr2_raw;
  arr2[tid] = arr1[tid];
}

template <class T, int SIZE_T>
__global__ void segmented_memory_copy(void* arr1_raw, void* arr2_raw, unsigned size, unsigned read_segment_length, unsigned nof_write_segments){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int nof_elements = size/SIZE_T;
  int write_segment_length = nof_elements / nof_write_segments;
  int r_segment_idx = tid / read_segment_length;
  int r_segment_tid = tid % read_segment_length;
  int w_segment_idx = r_segment_idx % nof_write_segments;
  int w_segment_tid = r_segment_idx / nof_write_segments;
  int addr = w_segment_idx * write_segment_length + w_segment_tid * read_segment_length + r_segment_tid;
  // if (tid < 50) printf("tid %d, addr %d\n", tid, addr);
  if (tid >= nof_elements) return;
  T* arr1=(T*)arr1_raw;
  T* arr2=(T*)arr2_raw;
  arr2[addr] = arr1[addr];
}


template <class T, int SIZE_T>
__global__ void multi_memory_copy1(void* arr1_raw, void* arr2_raw, unsigned size, unsigned nof_elements_per_thread){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int nof_elements = size/SIZE_T;
  int segment_length = nof_elements / nof_elements_per_thread;
  if (tid >= segment_length) return;
  T* arr1=(T*)arr1_raw;
  T* arr2=(T*)arr2_raw;
  for (int i = 0; i < nof_elements_per_thread; i++)
  {
    arr2[tid + i*segment_length] = arr1[tid + i*segment_length];
  }
}

template <class T, int SIZE_T>
__global__ void multi_memory_copy2(void* arr1_raw, void* arr2_raw, unsigned size, unsigned nof_elements_per_thread){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int nof_elements = size/SIZE_T;
  int nof_threads = nof_elements / nof_elements_per_thread;
  if (tid >= nof_threads) return;
  T* arr1=(T*)arr1_raw;
  T* arr2=(T*)arr2_raw;
  for (int i = 0; i < nof_elements_per_thread; i++)
  {
    arr2[tid*nof_elements_per_thread + i] = arr1[tid*nof_elements_per_thread + i];
  }
}

template <class T>
__global__ void simple_memory_copy(T* in, T* out, unsigned size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= size) return;
  out[tid] = in[tid];
}

template <class T>
__global__ void naive_transpose_write(T *in, T *out, int row_length){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= row_length * row_length) return;
  int row_id = tid / row_length;
  int col_id = tid % row_length;
  out[col_id * row_length + row_id] = in[tid];
}

template <class T>
__global__ void naive_transpose_read(T *in, T *out, int row_length){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= row_length * row_length) return;
  int row_id = tid / row_length;
  int col_id = tid % row_length;
  out[tid] = in[col_id * row_length + row_id];
}


template <class T>
__global__ void shmem_transpose(T *in, T *out, int row_length){
  __shared__ T shmem[16][17];
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= row_length * row_length) return;
  int shmem_col_id = threadIdx.x / 16;
  int shmem_row_id = threadIdx.x % 16;
  int blocks_per_row = row_length / 16;
  int block_row_id = blockIdx.x / blocks_per_row;
  int block_col_id = blockIdx.x % blocks_per_row;
  // shmem[shmem_col_id][shmem_row_id] = in[block_row_id*row_length*16 + block_col_id*16 + shmem_col_id*row_length + shmem_row_id];
  shmem[shmem_row_id][shmem_col_id] = in[block_row_id*row_length*16 + block_col_id*16 + shmem_col_id*row_length + shmem_row_id];
  __syncthreads();
  // out[block_col_id*row_length*16 + block_row_id*16 + shmem_col_id*row_length + shmem_row_id] = shmem[shmem_row_id][shmem_col_id];
  out[block_col_id*row_length*16 + block_row_id*16 + shmem_col_id*row_length + shmem_row_id] = shmem[shmem_col_id][shmem_row_id];
}