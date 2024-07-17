
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
  __shared__ T shmem[16][16];
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

template <class T, int REPS>
__global__ void add_many_times(T *in, T *out, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= size) return;
  T temp;
  #pragma unroll
  for (int i = 0; i < REPS; i++)
  {
    temp = i? temp + temp : in[tid];
  }
  out[tid] = temp;
}


template <class T, int REPS>
__global__ void multi_add(T *in, T *out, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int segment_length = size / REPS;
  if (tid >= segment_length) return;
  // #pragma unroll 1
  for (int i = 0; i < REPS; i++)
  {
    out[tid + i*segment_length] = in[tid + i*segment_length] + in[tid + i*segment_length];
  }
}

template <class T, int SEG_SIZE>
__global__ void segment_sum(T *inout, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int nof_segments = size / SEG_SIZE;
  if (tid >= nof_segments) return;
  T sum = T::zero();
  T sums_sum = T::zero();
  for (int i = 0; i < SEG_SIZE; i++)
  {
    sums_sum = sums_sum + sum;
    sum = sum + inout[tid * SEG_SIZE + i];
  }
  inout[tid * SEG_SIZE] = sums_sum;
  // inout[tid * SEG_SIZE] = sum;
}

template <class T, int SEG_SIZE>
__global__ void shmem_segment_sum(T *inout, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int nof_segments = size / SEG_SIZE;
  if (tid >= nof_segments) return;
  __shared__ T shmem[128*2];
  // T sum = T::zero();
  // T sums_sum = T::zero();
  shmem[2*threadIdx.x] = T::zero(); //sum
  shmem[2*threadIdx.x + 1] = T::zero(); //sums_sum
  for (int i = 0; i < SEG_SIZE; i++)
  {
    {T sum = shmem[2*threadIdx.x];
    T sums_sum = shmem[2*threadIdx.x + 1];
    shmem[2*threadIdx.x + 1] = sums_sum + sum;}
    // {T sum = shmem[2*(127-threadIdx.x)];
    // T sums_sum = shmem[2*(127-threadIdx.x) + 1];
    // shmem[2*(127-threadIdx.x) + 1] = sums_sum + sum;}
    // shmem[2*(127-threadIdx.x) + 1] = shmem[2*(127-threadIdx.x) + 1] + shmem[2*(127-threadIdx.x)];
    // shmem[2*threadIdx.x + 1] = shmem[2*threadIdx.x + 1] + shmem[2*threadIdx.x];
    // __syncthreads();
    {T sum = shmem[2*threadIdx.x];
    T sums_sum = inout[tid * SEG_SIZE + i];
    shmem[2*threadIdx.x] = sum + sums_sum;}
    // shmem[2*threadIdx.x] = shmem[2*threadIdx.x] + inout[tid * SEG_SIZE + i];
    // __syncthreads();
  }
  inout[tid * SEG_SIZE] = shmem[2*threadIdx.x + 1];
  // inout[tid * SEG_SIZE] = sum;
}

template <class T, int REPS>
__global__ void multi_mult(T *in, T *out, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int segment_length = size / REPS;
  if (tid >= segment_length) return;
  #pragma unroll 1
  for (int i = 0; i < REPS; i++)
  {
    out[tid + i*segment_length] = in[tid + i*segment_length] * in[tid + i*segment_length];
  }
}


template <class E>
DEVICE_INLINE void ntt8opt(E& X0, E& X1, E& X2, E& X3, E& X4, E& X5, E& X6, E& X7)
  {
    E T;

    T = X3 - X7;
    X7 = X3 + X7;
    X3 = X1 - X5;
    X5 = X1 + X5;
    X1 = X2 + X6;
    X2 = X2 - X6;
    X6 = X0 + X4;
    X0 = X0 - X4;

    X4 = X6 + X1;
    X6 = X6 - X1;
    X1 = X3 + T;
    X3 = X3 - T;
    T = X5 + X7;
    X5 = X5 - X7;
    X7 = X0 + X2;
    X0 = X0 - X2;

    X2 = X6 + X5;
    X6 = X6 - X5;
    X5 = X7 - X1;
    X1 = X7 + X1;
    X7 = X0 - X3;
    X3 = X0 + X3;
    X0 = X4 + T;
    X4 = X4 - T;
  }


  template <class E>
DEVICE_INLINE void ntt8(E& X0, E& X1, E& X2, E& X3, E& X4, E& X5, E& X6, E& X7)
  {
    E Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7;

    Y0 = X0 + X4;
    Y1 = X0 - X4;
    Y2 = X1 - X5;
    Y3 = X1 + X5;
    Y4 = X2 + X6;
    Y5 = X2 - X6;
    Y6 = X3 - X7;
    Y7 = X3 + X7;

    X0 = Y0 + Y2;
    X1 = Y0 - Y2;
    X2 = Y1 - Y3;
    X3 = Y1 + Y3;
    X4 = Y4 + Y6;
    X5 = Y4 - Y6;
    X6 = Y5 - Y7;
    X7 = Y5 + Y7;

    Y0 = X0 + X1;
    Y1 = X0 - X1;
    Y2 = X2 - X3;
    Y3 = X2 + X3;
    Y4 = X4 + X5;
    Y5 = X4 - X5;
    Y6 = X6 - X7;
    Y7 = X6 + X7;

    X0 = Y0;
    X1 = Y1;
    X2 = Y2;
    X3 = Y3;
    X4 = Y4;
    X5 = Y5;
    X6 = Y6;
    X7 = Y7;
  }



template <class T>
__global__ void multi_ntt8(T *in, T *out, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int segment_length = size / 8;
  if (tid >= segment_length) return;
  T X[8];
  #pragma unroll
  for (int i = 0; i < 8; i++)
  {
    X[i] = in[tid + i*segment_length];
  }
  // ntt8(X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7]);
  ntt8opt(X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7]);
  #pragma unroll
  for (int i = 0; i < 8; i++)
  {
    out[tid + i*segment_length] = X[i];
  }
}


__device__ void mul_naive(uint32_t *a, uint32_t *b, uint32_t *r){
    __align__(8) uint32_t odd[2];
    r[0] = ptx::mul_lo(a[0], b[0]);
    r[1] = ptx::mul_hi(a[0], b[0]);
    r[1] = ptx::mad_lo(a[0], b[1], r[1]);
    r[1] = ptx::mad_lo(a[1], b[0], r[1]);
    r[2] = ptx::mul_lo(a[1], b[1]);
    r[2] = ptx::mad_hi(a[1], b[0], r[2]);
    r[2] = ptx::mad_hi(a[0], b[1], r[2]);
    r[3] = ptx::mul_hi(a[1], b[1]);
  
    r[0] = ptx::add_cc(r[0], r[1]);
    r[1] = ptx::add_cc(r[2], r[3]);
}

__device__ void mul_icicle(uint32_t *a, uint32_t *b, uint32_t *r){
    __align__(8) uint32_t odd[2];
    r[0] = ptx::mul_lo(a[0], b[0]);
    r[1] = ptx::mul_hi(a[0], b[0]);
    r[2] = ptx::mul_lo(a[1], b[1]);
    r[3] = ptx::mul_hi(a[1], b[1]);
    odd[0] = ptx::mul_lo(a[0], b[1]);
    odd[1] = ptx::mul_hi(a[0], b[1]);
    odd[0] = ptx::mad_lo(a[1], b[0], odd[0]);
    odd[1] = ptx::mad_hi(a[1], b[0], odd[1]);
    r[1] = ptx::add_cc(r[1], odd[0]);
    r[2] = ptx::addc_cc(r[2], odd[1]);
    r[3] = ptx::addc(r[3], 0);

    r[0] = ptx::add_cc(r[0], r[1]);
    r[1] = ptx::add_cc(r[2], r[3]);
}



template <int REPS>
__global__ void limb_mult_bench(uint32_t *in, uint32_t *out, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid >= size/2) return;
  uint32_t res[4];
  res[0] = in[tid];
  res[1] = in[tid + size/2];
  // typename T::Wide temp;
  for (int i = 0; i < REPS; i++)
  {
    mul_naive(res, res, res);
    // mul_icicle(res, res, res);
    // T::multiply_raw_device(res.limbs_storage, res.limbs_storage, res.limbs_storage);
    // temp = T::mul_wide(res, res);
  }
  // out[tid] = T::reduce(temp);
  out[tid] = res[0];
  out[tid + size/2] = res[1];
}
