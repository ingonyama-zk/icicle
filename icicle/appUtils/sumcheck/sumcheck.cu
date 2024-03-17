

#define SHMEM_SIZE 64
#define MAX_SHMEM_LOG_SIZE 6

template <typename S>
__global__ void sum_reduction(S *v, S *v_r) {
	// Allocate shared memory
	__shared__ S partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 1; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0 || threadIdx.x == 1) {
		// printf("debug tid %d, val %d\n", threadIdx.x, partial_sum[threadIdx.x]);
		v_r[2*blockIdx.x + threadIdx.x] = partial_sum[threadIdx.x];
	}
}

template <typename S>
__global__ void update_evals_kernel(S* evals, S alpha){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  evals[tid] = (S::one() - alpha) * evals[2*tid] + alpha * evals[2*tid+1];
  // evals[tid] = (1 - alpha) * evals[2*tid] + alpha * evals[2*tid+1];
}

template <typename S>
void accumulate(S* in, S* out, uint32_t log_size, cudaStream_t stream){
  uint32_t nof_steps = (log_size - 1) / MAX_SHMEM_LOG_SIZE;
  uint32_t last_step_size = (log_size - 1) % MAX_SHMEM_LOG_SIZE;
	printf("nof steps %d last size %d\n", nof_steps, last_step_size);
  for (int i = 0; i < nof_steps; i++)
  {
    sum_reduction<<<1<<(log_size - 1 -(MAX_SHMEM_LOG_SIZE)*(i+1)), SHMEM_SIZE,0,stream>>>(i? out : in, out);
		printf("nof blocks %d\n", 1<<(log_size-(MAX_SHMEM_LOG_SIZE)*(i+1)-1));
		// cudaDeviceSynchronize();
  	// printf("cuda err %d\n", cudaGetLastError());
  }
  if (last_step_size) sum_reduction<<<1, 1<<last_step_size, 0,stream>>>(nof_steps? out : in, out);
	// cudaDeviceSynchronize();
  // printf("cuda err last %d\n", cudaGetLastError());
}

template <typename S>
__global__ void add_to_trace(S* trace, S* vals, int p){
	  trace[2*p+1] = vals[0];
    trace[2*p+2] = vals[1];
		// printf("%d  %d\n", vals[0], vals[1]);
}

template <typename S>
S my_hash(){
	return 1;
}

template <typename S>
void sumcheck_alg1(S* evals, S* t, S* T, S C, int n, cudaStream_t stream){
	// S alpha = 1;
	S alpha = S::one();
  // S alpha = my_hash(/*T, C*/);
  // S rp_even, rp_odd;
  for (int p = 0; p < n-1; p++)
  {
    int nof_threads = 1<<(n-1-p);
    // move update kernel here and unify
    // reduction_kernel<<<nof_threads>>>(evals, t, n-p); //accumulation
    accumulate(evals, t, n-p, stream); //accumulation
		add_to_trace<<<1,1,0,stream>>>(T, t, p);
    // T[2*p+1] = t[0];
    // T[2*p+2] = t[1];
    // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
		int NOF_THREADS = 256;
		int NOF_BLOCKS = (nof_threads + NOF_THREADS - 1) / NOF_THREADS;
    update_evals_kernel<<<NOF_BLOCKS, NOF_THREADS,0, stream>>>(evals, alpha); //phase 3
  }
	add_to_trace<<<1,1,0,stream>>>(T, evals, n-1);

}

template <typename S>
void sumcheck_alg1_ref(S* evals, S* t, S* T, S C, int n){
  // S alpha = my_hash(/*T, C*/);
	// S alpha = 1;
	S alpha = S::one();
  S rp_even, rp_odd;
  for (int p = 0; p < n; p++)
  {
		// rp_even = 0; rp_odd = 0;
		rp_even = S::zero(); rp_odd = S::zero();
		// printf("evals\n");
		// for (int i = 0; i < 1<<(n-p); i++)
		// {
		// 	printf("%d, ",evals[i]);
		// }
		// printf("\n");
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			rp_even = rp_even + evals[2*i];
			rp_odd = rp_odd + evals[2*i+1];
		}
    T[2*p+1] = rp_even;
    T[2*p+2] = rp_odd;
    // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
		// alpha = 1;
		alpha = S::one();
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			t[i] = (S::one() - alpha) * evals[2*i] + alpha * evals[2*i+1];
			// t[i] = (1-alpha)*evals[2*i] + alpha*evals[2*i+1];
		}
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			evals[i] = t[i];
		}
  }
}