


template <typename S>
void sumcheck_alg1(S* evals, S* t, S* T, S C, int n){
  S alpha = my_hash(T, C);
  // S rp_even, rp_odd;
  for (int p = 0; p < n; p++)
  {
    int nof_threads = 1<<(n-1-p);
    // move update kernel here and unify
    reduction_kernel<<<nof_threads>>>(evals, t, n-p); //accumulation
    T[2*p+1] = t[0];
    T[2*p+2] = t[1];
    alpha = my_hash(alpha, t[0], t[1]); //phase 2
    update_evals_kernel<<<nof_threads>>>(evals, alpha); //phase 3
  }
  
}