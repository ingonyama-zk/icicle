#pragma once


int interpolate_scalars(scalar_t* d_out, scalar_t* d_evaluations, scalar_t* d_domain, unsigned n);

int interpolate_scalars_batch(scalar_t* d_out, scalar_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size);

int interpolate_points(projective_t* d_out, projective_t* d_evaluations, scalar_t* d_domain, unsigned n);

int interpolate_points_batch(projective_t* d_out, projective_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size);

int evaluate_scalars(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n);

int evaluate_scalars_batch(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n, unsigned batch_size);

int evaluate_points(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n);

int evaluate_points_batch(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, 
                          unsigned domain_size, unsigned n, unsigned batch_size);

int evaluate_scalars_on_coset(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, 
                              unsigned domain_size, unsigned n, scalar_t* coset_powers);

int evaluate_scalars_on_coset_batch(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                    unsigned n, unsigned batch_size, scalar_t* coset_powers);

int evaluate_points_on_coset(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, 
                             unsigned domain_size, unsigned n, scalar_t* coset_powers);

int evaluate_points_on_coset_batch(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                   unsigned n, unsigned batch_size, scalar_t* coset_powers);
