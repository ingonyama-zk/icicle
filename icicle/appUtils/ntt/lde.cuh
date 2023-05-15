#ifndef LDE_H
#define LDE_H
#pragma once

template <typename S> 
int interpolate_scalars(S* d_out, S* d_evaluations, S* d_domain, unsigned n);

template <typename S> 
int interpolate_scalars_batch(S* d_out, S* d_evaluations, S* d_domain, unsigned n, unsigned batch_size);

template <typename E, typename S> 
int interpolate_points(E* d_out, E* d_evaluations, S* d_domain, unsigned n);

template <typename E, typename S> 
int interpolate_points_batch(E* d_out, E* d_evaluations, S* d_domain, unsigned n, unsigned batch_size);

template <typename S> 
int evaluate_scalars(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, unsigned n);

template <typename S> 
int evaluate_scalars_batch(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, unsigned n, unsigned batch_size);

template <typename E, typename S> 
int evaluate_points(E* d_out, E* d_coefficients, S* d_domain, unsigned domain_size, unsigned n);

template <typename E, typename S> 
int evaluate_points_batch(E* d_out, E* d_coefficients, S* d_domain, 
                          unsigned domain_size, unsigned n, unsigned batch_size);

template <typename S> 
int evaluate_scalars_on_coset(S* d_out, S* d_coefficients, S* d_domain, 
                              unsigned domain_size, unsigned n, S* coset_powers);

template <typename S>                               
int evaluate_scalars_on_coset_batch(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, 
                                    unsigned n, unsigned batch_size, S* coset_powers);

template <typename E, typename S> 
int evaluate_points_on_coset(E* d_out, E* d_coefficients, S* d_domain, 
                             unsigned domain_size, unsigned n, S* coset_powers);

template <typename E, typename S> 
int evaluate_points_on_coset_batch(E* d_out, E* d_coefficients, S* d_domain, unsigned domain_size,
                                   unsigned n, unsigned batch_size, S* coset_powers);

#endif