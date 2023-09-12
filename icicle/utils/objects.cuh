#pragma once
template <class F>
class Element
{
public:
  int v;
  __device__ __host__ Element<F>() { v = 0; }
  __device__ __host__ Element<F>(int r)
  {
    v = r % F::q;
    if (r == F::q) v = F::q;
  }
  __device__ __host__ Element<F> operator+(Element<F> const& obj)
  {
    Element<F> res;
    res.v = (v + obj.v) % F::q;
    return res;
  }
  __device__ __host__ Element<F> operator-(Element<F> const& obj)
  {
    Element<F> res;
    res.v = (v - obj.v) % F::q;
    if (res.v < 0) { res.v = F::q + res.v; }
    return res;
  }
};

template <class F>
class Scalar
{
public:
  int v;
  __device__ __host__ Scalar<F>() { v = 0; }
  __device__ __host__ Scalar<F>(int r) { v = r % F::q; }
  __device__ __host__ Scalar<F> operator+(Scalar<F> const& obj)
  {
    Scalar<F> res;
    res.v = (v + obj.v) % F::q;
    return res;
  }
  __device__ __host__ Scalar<F> operator*(Scalar<F> const& obj)
  {
    Scalar<F> res;
    res.v = (v * obj.v) % F::q;
    return res;
  }
  __device__ __host__ Element<F> operator*(Element<F> const& obj)
  {
    Element<F> res;
    res.v = (v * obj.v) % F::q;
    return res;
  }
  Scalar<F> operator-(Scalar<F> const& obj)
  {
    Scalar<F> res;
    res.v = (v - obj.v) % F::q;
    if (res.v < 0) { res.v = F::q + res.v; }
    return res;
  }
  bool operator<(Scalar<F> const& obj) { return v < obj.v; }
  static Scalar<F> one() { return Scalar<F>(1); }
  static Scalar<F> zero() { return Scalar<F>(0); }
};