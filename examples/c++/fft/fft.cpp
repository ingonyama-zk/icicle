#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;

// const long long mod = 7340033;
// const long long root = 5;
// const long long root_1 = 4404020;
// const long long root_pw = 1 << 20;

// const long long mod = 97;
// const long long root = 33;
// const long long root_pw = 1 << 3;

const long long mod = 337;
const long long root = 85;
const long long root_pw = 1 << 3;

long long extended_euclidean(long long a1, long long a2, long long *x, long long *y)
{
  // Base Case
  if (a1 == 0)
  {
    *x = 0, *y = 1;
    return a2;
  }

  // To store results of recursive call
  long long x1, y1;
  long long gcd = extended_euclidean(a2 % a1, a1, &x1, &y1);

  // Update x and y using results of recursive
  // call
  *x = y1 - (a2 / a1) * x1;
  *y = x1;

  return gcd;
}

long long inverse(long long n, long long m)
{
  long long x, y;
  long long g = extended_euclidean(n, m, &x, &y);
  if (g != 1)
  {
    cout << "No solution!";
  }
  else
  {
    x = (x + m) % m;
  }

  return x;
}

vector<long long> fft(vector<long long> a, bool invert)
{
  vector<long long> b = a;
  long long root_1 = inverse(root, mod);

  int n = b.size();
  for (int i = 1, j = 0; i < n; i++)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;

    if (i < j)
      swap(b[i], b[j]);
  }

  for (int len = 2; len <= n; len <<= 1)
  {
    long long wlen = invert ? root_1 : root;
    for (int i = len; i < root_pw; i <<= 1)
      wlen = (int)(1LL * wlen * wlen % mod);

    for (int i = 0; i < n; i += len)
    {
      int w = 1;
      for (int j = 0; j < len / 2; j++)
      {
        int u = b[i + j], v = (int)(1LL * b[i + j + len / 2] * w % mod);
        b[i + j] = u + v < mod ? u + v : u + v - mod;
        b[i + j + len / 2] = u - v >= 0 ? u - v : u - v + mod;
        w = (int)(1LL * w * wlen % mod);
      }
    }
  }

  if (invert)
  {
    long long n_1 = inverse(n, mod);
    for (long long &x : b)
    {
      x = (int)(1LL * x * n_1 % mod);
    }
    cout << endl;
  }

  return b;
}

void print_vec(vector<long long> b)
{
  int n = b.size();
  for (int i = 0; i < n; i++)
  {
    cout << b[i] << ", ";
  }
  cout << endl;
}

void test_mul_perf()
{
  int n = 1000000;
  unsigned long long a[n];
  for (int i = 0; i < n; i++)
  {
    a[i] = rand();
  }

  // auto start = std::chrono::steady_clock::now();
  clock_t begin = clock();

  unsigned long long sum = 0;
  for (int k = 0; k < 100; k++)
  {
    for (int i = 0; i < n - 4; i++)
    {
      unsigned long long t = a[i] * a[i + 1] + a[i + 2] * a[i + 3];
      sum += t;
    }
  }
  // auto end = std::chrono::steady_clock::now();
  // std::cout << "Elapsed(ms)=" << since(start).count() << std::endl;
  // std::cout << "Time difference = " << chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "[µs]" << std::endl;

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << " elapsed_secs = " << elapsed_secs << endl;
  cout << "sum = " << sum << endl;
}

int main(void)
{
  // vector<long long> a = {1, 85, 148, 111, 336, 252, 189, 226};
  // vector<long long> a = {3, 1, 4, 1, 5, 9, 2, 6};
  vector<long long> a = {31, 70, 109, 74, 334, 181, 232, 4};
  vector<long long> b = fft(a, true);
  print_vec(b);

  vector<long long> c = fft(b, false);
  print_vec(c);

  // test_mul_perf();
}
