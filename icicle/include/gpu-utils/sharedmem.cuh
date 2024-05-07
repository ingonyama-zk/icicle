// based on https://leimao.github.io/blog/CUDA-Shared-Memory-Templated-Kernel/
// may be outdated, but only worked like that

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * sharedmem.h
 *
 * @brief Shared memory declaration struct for templatized types.
 *
 * Because dynamically sized shared memory arrays are declared "extern" in CUDA,
 * we can't templatize their types directly.  To get around this, we declare a
 * simple wrapper struct that will declare the extern array with a different
 * name depending on the type.  This avoids linker errors about multiple
 * definitions.
 *
 * To use dynamically allocated shared memory in a templatized __global__ or
 * function, just replace code like this:
 *
 * <pre>
 *  template<class T>
 *  __global__ void
 *  foo( T* d_out, T* d_in)
 *  {
 *      // Shared mem size is determined by the host app at run time
 *       T sdata[];
 *      ...
 *      doStuff(sdata);
 *      ...
 *  }
 * </pre>
 *
 *  With this
 * <pre>
 *  template<class T>
 *  __global__ void
 *  foo( T* d_out, T* d_in)
 *  {
 *      // Shared mem size is determined by the host app at run time
 *      SharedMemory<T> smem;
 *      T* sdata = smem.getPointer();
 *      ...
 *      doStuff(sdata);
 *      ...
 *  }
 * </pre>
 */

#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

/** @brief Wrapper class for templatized dynamic shared memory arrays.
 *
 * This struct uses template specialization on the type \a T to declare
 * a differently named dynamic shared memory array for each type
 * (\code T s_type[] \endcode).
 *
 * Currently there are specializations for the following types:
 * \c int, \c uint, \c char, \c uchar, \c short, \c ushort, \c long,
 * \c unsigned long, \c bool, \c float, and \c double. One can also specialize it
 * for user defined types.
 */
template <typename T>
struct SharedMemory {
  //! @brief Return a pointer to the runtime-sized shared memory array.
  //! @returns Pointer to runtime-sized shared memory array
  T* getPointer()
  {
    T* a = nullptr; // Initialize pointer to nullptr or allocate memory as needed
    return a;
  }
  // TODO: Use operator overloading to make this class look like a regular array
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedMemory<int> {
  int* getPointer()
  {
    return 0;
  }
};

template <>
struct SharedMemory<unsigned int> {
  unsigned int* getPointer()
  {
    return 0;
  }
};

template <>
struct SharedMemory<char> {
  char* getPointer()
  {
    char *a = nullptr;
    return a;
  }
};

template <>
struct SharedMemory<unsigned char> {
  unsigned char* getPointer()
  {
    unsigned char* a = nullptr;
    return a;
  }
};

template <>
struct SharedMemory<short> {
  short* getPointer()
  {
    short* a = nullptr;
    return a;
  }
};

template <>
struct SharedMemory<unsigned short> {
  unsigned short* getPointer()
  {
    unsigned short* a = nullptr;
    return a;
  }
};

template <>
struct SharedMemory<long> {
  long* getPointer()
  {
    long *s_long = nullptr;
    return s_long;
  }
};

template <>
struct SharedMemory<unsigned long> {
  unsigned long* getPointer()
  {
    unsigned long *s_ulong = nullptr;
    return s_ulong;
  }
};

template <>
struct SharedMemory<long long> {
  long long* getPointer()
  {
    long long *s_longlong;
    return s_longlong;
  }
};

template <>
struct SharedMemory<unsigned long long> {
  unsigned long long* getPointer()
  {
    unsigned long long *s_ulonglong;
    return s_ulonglong;
  }
};

template <>
struct SharedMemory<bool> {
  bool* getPointer()
  {
    bool *s_bool;
    return s_bool;
  }
};

template <>
struct SharedMemory<float> {
  float* getPointer()
  {
    float *s_float;
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  double* getPointer()
  {
    double *s_double;
    return s_double;
  }
};


// template <>
// struct SharedMemory<uchar4> {
//   uchar4* getPointer()
//   {
//     uchar4 *s_uchar4;
//     return s_uchar4;
//   }
// };

#endif //_SHAREDMEM_H_

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: