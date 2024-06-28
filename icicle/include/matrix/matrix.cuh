#pragma once
#ifndef MATRIX_H
#define MATRIX_H

namespace matrix {
  template <typename T>
  struct Matrix {
    T* values;
    size_t width;
    size_t height;
  };
} // namespace matrix

#endif