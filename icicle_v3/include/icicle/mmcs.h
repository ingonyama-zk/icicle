#pragma once

#include "errors.h"
#include "runtime.h"
#include "hash.h"
#include "merkle_tree.h"
#include "icicle/utils/utils.h"

#include <cstdint>
#include <functional>


template <typename T>
  struct Matrix {
    T* values;
    size_t width;
    size_t height;
  };

eIcicleError build_mmcs_tree(const Matrix<limb_t>* inputs,
    const unsigned int number_of_inputs,
    limb_t** outputs,
    const Hash& hash,
    const Hash& compression,
    const MerkleTreeConfig& config);
    
    //create hash <-hasher,compressor

    //sort, and call merkle tree
    //how to return outputs?