#include "fields/field_config.cuh"
using namespace field_config;

#include "fri.cu"
#include "utils/utils.h"

namespace fri {
  /**
   * Extern "C" version of [fold_line](@ref fold_line) function with the following values of
   * template parameters (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the extension field type used for evaluations and alpha
   *  - `S` is the scalar field type used for domain elements
   * @param line_eval Pointer to the array of evaluations on the line
   * @param domain_elements Pointer to the array of domain elements
   * @param alpha The folding factor
   * @param folded_evals Pointer to the array where folded evaluations will be stored
   * @param n The number of evaluations
   * @param ctx The device context; if the stream is not 0, then everything is run async
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_line)(
    extension_t* line_eval,
    scalar_t* domain_elements,
    extension_t alpha,
    extension_t* folded_evals,
    int n,
    FriConfig& cfg)
  {
    return fri::fold_line(line_eval, domain_elements, alpha, folded_evals, n, cfg);
  };

  /**
   * Extern "C" version of [fold_circle_into_line](@ref fold_circle_into_line) function with the following values of
   * template parameters (where the field is given by `-DFIELD` env variable during build):
   *  - `E` is the extension field type used for evaluations and alpha
   *  - `S` is the scalar field type used for domain elements
   * @param circle_evals Pointer to the array of evaluations on the circle
   * @param domain_elements Pointer to the array of domain elements
   * @param alpha The folding factor
   * @param folded_line_evals Pointer to the array where folded evaluations will be stored
   * @param n The number of evaluations
   * @param ctx The device context; if the stream is not 0, then everything is run async
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, fold_circle_into_line)(
    extension_t* circle_evals,
    scalar_t* domain_elements,
    extension_t alpha,
    extension_t* folded_line_evals,
    int n,
    FriConfig& cfg)
  {
    return fri::fold_circle_into_line(circle_evals, domain_elements, alpha, folded_line_evals, n, cfg);
  };
} // namespace fri
