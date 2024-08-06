
#include "icicle/backend/polynomial_backend.h"
#include "icicle/polynomials/default_backend/default_poly_context.h"
#include "icicle/polynomials/default_backend/default_poly_backend.h"

namespace polynomials {

  using icicle::DefaultPolynomialBackend;
  using icicle::DefaultPolynomialContext;

  /*============================== Polynomial CPU-factory ==============================*/

  template <typename C = scalar_t, typename D = C, typename I = C>
  class CPUPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
  public:
    CPUPolynomialFactory() {}
    ~CPUPolynomialFactory() {}
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialContext<C, D, I>> CPUPolynomialFactory<C, D, I>::create_context()
  {
    return std::make_shared<DefaultPolynomialContext<C, D, I>>(nullptr);
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialBackend<C, D, I>> CPUPolynomialFactory<C, D, I>::create_backend()
  {
    return std::make_shared<DefaultPolynomialBackend<C, D, I>>(nullptr);
  }

  /************************************** BACKEND REGISTRATION **************************************/

  REGISTER_SCALAR_POLYNOMIAL_FACTORY_BACKEND("CPU", CPUPolynomialFactory<scalar_t>)

} // namespace polynomials