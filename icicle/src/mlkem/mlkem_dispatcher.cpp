#include "icicle/mlkem/mlkem.h"
#include "icicle/backend/backend.h"
#include "icicle/backend/cuda/cuda_backend.h"
#include "icicle/backend/cpu/cpu_backend.h"
#include <memory>

namespace mlkem {

namespace {
    // Singleton dispatcher instance
    class MLKEMDispatcher {
    public:
        static MLKEMDispatcher& get_instance() {
            static MLKEMDispatcher instance;
            return instance;
        }
        
        void set_backend_type(backend::BackendType type) {
            backend_type_ = type;
        }
        
        backend::BackendType get_backend_type() const {
            return backend_type_;
        }
        
        std::shared_ptr<backend::Backend> get_backend() {
            if (!backend_) {
                switch (backend_type_) {
                    case backend::BackendType::CUDA:
                        backend_ = std::make_shared<backend::CUDABackend>();
                        break;
                    case backend::BackendType::CPU:
                        backend_ = std::make_shared<backend::CPUBackend>();
                        break;
                    default:
                        throw std::runtime_error("Unsupported backend type");
                }
            }
            return backend_;
        }
        
    private:
        MLKEMDispatcher() : backend_type_(backend::BackendType::CPU) {}
        backend::BackendType backend_type_;
        std::shared_ptr<backend::Backend> backend_;
    };
}

// Initialize the ML-KEM backend
MLKEMResult<void> init(backend::BackendType backend_type) {
    try {
        MLKEMDispatcher::get_instance().set_backend_type(backend_type);
        return MLKEMResult<void>::success({});
    } catch (const std::exception& e) {
        return MLKEMResult<void>::error(
            MLKEMError::BackendError,
            std::string("Failed to initialize backend: ") + e.what());
    }
}

// Get the current ML-KEM backend
std::shared_ptr<backend::Backend> get_backend() {
    return MLKEMDispatcher::get_instance().get_backend();
}

} // namespace mlkem 