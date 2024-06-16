#[cfg(test)]
mod tests {
    use crate::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn get_main_target() -> Device {
        initialize();
        let cuda_device = Device::new("CUDA", 0);

        // if cuda is available use it as main target. Otherwise fallback to CPU.
        if is_device_available(&cuda_device) == eIcicleError::Success {
            return cuda_device;
        }
        Device::new("CPU", 0)
    }

    fn get_ref_target() -> Device {
        initialize();
        let cuda_device = Device::new("CUDA", 0);

        // if cuda is available use CPU as reference target. Otherwise use CPU_REF
        if is_device_available(&cuda_device) == eIcicleError::Success {
            return Device::new("CPU", 0);
        }
        Device::new("CPU_REF", 0)
    }

    fn initialize() {
        INIT.call_once(|| {
            // load backends to process
            assert_eq!(
                load_backend(&env!("DEFAULT_BACKEND_INSTALL_DIR"), true),
                eIcicleError::Success
            );
        });
    }

    #[test]
    fn test_set_device() {
        initialize();

        assert_eq!(set_device(&get_main_target()), eIcicleError::Success);
        assert_eq!(set_device(&get_ref_target()), eIcicleError::Success);
    }
}
