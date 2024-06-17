#[cfg(test)]
mod tests {
    use crate::memory::{DeviceVec, HostSlice};
    use crate::stream::IcicleStream;
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

    #[test]
    fn test_sync_memory_copy() {
        initialize();
        assert_eq!(set_device(&get_main_target()), eIcicleError::Success);

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // copy from input_host -> device --> output_host and compare
        let mut d_mem = DeviceVec::device_malloc(input.len()).unwrap();
        d_mem.copy_from_host(HostSlice::from_slice(&input));
        d_mem.copy_to_host(HostSlice::from_mut_slice(&mut output));
        assert_eq!(input, output);
    }

    #[test]
    fn test_async_memory_copy() {
        initialize();
        assert_eq!(set_device(&get_main_target()), eIcicleError::Success);

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // ASYNC copy from input_host -> device --> output_host and compare
        let stream = IcicleStream::create().unwrap();

        let mut d_mem = DeviceVec::device_malloc_async(input.len(), &stream).unwrap();
        d_mem.copy_from_host_async(HostSlice::from_slice(&input), &stream);
        d_mem.copy_to_host_async(HostSlice::from_mut_slice(&mut output), &stream);
        stream.synchronize();
        assert_eq!(input, output);
    }

    #[test]
    fn test_get_available_memory() {
        initialize();
        assert_eq!(set_device(&get_main_target()), eIcicleError::Success);

        let (total, free) = get_available_memory().unwrap();
        assert!(total > 0 && free > 0 && total >= free);
    }

    #[test]
    fn test_get_device_props() {
        initialize();
        let device = Device::new("CUDA", 0);

        if is_device_available(&device) == eIcicleError::Success {
            assert_eq!(set_device(&device), eIcicleError::Success);

            let device_props = get_device_properties().unwrap();
            assert_eq!(device_props.using_host_memory, false); // for "cuda"
        }
    }
}
