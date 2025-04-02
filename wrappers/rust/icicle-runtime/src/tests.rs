#[cfg(test)]
mod tests {
    use crate::config::ConfigExtension;
    use crate::memory::{DeviceVec, HostSlice};
    use crate::stream::IcicleStream;
    use crate::test_utilities;
    use crate::*;
    use std::sync::Once;
    use std::thread;

    static INIT: Once = Once::new();

    pub fn initialize() {
        INIT.call_once(move || {
            test_utilities::test_load_and_init_devices();
            // init domain for both devices
            test_utilities::test_set_ref_device();

            test_utilities::test_set_main_device();
        });
        test_utilities::test_set_main_device();
    }

    #[test]
    fn test_set_device() {
        initialize();

        test_utilities::test_set_main_device();
        test_utilities::test_set_ref_device();
    }

    #[test]
    fn test_get_device_count() {
        initialize();

        test_utilities::test_set_main_device();
        let device_count = get_device_count().unwrap();
        assert!(device_count > 0);
        for device_id in 0..device_count {
            test_utilities::test_set_main_device_with_id(device_id); // This fails if not available
        }
        test_utilities::test_set_ref_device();
        let device_count = get_device_count().unwrap();
        assert!(device_count > 0);
    }

    #[test]
    fn test_set_default_device() {
        initialize();

        // block scope is necessary in order to free the mutex lock
        // to be used by the spawned thread
        let outer_thread_id = thread::current().id();
        {
            let main_device = test_utilities::TEST_MAIN_DEVICE
                .lock()
                .unwrap();
            set_default_device(&main_device).unwrap();

            let active_device = get_active_device().unwrap();
            assert_eq!(*main_device, active_device);
        }

        let handle = thread::spawn(move || {
            let inner_thread_id = thread::current().id();
            assert_ne!(outer_thread_id, inner_thread_id);

            let active_device = get_active_device().unwrap();
            let main_device = test_utilities::TEST_MAIN_DEVICE
                .lock()
                .unwrap();
            assert_eq!(*main_device, active_device);
        });

        let _ = handle.join();
    }

    #[test]
    fn test_sync_memory_copy() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // copy from input_host -> device --> output_host and compare
        let mut d_mem = DeviceVec::device_malloc(input.len()).unwrap();
        d_mem
            .copy_from_host(HostSlice::from_slice(&input))
            .unwrap();
        d_mem
            .copy_to_host(HostSlice::from_mut_slice(&mut output))
            .unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_async_memory_copy() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // ASYNC copy from input_host -> device --> output_host and compare
        let mut stream = IcicleStream::create().unwrap();

        let mut d_mem = DeviceVec::device_malloc_async(input.len(), &stream).unwrap();
        d_mem
            .copy_from_host_async(HostSlice::from_slice(&input), &stream)
            .unwrap();
        d_mem
            .copy_to_host_async(HostSlice::from_mut_slice(&mut output), &stream)
            .unwrap();
        stream
            .synchronize()
            .unwrap();
        stream
            .destroy()
            .unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_device_to_device_copy() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // Copy from host to first device
        let mut d_mem1 = DeviceVec::device_malloc(input.len()).unwrap();
        d_mem1
            .copy_from_host(HostSlice::from_slice(&input))
            .unwrap();

        // Copy from first device to second device
        let mut d_mem2 = DeviceVec::device_malloc(input.len()).unwrap();
        d_mem2
            .copy_from_device(&d_mem1)
            .unwrap();

        // Copy back to host and verify
        d_mem2
            .copy_to_host(HostSlice::from_mut_slice(&mut output))
            .unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_device_to_device_copy_async() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);

        // Create stream for async operations
        let mut stream = IcicleStream::create().unwrap();

        // Copy from host to first device
        let mut d_mem1 = DeviceVec::device_malloc_async(input.len(), &stream).unwrap();
        d_mem1
            .copy_from_host_async(HostSlice::from_slice(&input), &stream)
            .unwrap();

        // Copy from first device to second device
        let mut d_mem2 = DeviceVec::device_malloc_async(input.len(), &stream).unwrap();
        d_mem2
            .copy_from_device_async(&d_mem1, &stream)
            .unwrap();

        // Copy back to host and verify
        d_mem2
            .copy_to_host_async(HostSlice::from_mut_slice(&mut output), &stream)
            .unwrap();

        // Synchronize and cleanup
        stream
            .synchronize()
            .unwrap();
        stream
            .destroy()
            .unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_get_device_props() {
        initialize();
        let device = Device::new("CUDA", 0);

        if is_device_available(&device) {
            set_device(&device).unwrap();

            let device_props = get_device_properties().unwrap();
            assert_eq!(device_props.using_host_memory, false); // for "cuda"
        }
    }

    #[test]
    fn test_config_extension() {
        let config_ext = ConfigExtension::new();

        config_ext.set_int("example_int", 42);
        config_ext.set_bool("example_bool", true);

        assert_eq!(config_ext.get_int("example_int"), 42);
        assert_eq!(config_ext.get_bool("example_bool"), true);
    }
}
