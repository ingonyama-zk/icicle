#[cfg(test)]
mod tests {
    use crate::config::ConfigExtension;
    use crate::memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut};
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
        let d_mem = DeviceVec::<u8>::from_host_slice(&input);
        output = d_mem.to_host_vec();
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

        let mut d_mem = DeviceVec::<u8>::device_malloc_async(input.len(), &stream).unwrap();
        d_mem
            .copy_from_host_async(input.into_slice(), &stream)
            .unwrap();
        d_mem
            .copy_to_host_async(output.into_slice_mut(), &stream)
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
    fn test_copy() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let input2 = vec![0; input.len()];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);
        let h_input = input.into_slice();
        let h_input2 = input2.into_slice();

        // H -> D -> D -> H
        {
            let h_output = output.into_slice_mut();
            let mut d_mem1 = DeviceVec::<u8>::malloc(input.len());
            d_mem1
                .copy(h_input)
                .unwrap();

            let mut d_mem2 = DeviceVec::<u8>::malloc(input.len() * 5);
            d_mem2
                .copy(&d_mem1)
                .unwrap();

            h_output
                .copy(&d_mem2[0..input.len()])
                .unwrap();
            assert_eq!(input, output);
        }

        // H -> H
        {
            let h_output = output.into_slice_mut();
            h_output
                .copy(h_input2)
                .unwrap();
            assert_eq!(input2, output);
        }
    }

    #[test]
    fn test_copy_async() {
        initialize();
        test_utilities::test_set_main_device();

        let input = vec![1, 2, 3, 4];
        let input2 = vec![0; input.len()];
        let mut output = vec![0; input.len()];
        assert_ne!(input, output);
        let h_input = input.into_slice();
        let h_input2 = input2.into_slice();

        // H -> D -> D -> H
        {
            let mut stream = IcicleStream::create().unwrap();
            let h_output = output.into_slice_mut();

            let mut d_mem1 = DeviceVec::<u8>::malloc(input.len());
            d_mem1
                .copy_async(h_input, &stream)
                .unwrap();

            let mut d_mem2 = DeviceVec::<u8>::malloc(input.len() * 5);
            d_mem2
                .copy_async(&d_mem1, &stream)
                .unwrap();

            h_output
                .copy_async(&d_mem2[0..input.len()], &stream)
                .unwrap();

            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            assert_eq!(input, output);
        }

        // H -> H
        {
            let mut stream = IcicleStream::create().unwrap();
            let h_output = output.into_slice_mut();

            h_output
                .copy_async(h_input2, &stream)
                .unwrap();

            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            assert_eq!(input2, output);
        }
    }

    #[test]
    fn test_get_device_props() {
        initialize();
        let device = Device::new("CUDA", 0);

        if is_device_available(&device) {
            set_device(&device).unwrap();

            let device_props = get_device_properties().unwrap();
            assert!(!device_props.using_host_memory); // for "cuda"
        }
    }

    #[test]
    fn test_config_extension() {
        let config_ext = ConfigExtension::new();

        config_ext.set_int("example_int", 42);
        config_ext.set_bool("example_bool", true);

        assert_eq!(config_ext.get_int("example_int"), 42);
        assert!(config_ext.get_bool("example_bool"));
    }
    #[test]
    fn test_memset() {
        initialize();
        test_utilities::test_set_main_device();
        let size = 1 << 10;
        let val = 42;
        let expected = vec![val; size];
        let mut device_vec = DeviceVec::<u8>::malloc(size);
        device_vec
            .memset(val, size)
            .unwrap();

        let host_vec = device_vec.to_host_vec();

        assert_eq!(host_vec, expected);
    }

    #[test]
    fn test_memset_async() {
        initialize();
        test_utilities::test_set_main_device();
        let size = 1 << 10;
        let val = 42;
        let expected = vec![val; size >> 1];
        let stream = IcicleStream::create().unwrap();
        let mut device_vec = DeviceVec::<u8>::malloc(size);
        device_vec.as_mut_slice()[1..size >> 1] // set only part of the slice
            .memset_async(val, (size >> 1) - 1, &stream)
            .unwrap();
        stream
            .synchronize()
            .unwrap();

        let host_slice = device_vec.to_host_vec();

        assert_eq!(host_slice.as_slice()[1..size >> 1], expected[1..]);
    }
}
