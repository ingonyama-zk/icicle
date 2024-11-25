#[cfg(test)]
mod tests {
    use crate::config::ConfigExtension;
    use crate::memory::{DeviceVec, HostSlice};
    use crate::stream::IcicleStream;
    use crate::test_utilities;
    use crate::*;
    use std::sync::Once;

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
