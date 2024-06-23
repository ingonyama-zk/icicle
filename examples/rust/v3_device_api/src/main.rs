use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    runtime, Device,
};

fn main() {
    runtime::load_backend(
        "/home/administrator/users/yuvals/icicle/examples/rust/v3_device_api/target/debug/deps/icicle/lib/backend/cuda",
        true,
    );

    let device = Device::new("CUDA", 0);

    let _cuda_available = runtime::is_device_available(&device);

    runtime::set_device(&device).unwrap();

    let input = vec![1, 2, 3, 4];
    let mut output = vec![0; 4];
    let mut d_mem = DeviceVec::<i32>::device_malloc(input.len()).unwrap();
    d_mem
        .copy_from_host(HostSlice::from_slice(&input))
        .unwrap();
    d_mem
        .copy_to_host(HostSlice::from_mut_slice(&mut output))
        .unwrap();
    assert_eq!(input, output);
    println!("success");
}
