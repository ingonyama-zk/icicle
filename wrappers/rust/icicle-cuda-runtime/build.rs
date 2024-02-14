// Based on https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/cudart-sys/build.rs
use std::fs;
use std::path::PathBuf;

fn cuda_include_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "\\include")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/include"
    }
}

fn cuda_lib_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "\\lib\\x64")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/lib64"
    }
}

fn main() {
    let cuda_runtime_api_path = PathBuf::from(cuda_include_path())
        .join("cuda_runtime_api.h")
        .to_string_lossy()
        .to_string();
    println!("cargo:rustc-link-search=native={}", cuda_lib_path());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed={}", cuda_runtime_api_path);

    let bindings = bindgen::Builder::default()
        .header(cuda_runtime_api_path)
        .size_t_is_usize(true)
        .generate_comments(false)
        .layout_tests(false)
        .allowlist_type("cudaError")
        .rustified_enum("cudaError")
        .must_use_type("cudaError")
        // device management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        .allowlist_function("cudaSetDevice")
        // error handling
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
        .allowlist_function("cudaGetLastError")
        // stream management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        .allowlist_function("cudaStreamCreate")
        .allowlist_var("cudaStreamDefault")
        .allowlist_var("cudaStreamNonBlocking")
        .allowlist_function("cudaStreamCreateWithFlags")
        .allowlist_function("cudaStreamDestroy")
        .allowlist_function("cudaStreamQuery")
        .allowlist_function("cudaStreamSynchronize")
        .allowlist_var("cudaEventWaitDefault")
        .allowlist_var("cudaEventWaitExternal")
        .allowlist_function("cudaStreamWaitEvent")
        // memory management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
        .allowlist_function("cudaFree")
        .allowlist_function("cudaMalloc")
        .allowlist_function("cudaMemcpy")
        .allowlist_function("cudaMemcpyAsync")
        .allowlist_function("cudaMemset")
        .allowlist_function("cudaMemsetAsync")
        .rustified_enum("cudaMemcpyKind")
        // Stream Ordered Memory Allocator
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
        .allowlist_function("cudaFreeAsync")
        .allowlist_function("cudaMallocAsync")
        //
        .generate()
        .expect("Unable to generate bindings");

    fs::write(PathBuf::from("src").join("bindings.rs"), bindings.to_string()).expect("Couldn't write bindings!");
}
