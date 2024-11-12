use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    // Construct the path to the deps directory
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));
    let deps_dir = build_dir.join("deps");

    let main_dir = env::current_dir().expect("Failed to get current directory");
    let icicle_src_dir = PathBuf::from(format!("{}/../../../icicle", main_dir.display()));

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed={}", icicle_src_dir.display());

    // Base config
    let mut config = Config::new(format!("{}", icicle_src_dir.display()));
    // cross-compilation
    let xcompile = env::var("XCOMPILE").unwrap_or_else(|_| "true".to_string());
    if xcompile == "true" {
        // Cross-compilation configuration
        let ndk_path = env::var("ANDROID_NDK").expect("ANDROID_NDK not set");
        let abi = env::var("ANDROID_ABI").unwrap_or_else(|_| "arm64-v8a".to_string());
        let api_level = env::var("ANDROID_PLATFORM").unwrap_or_else(|_| "21".to_string());

        config.define("CMAKE_TOOLCHAIN_FILE", format!("{}/build/cmake/android.toolchain.cmake", ndk_path))
           .define("ANDROID_ABI", &abi)
           .define("ANDROID_PLATFORM", format!("android-{}", api_level));
        //    .generator("Ninja");

        println!("Configuring for cross-compilation to Android.");
    }
    // Check if ICICLE_INSTALL_DIR is defined
    let icicle_install_dir = if let Ok(dir) = env::var("ICICLE_INSTALL_DIR") {
        PathBuf::from(dir)
    } else {
        // Define the default install directory to be under the build directory
        PathBuf::from(format!("{}/icicle/", deps_dir.display()))
    };
    config
        .define("HASH", "ON")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);

    // build (or pull and build) cuda backend if feature enabled.
    // Note: this requires access to the repo
    if cfg!(feature = "cuda_backend") {
        config.define("CUDA_BACKEND", "local");
    } else if cfg!(feature = "pull_cuda_backend") {
        config.define("CUDA_BACKEND", "main");
    }
    if cfg!(feature = "metal_backend") {
        config.define("METAL_BACKEND", "local");
    } else if cfg!(feature = "pull_metal_backend") {
        config.define("METAL_BACKEND", "main");
    }

    // Build
    let _ = config
        .build_target("install")
        .build();

    println!("cargo:rustc-link-search={}/lib", icicle_install_dir.display());
    println!("cargo:rustc-link-lib=icicle_hash");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", icicle_install_dir.display()); // Add RPATH linker arguments

    // default backends dir
    // default backends dir
    if cfg!(feature = "cuda_backend")
        || cfg!(feature = "pull_cuda_backend")
        || cfg!(feature = "metal_backend")
        || cfg!(feature = "pull_metal_backend")
    {
        println!(
            "cargo:rustc-env=ICICLE_BACKEND_INSTALL_DIR={}/lib/backend",
            icicle_install_dir.display()
        );
    }
}
