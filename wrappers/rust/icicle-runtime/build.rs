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
    // Check if ICICLE_INSTALL_DIR is defined
    let icicle_install_dir = if let Ok(dir) = env::var("ICICLE_INSTALL_DIR") {
        PathBuf::from(dir)
    } else {
        // Define the default install directory to be under the build directory
        PathBuf::from(format!("{}/icicle/", deps_dir.display()))
    };
    config
        .define("HASH", "OFF")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);
    // check if cross-compilation is required
    let target = std::env::var("TARGET").unwrap();
    if target.contains("android") {
        config.define("BUILD_FOR_ANDROID", "ON");
    } else if target.contains("apple-ios") {
        config.define("BUILD_FOR_IOS", "ON");
        
        // Determine if we're building for simulator or device
        let is_simulator = target.ends_with("-sim");
        if is_simulator {
            config.define("IOS_SIMULATOR", "ON");
            config.define("IOS_DEVICE", "OFF");
        } else {
            config.define("IOS_SIMULATOR", "OFF");
            config.define("IOS_DEVICE", "ON");
        }
    }
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
    if cfg!(feature = "vulkan_backend") {
        config.define("VULKAN_BACKEND", "local");
    } else if cfg!(feature = "pull_vulkan_backend") {
        config.define("VULKAN_BACKEND", "main");
    }

    // Build
    let _ = config
        .build_target("install")
        .build();

    println!("cargo:rustc-link-search={}/lib", icicle_install_dir.display());
    println!("cargo:rustc-link-lib=icicle_device");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", icicle_install_dir.display()); // Add RPATH linker arguments

    // Add iOS-specific linker flags
    if target.contains("apple-ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

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
