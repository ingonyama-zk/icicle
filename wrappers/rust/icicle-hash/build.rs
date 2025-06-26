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
        .define("HASH", "ON")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);
    
    // check if cross-compilation is required
    // ====================================================================
    // ** NEW ANDROID-AWARE LOGIC **
    // ====================================================================
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "android" {
        println!("cargo:warning=Configuring for Android cross-compilation");

        let ndk_home = env::var("ANDROID_NDK_HOME")
            .expect("ANDROID_NDK_HOME is not set. Please set it to your Android NDK root.");

        // let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        // let android_abi = match target_arch.as_str() {
        //     "aarch64" => "arm64-v8a",
        //     "arm" => "armeabi-v7a",
        //     "x86_64" => "x86_64",
        //     "x86" => "x86",
        //     _ => panic!("Unsupported Android architecture: {}", target_arch),
        // };

        // let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk_home);

        config
            .define("BUILD_FOR_ANDROID", "ON")
            .define("ANDROID_NDK_HOME", ndk_home);
        // .define("CMAKE_TOOLCHAIN_FILE", toolchain_file);
        // .define("ANDROID_ABI", android_abi)
    } else {
        println!("cargo:warning=Configuring for non-Android platform");
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
    // Linking should be done at the application level, not here
    if target_os != "android" {
        println!("cargo:rustc-link-lib=icicle_hash");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", icicle_install_dir.display()); // Add RPATH linker arguments
    }

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
