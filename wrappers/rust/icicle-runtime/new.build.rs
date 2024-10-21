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
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir)
        .define("CMAKE_TOOLCHAIN_FILE", format!("{}/build/cmake/android.toolchain.cmake", env::var("ANDROID_NDK_HOME").expect("ANDROID_NDK_HOME is not set")))
        .define("CMAKE_SYSTEM_NAME", "Android")
        .define("CMAKE_SYSTEM_PROCESSOR", "aarch64")
        .define("CMAKE_MAKE_PROGRAM", "/usr/bin/make");

    // build (or pull and build) cuda backend if feature enabled.
    // Note: this requires access to the repo
    if cfg!(feature = "cuda_backend") {
        config.define("CUDA_BACKEND", "local");
    } else if cfg!(feature = "pull_cuda_backend") {
        config.define("CUDA_BACKEND", "main");
    }

    // Build
    let _ = config
        .build_target("install")
        .build();
}