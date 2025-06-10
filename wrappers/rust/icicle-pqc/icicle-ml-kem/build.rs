use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    // Get target build directories
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));
    let deps_dir = build_dir.join("deps");

    // Locate icicle source directory
    let main_dir = env::current_dir().expect("Failed to get current directory");
    let icicle_src_dir = main_dir
        .join("../../../../icicle")
        .canonicalize()
        .expect("Failed to canonicalize icicle path");

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed={}", icicle_src_dir.display());

    // Setup CMake build
    let mut config = Config::new(&icicle_src_dir);

    // Set install directory (default or user-specified)
    let icicle_install_dir = env::var("ICICLE_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| deps_dir.join("icicle"));

    config
        .define("PQC", "ON")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir)
        .define("CUDA_PQC_BACKEND", "ON");

    // Build and install the library
    config
        .build_target("install")
        .build();

    let lib_path = icicle_install_dir.join("lib");

    // Link against the installed library
    println!("cargo:rustc-link-search={}", lib_path.display());
    println!("cargo:rustc-link-lib=icicle_pqc");
}
