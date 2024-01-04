use std::env::{self, var};

use cmake::Config;

fn main() {
    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/target/{}", cargo_dir, profile);
    let build_output_dir = format!("{}/build", target_output_dir);

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./icicle");
    println!("cargo:rerun-if-changed=./target/{}", profile); // without this it ignores manual changes to build folder

    let mut cmake = Config::new("./icicle");
    cmake
        .define("BUILD_TESTS", "OFF")
        .out_dir(&target_output_dir)
        .build_target("icicle");

    let target_profile: &str = if profile == "release" { "Release" } else { "Debug" };

    cmake.define("CMAKE_BUILD_TYPE", "Release");

    if cfg!(feature = "g2") {
        cmake.define("G2_DEFINED", "");
    }

    cmake.build();

    if cfg!(unix) {
        if let Ok(cuda_path) = var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    } else if cfg!(windows) {
        let build_output_dir_cmake = format!("{}/{}", build_output_dir, target_profile);

        println!("cargo:rustc-link-search={}", &build_output_dir_cmake);
    }

    println!("cargo:rustc-link-search={}", &build_output_dir);
    println!("cargo:rustc-link-search={}", &target_output_dir);
    println!("cargo:rustc-link-lib=ingo_icicle");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
