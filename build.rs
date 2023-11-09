use std::env::{self, var};

use cmake::Config;

fn main() {
    //TODO: check cargo features selected

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/target/{}", cargo_dir, profile);
    let build_output_dir = format!("{}/build", target_output_dir);

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./icicle");
    println!("cargo:rerun-if-changed=./target/{}", profile); // TODO: without this it ignores manual changes to build folder

    // let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native")); //TODO: pass to cmake
    // let stream_type = env::var("DEFAULT_STREAM").unwrap_or(String::from("legacy"));
    let mut cmake = Config::new("./icicle");
    cmake.define("BUILD_TESTS", "OFF") //TODO: feature
          //.define("LIBRARY_OUTPUT_DIRECTORY", &target_output_dir) //TODO: cmake vars don't work here
          //.define("LIBRARY_OUTPUT_NAME", "libingo_icicle.a");
         .out_dir(&target_output_dir)
         .build_target("icicle");

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
    }

    if cfg!(windows) {
        let target_profile: &str;
        if profile == "release" {
            target_profile = "Release";
        } else {
            target_profile = "Debug";
        }
        
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
