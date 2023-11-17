use cmake::Config;
use std::env::var;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../icicle");

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/../target/{}", cargo_dir, profile);

    Config::new("./icicle")
                .define("BUILD_TESTS", "OFF") //TODO: feature
                // .define("CURVE", "bls12_381")
                .define("CURVE", "bn254")
                // .define("ECNTT_DEFINED", "") //TODO: feature
                .define("LIBRARY_OUTPUT_DIRECTORY", &target_output_dir)
                .define("CMAKE_BUILD_TYPE", "Release")
                .build_target("icicle")
                .build();

    println!("cargo:rustc-link-search={}", &target_output_dir);

    // println!("cargo:rustc-link-lib=icicle");
    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
