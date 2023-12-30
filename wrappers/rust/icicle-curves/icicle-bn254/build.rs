use cmake::Config;
use std::env::var;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let out_dir = Config::new("../../../../icicle")
                .define("BUILD_TESTS", "OFF") //TODO: feature
                .define("CURVE", "bn254")
                .define("CMAKE_BUILD_TYPE", "Release")
                .build_target("icicle")
                .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
