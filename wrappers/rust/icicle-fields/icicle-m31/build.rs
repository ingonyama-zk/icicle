use cmake::Config;
use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle/");
    config
        .define("FIELD", "m31")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("EXT_FIELD", "ON");

    if let Ok(cuda_arch) = env::var("CUDA_ARCH") {
        config.define("CUDA_ARCH", &cuda_arch);
    }

    // Build
    let out_dir = config
        .build_target("icicle_field")
        .build();

    println!("cargo:rustc-link-search={}/build/lib", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_field_m31");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
