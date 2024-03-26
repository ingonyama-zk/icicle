use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle/curves");
    config
        .define("BUILD_TESTS", "OFF")
        .define("CURVE", "bn254")
        .define("CMAKE_BUILD_TYPE", "Release");

    // Optional Features
    #[cfg(feature = "g2")]
    config.define("G2_DEFINED", "ON");

    // Build
    let out_dir = config
        .build_target("icicle_bn254")
        .build();

    println!("cargo:rustc-link-search={}/build/bn254", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
