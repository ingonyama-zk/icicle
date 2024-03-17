use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle");
    config
        .define("BUILD_TESTS", "OFF")
        .define("CURVE", "grumpkin")
        .define("CMAKE_BUILD_TYPE", "Release");

    #[cfg(feature = "devmode")]
    config.define("DEVMODE", "ON");

    // Build
    let out_dir = config
        .build_target("icicle")
        .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_grumpkin");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
