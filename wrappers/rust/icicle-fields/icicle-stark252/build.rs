use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle/");
    config
        .define("FIELD", "stark252")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("EXT_FIELD", "OFF");

    // Build
    let out_dir = config
        .build_target("icicle_field")
        .build();

    println!("cargo:rustc-link-search={}/build/lib", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_field_stark252");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
