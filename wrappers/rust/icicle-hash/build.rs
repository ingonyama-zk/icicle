use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../icicle");

    // Base config
    let mut config = Config::new("../../../icicle/");
    config.define("CMAKE_BUILD_TYPE", "Release");
    config.define("BUILD_HASH", "ON");

    // Build
    let out_dir = config
        .build_target("icicle_hash")
        .build();

    println!("cargo:rustc-link-search={}/build/src/hash/", out_dir.display());
    println!("cargo:rustc-link-lib=ingo_hash");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
